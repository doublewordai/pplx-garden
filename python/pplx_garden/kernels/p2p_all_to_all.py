import os
import pickle
import weakref
from dataclasses import dataclass
from typing import Any, Optional

import torch
from typing_extensions import override

from pplx_garden.distributed import ParallelGroup
from pplx_garden.fabric_lib import (
    DomainAddress,
    MemoryRegionDescriptor,
    MemoryRegionHandle,
    TransferEngine,
)
from pplx_garden.kernels.all_to_all import AllToAllKernel
from pplx_garden.native.cumem import (
    CUMemAllocHandle,
    CUMemExportHandle,
    CUMemHandleKind,
    CUMemMapping,
)
from pplx_garden.native.p2p_all_to_all import AllToAllContext
from pplx_garden.utils import logging_utils
from pplx_garden.utils.math import ceil_div, round_up

logger = logging_utils.get_logger(__name__)

_PAGE_SIZE = 4096

_CUDA_GRAPH_KERNELS: weakref.WeakSet["P2PAllToAll"] = weakref.WeakSet()


def reset_all_cuda_graph_capture_slots() -> None:
    for kernel in list(_CUDA_GRAPH_KERNELS):
        kernel.reset_cuda_graph_capture_slots()


def has_cuda_graph_capture_kernels() -> bool:
    return any(True for _ in _CUDA_GRAPH_KERNELS)


@dataclass
class _RdmaRankData:
    address: bytes
    num_routed_descs: list[bytes]
    recv_buffer_descs: list[bytes]


@dataclass
class _NVLRankData:
    sync_fds: list[CUMemExportHandle]
    send_fds: list[CUMemExportHandle]
    recv_fds: list[CUMemExportHandle]


@dataclass
class _NVLRankMapping:
    sync_mapping: CUMemMapping
    send_mapping: CUMemMapping
    recv_mapping: CUMemMapping


@dataclass
class P2PDispatchHandle:
    kernel: "P2PAllToAll"
    out_expert_num_tokens: torch.Tensor
    out_expert_x: torch.Tensor
    out_expert_x_scale: Optional[torch.Tensor]
    dp_x: torch.Tensor
    dp_x_scale: Optional[torch.Tensor]
    indices: torch.Tensor
    weights: torch.Tensor
    bound_m: Optional[torch.Tensor]
    send_done_event: torch.cuda.Event
    _slot: int
    recv_done_event: Optional[torch.cuda.Event] = None
    recv_done: bool = False

    def recv(self) -> None:
        if self.recv_done:
            return
        torch.cuda.current_stream(self.dp_x.device).wait_event(self.send_done_event)
        self.kernel.dispatch(
            out_expert_num_tokens=self.out_expert_num_tokens,
            out_expert_x=self.out_expert_x,
            out_expert_x_scale=self.out_expert_x_scale,
            dp_x=self.dp_x,
            dp_x_scale=self.dp_x_scale,
            indices=self.indices,
            weights=self.weights,
            bound_m=self.bound_m,
            _slot=self._slot,
            do_send=False,
            do_recv=True,
        )
        self.recv_done_event = torch.cuda.Event()
        self.recv_done_event.record(torch.cuda.current_stream(self.dp_x.device))
        self.recv_done = True

    def wait_recv_done(self) -> None:
        self.recv()
        assert self.recv_done_event is not None
        torch.cuda.current_stream(self.dp_x.device).wait_event(
            self.recv_done_event
        )


@dataclass
class P2PCombineHandle:
    kernel: "P2PAllToAll"
    dispatch_handle: P2PDispatchHandle
    out_tokens: torch.Tensor
    expert_y: torch.Tensor
    bound_m: Optional[torch.Tensor]
    accumulate: bool
    send_done_event: torch.cuda.Event
    _slot: int
    recv_done_event: Optional[torch.cuda.Event] = None
    recv_done: bool = False

    def recv(self) -> None:
        if self.recv_done:
            return
        torch.cuda.current_stream(self.out_tokens.device).wait_event(
            self.send_done_event
        )
        self.kernel.combine(
            out_tokens=self.out_tokens,
            indices=self.dispatch_handle.indices,
            weights=self.dispatch_handle.weights,
            expert_y=self.expert_y,
            bound_m=self.bound_m,
            _slot=self._slot,
            do_send=False,
            do_recv=True,
            accumulate=self.accumulate,
        )
        self.recv_done_event = torch.cuda.Event()
        self.recv_done_event.record(torch.cuda.current_stream(self.out_tokens.device))
        self.recv_done = True

    def wait_recv_done(self) -> None:
        self.recv()
        assert self.recv_done_event is not None
        torch.cuda.current_stream(self.out_tokens.device).wait_event(
            self.recv_done_event
        )


class P2PAllToAll(AllToAllKernel):
    def __init__(
        self,
        *,
        max_num_tokens: int,
        num_experts: int,
        expert_padding: int,
        hidden_dim: int,
        hidden_dim_scale: Optional[int],
        in_dtype: torch.dtype,
        out_dtype: torch.dtype,
        scale_dtype: Optional[torch.dtype],
        num_experts_per_token: int,
        nets_per_gpu: int,
        max_private_tokens: Optional[int],
        device: torch.device,
        dp_group: Optional[ParallelGroup],
        node_group: Optional[ParallelGroup],
        global_group: ParallelGroup,
        imm_base: int = 0x80000000,
        transfer_engine: Optional[TransferEngine] = None,
        worker_cpu: Optional[int] = None,
        max_tokens_per_expert: Optional[int] = None,
    ) -> None:
        self._hidden_dim = hidden_dim
        self._hidden_dim_scale = hidden_dim_scale
        self._num_experts_per_token = num_experts_per_token
        self._in_dtype = in_dtype
        self._out_dtype = out_dtype
        self._scale_dtype = scale_dtype
        self._device = device
        self._global_group = global_group
        self._max_tokens_per_expert = max_tokens_per_expert
        self._handle_kind = CUMemHandleKind.FileDescriptor
        self._num_slots = int(os.environ.get("PPLX_GARDEN_NUM_SLOTS", "2"))
        if self._num_slots < 1:
            raise ValueError("PPLX_GARDEN_NUM_SLOTS must be >= 1")
        self._capture_free_slots = list(range(self._num_slots))
        self._sync_slot: Optional[int] = None
        _CUDA_GRAPH_KERNELS.add(self)

        # Determine the number of local experts.
        self._node_group: Optional[ParallelGroup]
        if dp_group is not None:
            self._node_group = node_group or dp_group
            self._dp_size = dp_group.size
        else:
            self._node_group = node_group
            self._dp_size = 1

        rank = global_group.rank
        world_size = global_group.size
        num_dp_groups = world_size // self._dp_size
        self._num_local_experts = ceil_div(num_experts, num_dp_groups)

        # Determine the size of the recv buffers.
        avg_tokens_per_expert = int(
            ceil_div(max_num_tokens * num_experts_per_token, num_experts) * 1.2
        )

        if max_private_tokens is None:
            max_private_tokens = avg_tokens_per_expert * self._num_local_experts
        assert max_private_tokens >= 0

        num_tokens = max_num_tokens * num_dp_groups
        max_recv_tokens = max_private_tokens * num_dp_groups + round_up(
            max(
                min(
                    num_tokens * num_experts_per_token
                    + self._num_local_experts * (expert_padding - 1),
                    num_tokens * self._num_local_experts,
                ),
                self._num_local_experts * expert_padding,
            ),
            expert_padding,
        )

        self._transfer_engine: Optional[TransferEngine] = None
        self._all_to_all: Optional[AllToAllContext] = None

        # Detect topology and identify NICs and CPUs.
        system_topo = TransferEngine.detect_topology()

        device_groups = [
            group for group in system_topo if group.cuda_device == device.index
        ]
        if len(device_groups) == 1:
            group = device_groups[0]
        elif device.index is not None and 0 <= device.index < len(system_topo):
            group = system_topo[device.index]
            logger.warning(
                "Falling back to topology group %s for cuda:%s; detected CUDA groups are %s",
                device.index,
                device.index,
                [group.cuda_device for group in system_topo],
            )
        else:
            msg = (
                f"Cannot identify topology group for cuda:{device.index}; "
                f"detected CUDA groups are {[group.cuda_device for group in system_topo]}"
            )
            raise RuntimeError(msg)

        if len(group.cpus) < 2:
            msg = f"Not enough CPUs in device group for cuda:{device.index}"
            raise RuntimeError(msg)

        default_worker_cpu, domain_cpu, uvm_cpu, *_ = group.cpus
        if worker_cpu is None:
            worker_cpu = default_worker_cpu
        domains = group.domains[:nets_per_gpu]

        # Build or reuse the transfer engine. Multiple live P2PAllToAll contexts
        # should not each spawn their own fabric workers for the same GPU/NIC
        # resources; that path is extremely slow on the CXI backend.
        self._owns_transfer_engine = transfer_engine is None
        if transfer_engine is None:
            builder = TransferEngine.builder()
            builder.add_gpu_domains(group.cuda_device, domains, domain_cpu, uvm_cpu)
            transfer_engine = builder.build()
        self._transfer_engine = transfer_engine

        num_slots = self._num_slots

        # Allocate and register per-slot buffers for per-expert routed counts on
        # the host. These are small, but they must not be overwritten while a
        # previous microbatch's worker is still consuming its routing table.
        self._num_routed_buffers = [
            torch.empty(
                (
                    round_up(
                        num_dp_groups * num_experts * torch.uint32.itemsize,
                        _PAGE_SIZE,
                    ),
                ),
                dtype=torch.uint8,
                pin_memory=True,
            ).view(torch.uint32)
            for _ in range(num_slots)
        ]
        num_routed_mrs: list[MemoryRegionHandle] = []
        num_routed_descs: list[MemoryRegionDescriptor] = []
        for num_routed_buffer in self._num_routed_buffers:
            num_routed_mr, num_routed_desc = self._transfer_engine.register_tensor(
                num_routed_buffer
            )
            num_routed_mrs.append(num_routed_mr)
            num_routed_descs.append(num_routed_desc)

        # Allocate a a buffer to send from.
        token_dim_dispatch = round_up(hidden_dim * in_dtype.itemsize, 16) + 16
        if hidden_dim_scale is not None or scale_dtype is not None:
            assert scale_dtype is not None
            assert hidden_dim_scale is not None
            token_dim_dispatch += round_up(hidden_dim_scale * scale_dtype.itemsize, 16)

            # TODO: support other scale dtypes
            assert scale_dtype == torch.float32

        token_dim_combine = round_up(hidden_dim * out_dtype.itemsize, 16)
        token_dim = max(token_dim_dispatch, token_dim_combine)

        send_buffer_bytes = round_up(max_recv_tokens * token_dim, _PAGE_SIZE)
        recv_buffer_bytes = round_up(max_recv_tokens * token_dim, _PAGE_SIZE)

        self._send_buffer_handles = [
            CUMemAllocHandle(send_buffer_bytes, self._device, self._handle_kind)
            for _ in range(num_slots)
        ]
        self._send_buffer_mappings = [
            handle.map(self._device) for handle in self._send_buffer_handles
        ]
        send_buffer_mrs: list[MemoryRegionHandle] = []
        for mapping in self._send_buffer_mappings:
            send_buffer_mr, _send_buffer_desc = self._transfer_engine.register_tensor(
                mapping.to_tensor((send_buffer_bytes,), torch.uint8)
            )
            send_buffer_mrs.append(send_buffer_mr)

        self._recv_buffer_handles = [
            CUMemAllocHandle(recv_buffer_bytes, self._device, self._handle_kind)
            for _ in range(num_slots)
        ]
        self._recv_buffer_mappings = [
            handle.map(self._device) for handle in self._recv_buffer_handles
        ]
        recv_buffer_mrs: list[MemoryRegionHandle] = []
        recv_buffer_descs: list[MemoryRegionDescriptor] = []
        for mapping in self._recv_buffer_mappings:
            recv_buffer_mr, recv_buffer_desc = self._transfer_engine.register_tensor(
                mapping.to_tensor((recv_buffer_bytes,), torch.uint8)
            )
            recv_buffer_mrs.append(recv_buffer_mr)
            recv_buffer_descs.append(recv_buffer_desc)

        # Exchange NVLink buffers.
        self._nvl_mappings: list[list[_NVLRankMapping]] = []
        sync_ptrs: list[list[int]] = [[] for _ in range(num_slots)]
        send_ptrs: list[list[int]] = [[] for _ in range(num_slots)]
        recv_ptrs: list[list[int]] = [[] for _ in range(num_slots)]
        if self._node_group is not None:
            logger.info(
                "Setting up RDMA (%d) + NVLink (%d)",
                global_group.size,
                self._node_group.size,
            )
            self._sync_buffer_handles = [
                CUMemAllocHandle(
                    torch.uint32.itemsize * self._node_group.size * 2,
                    self._device,
                    self._handle_kind,
                )
                for _ in range(num_slots)
            ]
            sync_mappings = [
                handle.map(self._device) for handle in self._sync_buffer_handles
            ]
            for sync_mapping in sync_mappings:
                sync_mapping.to_tensor(
                    (self._node_group.size * 2,),
                    torch.uint32,
                ).fill_(0)

            local_handle = _NVLRankData(
                sync_fds=[handle.export() for handle in self._sync_buffer_handles],
                send_fds=[handle.export() for handle in self._send_buffer_handles],
                recv_fds=[handle.export() for handle in self._recv_buffer_handles],
            )
            handles = self._node_group.all_gather_object(pickle.dumps(local_handle))

            self._nvl_mappings = [[] for _ in range(num_slots)]
            for peer, h in enumerate(handles):
                if peer == self._node_group.rank:
                    for slot in range(num_slots):
                        self._nvl_mappings[slot].append(
                            _NVLRankMapping(
                                sync_mapping=sync_mappings[slot],
                                send_mapping=self._send_buffer_mappings[slot],
                                recv_mapping=self._recv_buffer_mappings[slot],
                            )
                        )
                else:
                    assert h is not None
                    peer_data = pickle.loads(h)
                    assert isinstance(peer_data, _NVLRankData)
                    if (
                        len(peer_data.sync_fds) != num_slots
                        or len(peer_data.send_fds) != num_slots
                        or len(peer_data.recv_fds) != num_slots
                    ):
                        raise RuntimeError(
                            "Peer NVLink handle slot count mismatch: "
                            f"expected {num_slots}, got sync={len(peer_data.sync_fds)} "
                            f"send={len(peer_data.send_fds)} recv={len(peer_data.recv_fds)}"
                        )
                    for slot in range(num_slots):
                        self._nvl_mappings[slot].append(
                            _NVLRankMapping(
                                sync_mapping=peer_data.sync_fds[slot]
                                .bind()
                                .map(self._device),
                                send_mapping=peer_data.send_fds[slot]
                                .bind()
                                .map(self._device),
                                recv_mapping=peer_data.recv_fds[slot]
                                .bind()
                                .map(self._device),
                            )
                        )
                    del peer_data

            self._node_group.barrier()
            del local_handle

            node_size = self._node_group.size
            for slot in range(num_slots):
                for i in range(node_size):
                    recv_ptrs[slot].append(
                        self._nvl_mappings[slot][i].recv_mapping.data_ptr()
                    )
                    send_ptrs[slot].append(
                        self._nvl_mappings[slot][i].send_mapping.data_ptr()
                    )
                    sync_ptrs[slot].append(
                        self._nvl_mappings[slot][i].sync_mapping.data_ptr()
                    )
        else:
            logger.info("Setting up RDMA (%d)", global_group.size)
            node_size = 1

        # Collect the metadata associated with all ranks.
        gathered_rank_data = global_group.all_gather_object(
            _RdmaRankData(
                address=self._transfer_engine.main_address.as_bytes(),
                num_routed_descs=[desc.as_bytes() for desc in num_routed_descs],
                recv_buffer_descs=[desc.as_bytes() for desc in recv_buffer_descs],
            )
        )
        ranks = [
            (
                DomainAddress.from_bytes(data.address),
                [
                    MemoryRegionDescriptor.from_bytes(desc)
                    for desc in data.num_routed_descs
                ],
                [
                    MemoryRegionDescriptor.from_bytes(desc)
                    for desc in data.recv_buffer_descs
                ],
            )
            for data in gathered_rank_data
        ]

        # Set up the all-to-all context.
        self._all_to_all = AllToAllContext.create(
            hidden_dim=hidden_dim,
            hidden_dim_scale=hidden_dim_scale,
            in_elemsize=in_dtype.itemsize,
            out_elemsize=out_dtype.itemsize,
            out_dtype=out_dtype,
            scale_elemsize=scale_dtype.itemsize if scale_dtype else None,
            max_num_tokens=max_num_tokens,
            max_recv_tokens=max_recv_tokens,
            max_tokens_per_expert=max_tokens_per_expert,
            max_private_tokens=max_private_tokens,
            num_experts=num_experts,
            expert_padding=expert_padding,
            num_experts_per_token=num_experts_per_token,
            rank=rank,
            dp_size=self._dp_size,
            node_size=node_size,
            world_size=world_size,
            num_routed_ptrs=[buf.data_ptr() for buf in self._num_routed_buffers],
            num_routed_mrs=num_routed_mrs,
            send_buffer_ptrs=[m.data_ptr() for m in self._send_buffer_mappings],
            send_buffer_mrs=send_buffer_mrs,
            recv_buffer_ptrs=[m.data_ptr() for m in self._recv_buffer_mappings],
            recv_buffer_mrs=recv_buffer_mrs,
            sync_ptrs=sync_ptrs,
            send_ptrs=send_ptrs,
            recv_ptrs=recv_ptrs,
            device=device.index,
            imm_base=imm_base,
            ranks=ranks,
            transfer_engine=self._transfer_engine,
            worker_cpu=worker_cpu,
            num_slots=num_slots,
        )

        # Ensure that all ranks start the workers threads and registered imm callbacks.
        global_group.barrier()

    def reset_cuda_graph_capture_slots(self) -> None:
        # This is capture bookkeeping only. Runtime slot lifetime is owned by
        # the Rust worker, which releases a slot after combine-recv and the
        # final combine barrier.
        self._capture_free_slots = list(range(self._num_slots))

    def _acquire_cuda_graph_capture_slot(self) -> int:
        if not self._capture_free_slots:
            raise RuntimeError(
                "PPLX Garden CUDA graph capture needs more slots. "
                "A dispatch was captured before an earlier combine released "
                f"one of the {self._num_slots} slots."
            )
        return self._capture_free_slots.pop(0)

    def _release_cuda_graph_capture_slot(self, slot: int) -> None:
        if slot not in self._capture_free_slots:
            self._capture_free_slots.append(slot)

    @override
    def dispatch(
        self,
        out_expert_num_tokens: torch.Tensor,
        out_expert_x: torch.Tensor,
        out_expert_x_scale: Optional[torch.Tensor],
        dp_x: torch.Tensor,
        dp_x_scale: Optional[torch.Tensor],
        indices: torch.Tensor,
        weights: torch.Tensor,
        bound_m: Optional[torch.Tensor] = None,
        do_send: bool = True,
        do_recv: bool = True,
        _slot: Optional[int] = None,
    ) -> None:
        assert self._all_to_all is not None
        assert do_send or do_recv
        all_to_all = self._all_to_all

        num_tokens, _ = dp_x.shape

        # Verify the output count buffer.
        assert out_expert_num_tokens.shape == (self._num_local_experts,)
        assert out_expert_num_tokens.stride(0) == 1
        assert out_expert_num_tokens.dtype == torch.int32
        out_expert_num_tokens_ptr = out_expert_num_tokens.data_ptr()

        # Verify the output token buffer.
        out_expert_x = self._flatten_batched_experts(
            out_expert_x,
            dtype=self._in_dtype,
            hidden_dim=self._hidden_dim,
            name="out_expert_x",
        )
        out_x_ptr = out_expert_x.data_ptr()
        out_x_stride = out_expert_x.stride(0) * out_expert_x.dtype.itemsize

        # Verify the output scale buffer.
        out_x_scale_ptr: Optional[int]
        out_x_scale_stride_elem: Optional[int]
        out_x_scale_stride_token: Optional[int]
        if out_expert_x_scale is not None:
            assert self._scale_dtype is not None
            assert self._hidden_dim_scale is not None
            out_expert_x_scale = self._flatten_batched_experts(
                out_expert_x_scale,
                dtype=self._scale_dtype,
                hidden_dim=self._hidden_dim_scale,
                name="out_expert_x_scale",
            )
            assert out_expert_x_scale.dtype == self._scale_dtype
            out_x_scale_ptr = out_expert_x_scale.data_ptr()
            out_x_scale_stride_elem = out_expert_x_scale.stride(1)
            out_x_scale_stride_token = out_expert_x_scale.stride(0)
        else:
            out_x_scale_ptr = None
            out_x_scale_stride_elem = None
            out_x_scale_stride_token = None

        # Verify the input tokens.
        assert dp_x.shape == (num_tokens, self._hidden_dim)
        assert dp_x.stride(1) == 1
        assert dp_x.dtype == self._in_dtype
        x_ptr = dp_x.data_ptr()
        x_stride = dp_x.stride(0)

        # Verify the input scales.
        x_scale_ptr: Optional[int]
        x_scale_stride_elem: Optional[int]
        x_scale_stride_token: Optional[int]
        if dp_x_scale is not None:
            assert self._scale_dtype is not None
            assert self._hidden_dim_scale is not None
            assert out_expert_x_scale is not None
            assert dp_x_scale.dtype == self._scale_dtype
            x_scale_ptr = dp_x_scale.data_ptr()
            x_scale_stride_elem = dp_x_scale.stride(1)
            x_scale_stride_token = dp_x_scale.stride(0)
        else:
            assert self._scale_dtype is None
            assert self._hidden_dim_scale is None
            x_scale_ptr = None
            x_scale_stride_elem = None
            x_scale_stride_token = None

        # Verify the indices.
        assert indices.shape == (num_tokens, self._num_experts_per_token)
        assert indices.stride(1) == 1
        assert indices.dtype == torch.uint32
        indices_ptr = indices.data_ptr()
        indices_stride = indices.stride(0)

        # Verify the weights.
        assert weights.shape == (num_tokens, self._num_experts_per_token)
        assert weights.stride(1) == 1
        assert weights.dtype == torch.float32
        weights_ptr = weights.data_ptr()
        weights_stride = weights.stride(0)

        # Verify the dynamic `m` bound.
        bound_m_ptr: Optional[int]
        if bound_m is not None:
            assert bound_m.numel() == 1
            assert bound_m.dtype == torch.int32
            bound_m_ptr = bound_m.data_ptr()
        else:
            bound_m_ptr = None

        stream = torch.cuda.current_stream().cuda_stream

        if do_send:
            if _slot is None and torch.cuda.is_current_stream_capturing():
                _slot = self._acquire_cuda_graph_capture_slot()
                logger.info(
                    "PPLX CUDA graph dispatch capture slot=%s num_tokens=%s",
                    _slot,
                    num_tokens,
                )

            if _slot is None:
                _slot = all_to_all.dispatch_send(
                    num_tokens=num_tokens,
                    x_ptr=x_ptr,
                    x_stride=x_stride * self._in_dtype.itemsize,
                    x_scale_ptr=x_scale_ptr,
                    x_scale_stride_elem=x_scale_stride_elem,
                    x_scale_stride_token=x_scale_stride_token,
                    indices_ptr=indices_ptr,
                    indices_stride=indices_stride,
                    weights_ptr=weights_ptr,
                    weights_stride=weights_stride,
                    bound_m_ptr=bound_m_ptr,
                    stream=stream,
                )
            else:
                all_to_all.dispatch_send_on_slot(
                    slot=_slot,
                    num_tokens=num_tokens,
                    x_ptr=x_ptr,
                    x_stride=x_stride * self._in_dtype.itemsize,
                    x_scale_ptr=x_scale_ptr,
                    x_scale_stride_elem=x_scale_stride_elem,
                    x_scale_stride_token=x_scale_stride_token,
                    indices_ptr=indices_ptr,
                    indices_stride=indices_stride,
                    weights_ptr=weights_ptr,
                    weights_stride=weights_stride,
                    bound_m_ptr=bound_m_ptr,
                    stream=stream,
                )
            self._sync_slot = _slot

        if _slot is None:
            _slot = self._sync_slot
        assert _slot is not None

        if do_recv:
            all_to_all.dispatch_recv(
                slot=_slot,
                out_num_tokens_ptr=out_expert_num_tokens_ptr,
                out_x_ptr=out_x_ptr,
                out_x_stride=out_x_stride,
                out_x_scale_ptr=out_x_scale_ptr,
                out_x_scale_stride_elem=out_x_scale_stride_elem,
                out_x_scale_stride_token=out_x_scale_stride_token,
                stream=stream,
            )

    def _flatten_batched_experts(
        self,
        tensor: torch.Tensor,
        *,
        dtype: torch.dtype,
        hidden_dim: int,
        name: str,
    ) -> torch.Tensor:
        if tensor.ndim == 2:
            num_expert_tokens, _ = tensor.shape
            assert tensor.shape == (num_expert_tokens, hidden_dim)
            assert tensor.stride(1) == 1
            assert tensor.dtype == dtype
            return tensor

        assert self._max_tokens_per_expert is not None
        assert tensor.ndim == 3
        assert tensor.shape == (
            self._num_local_experts,
            self._max_tokens_per_expert,
            hidden_dim,
        ), f"{name} has unexpected shape {tuple(tensor.shape)}"
        assert tensor.stride(2) == 1
        assert tensor.dtype == dtype
        return tensor.reshape(-1, hidden_dim)

    def dispatch_async(
        self,
        *,
        out_expert_num_tokens: torch.Tensor,
        out_expert_x: torch.Tensor,
        out_expert_x_scale: Optional[torch.Tensor],
        dp_x: torch.Tensor,
        dp_x_scale: Optional[torch.Tensor],
        indices: torch.Tensor,
        weights: torch.Tensor,
        bound_m: Optional[torch.Tensor] = None,
    ) -> P2PDispatchHandle:
        self.dispatch(
            out_expert_num_tokens=out_expert_num_tokens,
            out_expert_x=out_expert_x,
            out_expert_x_scale=out_expert_x_scale,
            dp_x=dp_x,
            dp_x_scale=dp_x_scale,
            indices=indices,
            weights=weights,
            bound_m=bound_m,
            do_send=True,
            do_recv=False,
        )
        assert self._sync_slot is not None
        slot = self._sync_slot
        send_done_event = torch.cuda.Event()
        send_done_event.record(torch.cuda.current_stream(dp_x.device))
        return P2PDispatchHandle(
            kernel=self,
            out_expert_num_tokens=out_expert_num_tokens,
            out_expert_x=out_expert_x,
            out_expert_x_scale=out_expert_x_scale,
            dp_x=dp_x,
            dp_x_scale=dp_x_scale,
            indices=indices,
            weights=weights,
            bound_m=bound_m,
            send_done_event=send_done_event,
            _slot=slot,
        )

    @override
    def combine(
        self,
        out_tokens: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        expert_y: torch.Tensor,
        bound_m: Optional[torch.Tensor] = None,
        do_send: bool = True,
        do_recv: bool = True,
        accumulate: bool = False,
        _slot: Optional[int] = None,
    ) -> None:
        assert self._all_to_all is not None
        assert do_send or do_recv
        all_to_all = self._all_to_all

        # TODO: accumulate with TP across NVLink
        assert not accumulate or self._dp_size == 1

        num_tokens, _ = indices.shape
        expert_y = self._flatten_batched_experts(
            expert_y,
            dtype=expert_y.dtype,
            hidden_dim=self._hidden_dim,
            name="expert_y",
        )
        num_recv_tokens, _ = expert_y.shape

        assert out_tokens.shape == (num_tokens, self._hidden_dim)
        assert out_tokens.dtype == self._out_dtype
        assert out_tokens.stride(1) == 1
        out_tokens_ptr = out_tokens.data_ptr()
        out_tokens_stride = out_tokens.stride(0)

        assert indices.shape == (num_tokens, self._num_experts_per_token)
        assert indices.stride(1) == 1
        assert indices.dtype == torch.uint32
        indices_ptr = indices.data_ptr()
        indices_stride = indices.stride(0)

        assert weights.shape == (num_tokens, self._num_experts_per_token)
        assert weights.stride(1) == 1
        assert weights.dtype == torch.float32
        weights_ptr = weights.data_ptr()
        weights_stride = weights.stride(0)

        assert expert_y.shape == (num_recv_tokens, self._hidden_dim)
        expert_y_ptr = expert_y.data_ptr()
        expert_y_stride = expert_y.stride(0) * expert_y.dtype.itemsize

        bound_m_ptr: Optional[int]
        if bound_m is not None:
            assert bound_m.numel() == 1
            assert bound_m.dtype == torch.int32
            bound_m_ptr = bound_m.data_ptr()
        else:
            bound_m_ptr = None

        stream = torch.cuda.current_stream().cuda_stream

        if _slot is None:
            _slot = self._sync_slot
        assert _slot is not None

        if do_send:
            if torch.cuda.is_current_stream_capturing():
                logger.info(
                    "PPLX CUDA graph combine send slot=%s num_tokens=%s "
                    "num_recv_tokens=%s",
                    _slot,
                    num_tokens,
                    num_recv_tokens,
                )
            if os.environ.get("PPLX_GARDEN_TRACE") == "1":
                logger.warning(
                    "PPLX combine send trace slot=%s capturing=%s "
                    "stream=%s num_tokens=%s num_recv_tokens=%s "
                    "expert_y_ptr=%#x expert_y_stride=%s "
                    "expert_y_shape=%s expert_y_stride_elems=%s "
                    "out_tokens_ptr=%#x out_tokens_shape=%s "
                    "indices_ptr=%#x weights_ptr=%#x bound_m_ptr=%s",
                    _slot,
                    torch.cuda.is_current_stream_capturing(),
                    stream,
                    num_tokens,
                    num_recv_tokens,
                    expert_y_ptr,
                    expert_y_stride,
                    tuple(expert_y.shape),
                    tuple(expert_y.stride()),
                    out_tokens_ptr,
                    tuple(out_tokens.shape),
                    indices_ptr,
                    weights_ptr,
                    None if bound_m_ptr is None else hex(bound_m_ptr),
                )
            all_to_all.combine_send(
                slot=_slot,
                expert_x_ptr=expert_y_ptr,
                expert_x_stride=expert_y_stride,
                stream=stream,
            )

        if do_recv:
            all_to_all.combine_recv(
                slot=_slot,
                num_tokens=num_tokens,
                num_recv_tokens=num_recv_tokens,
                expert_y_dtype=expert_y.dtype,
                out_tokens_ptr=out_tokens_ptr,
                out_tokens_stride=out_tokens_stride,
                indices_ptr=indices_ptr,
                indices_stride=indices_stride,
                weights_ptr=weights_ptr,
                weights_stride=weights_stride,
                bound_m_ptr=bound_m_ptr,
                accumulate=accumulate,
                stream=stream,
            )
            if self._sync_slot == _slot:
                self._sync_slot = None
            if torch.cuda.is_current_stream_capturing():
                self._release_cuda_graph_capture_slot(_slot)

    def combine_async(
        self,
        *,
        out_tokens: torch.Tensor,
        dispatch_handle: P2PDispatchHandle,
        expert_y: torch.Tensor,
        bound_m: Optional[torch.Tensor] = None,
        accumulate: bool = False,
    ) -> P2PCombineHandle:
        dispatch_handle.wait_recv_done()
        self.combine(
            out_tokens=out_tokens,
            indices=dispatch_handle.indices,
            weights=dispatch_handle.weights,
            expert_y=expert_y,
            bound_m=bound_m,
            do_send=True,
            do_recv=False,
            accumulate=accumulate,
            _slot=dispatch_handle._slot,
        )
        send_done_event = torch.cuda.Event()
        send_done_event.record(torch.cuda.current_stream(out_tokens.device))
        return P2PCombineHandle(
            kernel=self,
            dispatch_handle=dispatch_handle,
            out_tokens=out_tokens,
            expert_y=expert_y,
            bound_m=bound_m,
            accumulate=accumulate,
            send_done_event=send_done_event,
            _slot=dispatch_handle._slot,
        )

    def get_perf_stats(self) -> dict[str, Any]:
        if self._all_to_all is None:
            return {}
        return self._all_to_all.get_perf_stats()

    def get_debug_state(self) -> list[dict[str, Any]]:
        if self._all_to_all is None:
            return []
        state = self._all_to_all.get_debug_state()
        for slot in state:
            slot["python_capture_free"] = slot["slot"] in self._capture_free_slots
            slot["python_sync_slot"] = self._sync_slot
        return state

    @override
    def destroy(self) -> None:
        """Clean up the all-to-all context."""

        # Stop the a2a engine, ensuring all RDMA transfers complete.
        self._global_group.barrier()
        self._all_to_all = None

        # Stop the transfer engine once no rank is active.
        self._global_group.barrier()
        if self._transfer_engine is not None and self._owns_transfer_engine:
            self._transfer_engine.stop()
            del self._transfer_engine
            self._transfer_engine = None

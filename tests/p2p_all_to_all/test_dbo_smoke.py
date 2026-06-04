# ruff: noqa: E402

from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from pplx_garden.distributed import ParallelGroup, ParallelLaunch
from pplx_garden.kernels.p2p_all_to_all import (
    P2PAllToAll,
    reset_all_cuda_graph_capture_slots,
)
from pplx_garden.utils import logging_utils
from pplx_garden.utils.math import round_up
from pplx_garden.utils.torch import has_tp
from tests.fabric import get_nets_per_gpu
from tests.markers import (
    gpu_only,
    mark_ci_2gpu,
    mark_cuda,
    mark_distributed,
    mark_fabric,
    mark_kernel,
)
from tests.p2p_all_to_all.data import RankTestData
from tests.p2p_all_to_all.layout import assert_canonical_batched_experts_layout

logger = logging_utils.get_logger(__name__)


@dataclass
class _DBOConfig:
    world_size: int
    dp_size: int
    nets_per_gpu: int
    max_num_tokens: int
    num_experts: int
    hidden_dim: int
    hidden_dim_scale: Optional[int]
    max_private_tokens: Optional[int]
    num_experts_per_token: int
    in_dtype: torch.dtype
    out_dtype: torch.dtype
    scale_dtype: Optional[torch.dtype]
    expert_padding: int
    nvlink_group: Optional[int]
    max_tokens_per_expert: Optional[int] = None


def _act(x: torch.Tensor, x_scale: Optional[torch.Tensor]) -> torch.Tensor:
    if x_scale is None:
        return x * 2

    _, hidden_dim = x.shape
    _, hidden_dim_scale = x_scale.shape
    return x.to(torch.float32) * x_scale.repeat(1, hidden_dim // hidden_dim_scale) * 2


def _expert_forward(
    out_expert_x: torch.Tensor,
    out_expert_x_scale: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if out_expert_x.ndim == 3:
        num_local_experts, max_tokens_per_expert, hidden_dim = out_expert_x.shape
        flat_out_expert_x = out_expert_x.reshape(-1, hidden_dim)
        flat_out_expert_x_scale = (
            None
            if out_expert_x_scale is None
            else out_expert_x_scale.reshape(-1, out_expert_x_scale.shape[-1])
        )
        expert_y = _act(flat_out_expert_x, flat_out_expert_x_scale).to(out_dtype)
        return expert_y.reshape(num_local_experts, max_tokens_per_expert, hidden_dim)
    else:
        return _act(out_expert_x, out_expert_x_scale).to(out_dtype)


def _test_dbo_smoke_worker(
    device: torch.device,
    tp_group: Optional[ParallelGroup],
    global_group: Optional[ParallelGroup],
    config: _DBOConfig,
) -> None:
    assert tp_group is not None
    assert global_group is not None

    dp_rank = global_group.rank // tp_group.size
    num_dp_groups = global_group.size // tp_group.size

    max_num_tokens = config.max_num_tokens
    num_experts = config.num_experts
    hidden_dim = config.hidden_dim
    hidden_dim_scale = config.hidden_dim_scale
    num_experts_per_token = config.num_experts_per_token
    in_dtype = config.in_dtype
    out_dtype = config.out_dtype
    scale_dtype = config.scale_dtype

    num_local_experts = num_experts // num_dp_groups
    first_expert = dp_rank * num_local_experts
    ll_max_tokens_per_expert = config.max_tokens_per_expert
    if ll_max_tokens_per_expert is None:
        ll_max_tokens_per_expert = max_num_tokens * global_group.size

    # Create dummy data for Batch A and Batch B with distinct seeds.
    rank_data_a = []
    rank_data_b = []
    for rank in range(num_dp_groups):
        generator_a = torch.Generator(device=device)
        generator_a.manual_seed(rank)
        rank_data_a.append(
            RankTestData.create(
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                max_num_tokens=max_num_tokens,
                hidden_dim=hidden_dim,
                hidden_dim_scale=hidden_dim_scale,
                in_dtype=in_dtype,
                scale_dtype=scale_dtype,
                generator=generator_a,
                device=device,
            )
        )

        generator_b = torch.Generator(device=device)
        generator_b.manual_seed(rank + 100)
        rank_data_b.append(
            RankTestData.create(
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                max_num_tokens=max_num_tokens,
                hidden_dim=hidden_dim,
                hidden_dim_scale=hidden_dim_scale,
                in_dtype=in_dtype,
                scale_dtype=scale_dtype,
                generator=generator_b,
                device=device,
            )
        )

    data_a = rank_data_a[dp_rank]
    data_b = rank_data_b[dp_rank]

    node_group: Optional[ParallelGroup]
    if config.nvlink_group is not None:
        assert config.nvlink_group > 0
        assert global_group.size % config.nvlink_group == 0
        node_group = global_group.slice_by_count(
            global_group.size // config.nvlink_group,
        )
    else:
        node_group = None

    # Instantiate the all-to-all kernel
    all_to_all = P2PAllToAll(
        max_num_tokens=max_num_tokens,
        num_experts=num_experts,
        expert_padding=config.expert_padding,
        hidden_dim=hidden_dim,
        hidden_dim_scale=hidden_dim_scale,
        max_private_tokens=config.max_private_tokens,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        scale_dtype=scale_dtype,
        num_experts_per_token=num_experts_per_token,
        nets_per_gpu=config.nets_per_gpu,
        device=device,
        dp_group=tp_group,
        node_group=node_group,
        global_group=global_group,
        max_tokens_per_expert=ll_max_tokens_per_expert,
    )

    try:
        # Pre-allocate output and intermediate tensors for Batch A
        expert_num_tokens_a = torch.empty(
            (num_local_experts,), dtype=torch.int32, device=device
        )
        out_expert_x_shape_a = (
            num_local_experts,
            ll_max_tokens_per_expert,
            hidden_dim,
        )
        out_expert_x_a = torch.empty(
            out_expert_x_shape_a, dtype=in_dtype, device=device
        )
        out_tokens_a = torch.empty(
            (max_num_tokens, hidden_dim), dtype=out_dtype, device=device
        )

        if hidden_dim_scale is not None or scale_dtype is not None:
            out_expert_x_scale_shape_a = (
                num_local_experts,
                ll_max_tokens_per_expert,
                hidden_dim_scale,
            )
            out_expert_x_scale_a = torch.empty(
                out_expert_x_scale_shape_a, dtype=scale_dtype, device=device
            )
        else:
            out_expert_x_scale_a = None

        # Pre-allocate output and intermediate tensors for Batch B
        expert_num_tokens_b = torch.empty(
            (num_local_experts,), dtype=torch.int32, device=device
        )
        out_expert_x_shape_b = (
            num_local_experts,
            ll_max_tokens_per_expert,
            hidden_dim,
        )
        out_expert_x_b = torch.empty(
            out_expert_x_shape_b, dtype=in_dtype, device=device
        )
        out_tokens_b = torch.empty(
            (max_num_tokens, hidden_dim), dtype=out_dtype, device=device
        )

        if hidden_dim_scale is not None or scale_dtype is not None:
            out_expert_x_scale_shape_b = (
                num_local_experts,
                ll_max_tokens_per_expert,
                hidden_dim_scale,
            )
            out_expert_x_scale_b = torch.empty(
                out_expert_x_scale_shape_b, dtype=scale_dtype, device=device
            )
        else:
            out_expert_x_scale_b = None

        # ----------------------------------------------------
        # 1. REFERENCE SYNCHRONOUS RUN
        # ----------------------------------------------------
        # Reference outputs used to verify correctness of async execution
        ref_out_tokens_a = torch.empty(
            (max_num_tokens, hidden_dim), dtype=out_dtype, device=device
        )
        ref_out_expert_x_a = torch.empty_like(out_expert_x_a)
        ref_out_expert_x_scale_a = (
            None
            if out_expert_x_scale_a is None
            else torch.empty_like(out_expert_x_scale_a)
        )
        ref_expert_num_tokens_a = torch.empty_like(expert_num_tokens_a)

        all_to_all.dispatch(
            out_expert_num_tokens=ref_expert_num_tokens_a,
            out_expert_x=ref_out_expert_x_a,
            out_expert_x_scale=ref_out_expert_x_scale_a,
            dp_x=data_a.dp_x,
            dp_x_scale=data_a.dp_x_scale,
            indices=data_a.indices,
            weights=data_a.weights,
            bound_m=None,
        )
        torch.cuda.synchronize()
        assert_canonical_batched_experts_layout(
            out_expert_x=ref_out_expert_x_a,
            out_expert_x_scale=ref_out_expert_x_scale_a,
            expert_num_tokens=ref_expert_num_tokens_a,
            rank_data=rank_data_a,
            first_expert=first_expert,
            num_local_experts=num_local_experts,
            rank=global_group.rank,
            dp_size=tp_group.size,
            node_size=node_group.size if node_group is not None else tp_group.size,
            world_size=global_group.size,
            expert_padding=config.expert_padding,
            max_tokens_per_expert=ll_max_tokens_per_expert,
        )
        ref_expert_y_a = _expert_forward(
            ref_out_expert_x_a, ref_out_expert_x_scale_a, out_dtype
        )
        all_to_all.combine(
            out_tokens=ref_out_tokens_a,
            indices=data_a.indices,
            weights=data_a.weights,
            expert_y=ref_expert_y_a,
            bound_m=data_a.bound_m,
        )

        ref_out_tokens_b = torch.empty(
            (max_num_tokens, hidden_dim), dtype=out_dtype, device=device
        )
        ref_out_expert_x_b = torch.empty_like(out_expert_x_b)
        ref_out_expert_x_scale_b = (
            None
            if out_expert_x_scale_b is None
            else torch.empty_like(out_expert_x_scale_b)
        )
        ref_expert_num_tokens_b = torch.empty_like(expert_num_tokens_b)

        all_to_all.dispatch(
            out_expert_num_tokens=ref_expert_num_tokens_b,
            out_expert_x=ref_out_expert_x_b,
            out_expert_x_scale=ref_out_expert_x_scale_b,
            dp_x=data_b.dp_x,
            dp_x_scale=data_b.dp_x_scale,
            indices=data_b.indices,
            weights=data_b.weights,
            bound_m=None,
        )
        torch.cuda.synchronize()
        assert_canonical_batched_experts_layout(
            out_expert_x=ref_out_expert_x_b,
            out_expert_x_scale=ref_out_expert_x_scale_b,
            expert_num_tokens=ref_expert_num_tokens_b,
            rank_data=rank_data_b,
            first_expert=first_expert,
            num_local_experts=num_local_experts,
            rank=global_group.rank,
            dp_size=tp_group.size,
            node_size=node_group.size if node_group is not None else tp_group.size,
            world_size=global_group.size,
            expert_padding=config.expert_padding,
            max_tokens_per_expert=ll_max_tokens_per_expert,
        )
        ref_expert_y_b = _expert_forward(
            ref_out_expert_x_b, ref_out_expert_x_scale_b, out_dtype
        )
        all_to_all.combine(
            out_tokens=ref_out_tokens_b,
            indices=data_b.indices,
            weights=data_b.weights,
            expert_y=ref_expert_y_b,
            bound_m=data_b.bound_m,
        )
        torch.cuda.synchronize()

        # Zero out the DBO output tensors to guarantee we aren't reading old values
        out_tokens_a.zero_()
        out_tokens_b.zero_()
        out_expert_x_a.zero_()
        out_expert_x_b.zero_()
        if out_expert_x_scale_a is not None:
            out_expert_x_scale_a.zero_()
            out_expert_x_scale_b.zero_()
        expert_num_tokens_a.zero_()
        expert_num_tokens_b.zero_()

        # ----------------------------------------------------
        # 2. DUAL BATCH OVERLAP (DBO) ASYNC PIPELINE
        # ----------------------------------------------------
        # Objective: overlap Slot A's computation with Slot B's background communication.

        # [STAGE 1] Trigger Async Dispatch on Batch A
        (
            out_expert_x_a,
            out_expert_x_scale_a,
            expert_num_tokens_a,
            dispatch_handle_a,
            dispatch_recv_a,
        ) = all_to_all.low_latency_dispatch(
            data_a.dp_x,
            data_a.indices,
            data_a.weights,
            data_a.dp_x_scale,
            slot_key=0,
        )

        ## Previous B compute goes here, i.e. Q/K Proj

        # [STAGE 2] Wait/Sync on Batch A's dispatch completion so we can start its compute
        dispatch_recv_a()
        torch.cuda.synchronize()
        assert_canonical_batched_experts_layout(
            out_expert_x=out_expert_x_a,
            out_expert_x_scale=out_expert_x_scale_a,
            expert_num_tokens=expert_num_tokens_a,
            rank_data=rank_data_a,
            first_expert=first_expert,
            num_local_experts=num_local_experts,
            rank=global_group.rank,
            dp_size=tp_group.size,
            node_size=node_group.size if node_group is not None else tp_group.size,
            world_size=global_group.size,
            expert_padding=config.expert_padding,
            max_tokens_per_expert=ll_max_tokens_per_expert,
        )

        # [STAGE 3] OVERLAP STEP: Kick off Async Dispatch on Batch B, and while it transmits,
        # run local expert forward computation on Batch A.
        (
            out_expert_x_b,
            out_expert_x_scale_b,
            expert_num_tokens_b,
            dispatch_handle_b,
            dispatch_recv_b,
        ) = all_to_all.low_latency_dispatch(
            data_b.dp_x,
            data_b.indices,
            data_b.weights,
            data_b.dp_x_scale,
            slot_key=1,
        )

        # --- Local Computation on Batch A (Overlapped with Dispatch B transfer) ---
        expert_y_a = _expert_forward(out_expert_x_a, out_expert_x_scale_a, out_dtype)

        # [STAGE 4] Wait/Sync on Batch B's dispatch completion
        dispatch_recv_b()
        torch.cuda.synchronize()
        assert_canonical_batched_experts_layout(
            out_expert_x=out_expert_x_b,
            out_expert_x_scale=out_expert_x_scale_b,
            expert_num_tokens=expert_num_tokens_b,
            rank_data=rank_data_b,
            first_expert=first_expert,
            num_local_experts=num_local_experts,
            rank=global_group.rank,
            dp_size=tp_group.size,
            node_size=node_group.size if node_group is not None else tp_group.size,
            world_size=global_group.size,
            expert_padding=config.expert_padding,
            max_tokens_per_expert=ll_max_tokens_per_expert,
        )

        # [STAGE 5] OVERLAP STEP: Kick off Async Combine on Batch A, and while it transmits,
        # run local expert forward computation on Batch B.
        _combine_handle_a, combine_recv_a = all_to_all.low_latency_combine(
            expert_y_a,
            dispatch_handle_a,
            out=out_tokens_a,
            bound_m=data_a.bound_m,
        )

        # --- Local Computation on Batch B (Overlapped with Combine A transfer) ---
        expert_y_b = _expert_forward(out_expert_x_b, out_expert_x_scale_b, out_dtype)

        # [STAGE 6] Wait/Sync on Batch A's combine completion.
        combine_recv_a()
        # Batch A is now fully complete!

        # [STAGE 7] Kick off Async Combine on Batch B.
        # Since this is the end of our microbatch stream, we can optionally overlap this
        # with downstream consumer logic, or simply wait on it.
        _combine_handle_b, combine_recv_b = all_to_all.low_latency_combine(
            expert_y_b,
            dispatch_handle_b,
            out=out_tokens_b,
            bound_m=data_b.bound_m,
        )

        # Downstream compute goes here: i.e. MLA proj

        # Wait/Sync on Batch B's combine completion.
        combine_recv_b()
        # Batch B is now fully complete!

        torch.cuda.synchronize()

        # ----------------------------------------------------
        # 3. VERIFY CORRECTNESS
        # ----------------------------------------------------
        # Verify that the asynchronous dual batch overlap execution
        # produced identical values to the reference synchronous run.
        torch.testing.assert_close(out_tokens_a, ref_out_tokens_a)
        torch.testing.assert_close(out_tokens_b, ref_out_tokens_b)
        torch.testing.assert_close(expert_num_tokens_a, ref_expert_num_tokens_a)
        torch.testing.assert_close(expert_num_tokens_b, ref_expert_num_tokens_b)

        # ----------------------------------------------------
        # 4. CUDA GRAPH CAPTURE/REPLAY GATE
        # ----------------------------------------------------
        # Capture the same two-slot LL dispatch/compute/combine sequence. During
        # capture the host only records launches; the worker observes the device
        # flags on replay, so replay must happen before any further slot reuse.
        assert out_expert_x_scale_a is None
        assert out_expert_x_scale_b is None
        assert data_a.dp_x_scale is None
        assert data_b.dp_x_scale is None

        graph_out_tokens_a = torch.empty_like(out_tokens_a)
        graph_out_tokens_b = torch.empty_like(out_tokens_b)
        graph_expert_y_a = torch.empty_like(out_expert_x_a, dtype=out_dtype)
        graph_expert_y_b = torch.empty_like(out_expert_x_b, dtype=out_dtype)
        graph_out_tokens_a.zero_()
        graph_out_tokens_b.zero_()
        graph_expert_y_a.zero_()
        graph_expert_y_b.zero_()
        torch.cuda.synchronize()

        reset_all_cuda_graph_capture_slots()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            (
                graph_expert_x_a,
                _graph_expert_x_scale_a,
                _graph_expert_num_tokens_a,
                graph_dispatch_handle_a,
                graph_dispatch_recv_a,
            ) = all_to_all.low_latency_dispatch(
                data_a.dp_x,
                data_a.indices,
                data_a.weights,
                data_a.dp_x_scale,
                slot_key=0,
            )
            graph_dispatch_recv_a()
            torch.mul(graph_expert_x_a, 2, out=graph_expert_y_a)

            (
                graph_expert_x_b,
                _graph_expert_x_scale_b,
                _graph_expert_num_tokens_b,
                graph_dispatch_handle_b,
                graph_dispatch_recv_b,
            ) = all_to_all.low_latency_dispatch(
                data_b.dp_x,
                data_b.indices,
                data_b.weights,
                data_b.dp_x_scale,
                slot_key=1,
            )
            graph_dispatch_recv_b()
            torch.mul(graph_expert_x_b, 2, out=graph_expert_y_b)

            _graph_combine_handle_a, graph_combine_recv_a = (
                all_to_all.low_latency_combine(
                    graph_expert_y_a,
                    graph_dispatch_handle_a,
                    out=graph_out_tokens_a,
                    bound_m=data_a.bound_m,
                )
            )
            graph_combine_recv_a()

            _graph_combine_handle_b, graph_combine_recv_b = (
                all_to_all.low_latency_combine(
                    graph_expert_y_b,
                    graph_dispatch_handle_b,
                    out=graph_out_tokens_b,
                    bound_m=data_b.bound_m,
                )
            )
            graph_combine_recv_b()

        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_out_tokens_a, ref_out_tokens_a)
        torch.testing.assert_close(graph_out_tokens_b, ref_out_tokens_b)

    except Exception:
        logger.exception("DBO smoke test failed")
        raise
    finally:
        logger.info("Destroying all-to-all engine")
        all_to_all.destroy()


@mark_fabric
@mark_kernel
@mark_cuda
@mark_distributed
@gpu_only
@pytest.mark.parametrize(
    "config",
    [
        # Standard configuration with TP=2
        pytest.param(
            _DBOConfig(
                world_size=2,
                dp_size=1,
                nets_per_gpu=1,
                max_num_tokens=128,
                num_experts=8,
                hidden_dim=256,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=2,
                in_dtype=torch.float32,
                out_dtype=torch.float32,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[pytest.mark.skipif(not has_tp(2), reason="Requires 2 devices")],
            id="TP2-FP32-DBO",
        ),
        # BF16 configuration with TP=2
        pytest.param(
            _DBOConfig(
                world_size=2,
                dp_size=1,
                nets_per_gpu=1,
                max_num_tokens=128,
                num_experts=8,
                hidden_dim=256,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=2,
                in_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[pytest.mark.skipif(not has_tp(2), reason="Requires 2 devices")],
            id="TP2-BF16-DBO",
        ),
    ],
)
def test_p2p_all_to_all_dbo_smoke(config: _DBOConfig) -> None:
    ParallelLaunch(world_size=config.world_size, dp_size=config.dp_size).run(
        _test_dbo_smoke_worker,
        config,
    )

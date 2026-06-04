use std::{
    ffi::c_void,
    ptr::null_mut,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    thread::JoinHandle,
};

use anyhow::{Result, anyhow};
use cuda_lib::{
    CudaDeviceMemory, cuda_check,
    rt::{CudartError, cudaGetNumSMs, cudaSetDevice},
};
use fabric_lib::{TransferEngine, api::MemoryRegionHandle};
use thread_lib::pin_cpu;
use torch_lib::ScalarType;

use crate::{
    a2a_handles::AllToAllRankHandle,
    a2a_worker::{
        LowLatencyRouteLayoutPlan, SlotPool, WorkerState,
        compute_low_latency_route_layout_plan,
    },
};

const LOW_LATENCY_WORKSPACE_ALIGNMENT: usize = 256;

#[derive(Debug, Clone)]
pub struct LowLatencyWorkspaceTensorSpec {
    pub name: &'static str,
    pub offset_bytes: usize,
    pub nbytes: usize,
    pub shape: Vec<usize>,
    pub dtype: &'static str,
}

#[derive(Debug, Clone)]
pub struct LowLatencyWorkspaceLayout {
    pub alignment: usize,
    pub total_bytes: usize,
    pub tensors: Vec<LowLatencyWorkspaceTensorSpec>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchHandleState {
    pub slot: usize,
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotGenerationState {
    Idle,
    DispatchSent { generation: u64, num_tokens: usize },
    DispatchReceived { generation: u64, num_tokens: usize },
    CombineSent { generation: u64, num_tokens: usize },
}

fn align_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

fn add_workspace_tensor(
    tensors: &mut Vec<LowLatencyWorkspaceTensorSpec>,
    total_bytes: &mut usize,
    name: &'static str,
    shape: Vec<usize>,
    dtype: &'static str,
    element_size: usize,
) {
    *total_bytes = align_up(*total_bytes, LOW_LATENCY_WORKSPACE_ALIGNMENT);
    let nbytes = shape.iter().product::<usize>() * element_size;
    tensors.push(LowLatencyWorkspaceTensorSpec {
        name,
        offset_bytes: *total_bytes,
        nbytes,
        shape,
        dtype,
    });
    *total_bytes += nbytes;
}

// Collects the private workspace buffers used by dispatch and combine.
struct DeviceWorkspace {
    /// The offset of each expert within the contiguous token buffer.
    expert_offsets: CudaDeviceMemory,
    /// The offset of the token within the expert group.
    token_offset: CudaDeviceMemory,
    /// Source-rank local `(token, topk)` to combine receive-buffer position.
    combine_recv_position: CudaDeviceMemory,
    /// Counter for the number of tokens sent during combine.
    token_counter: CudaDeviceMemory,
    /// Completion counter for dispatch-send.
    dispatch_send_counter: CudaDeviceMemory,
    /// Completion counter for dispatch-recv.
    dispatch_recv_counter: CudaDeviceMemory,
    /// Monotonic per-slot epoch counter incremented by dispatch-send on device.
    epoch_counter: CudaDeviceMemory,
    /// Current in-flight epoch for kernels later in the same slot operation.
    current_epoch: CudaDeviceMemory,
    /// Counter for synchronization barriers across NVLink.
    sync_counter: CudaDeviceMemory,
    /// Device-side sync pointers.
    sync_ptrs: Option<CudaDeviceMemory>,
    /// Device-side send pointers.
    send_ptrs: Option<CudaDeviceMemory>,
    /// Device-side recv pointers.
    recv_ptrs: Option<CudaDeviceMemory>,
}

impl DeviceWorkspace {
    pub fn new(
        num_experts: usize,
        max_num_tokens: usize,
        num_experts_per_token: usize,
        host_sync_ptrs: &[u64],
        host_send_ptrs: &[u64],
        host_recv_ptrs: &[u64],
    ) -> Result<Self, CudartError> {
        let expert_offsets =
            CudaDeviceMemory::device(num_experts * std::mem::size_of::<u32>())?;
        expert_offsets.zero();

        let token_offset = CudaDeviceMemory::device(
            max_num_tokens * num_experts_per_token * std::mem::size_of::<u32>(),
        )?;
        let combine_recv_position = CudaDeviceMemory::device(
            max_num_tokens * num_experts_per_token * std::mem::size_of::<u32>(),
        )?;

        let token_counter = CudaDeviceMemory::device(std::mem::size_of::<u32>())?;
        token_counter.zero();
        let sync_counter = CudaDeviceMemory::device(std::mem::size_of::<u32>())?;
        sync_counter.zero();
        let dispatch_send_counter =
            CudaDeviceMemory::device(std::mem::size_of::<u32>())?;
        dispatch_send_counter.zero();
        let dispatch_recv_counter =
            CudaDeviceMemory::device(std::mem::size_of::<u32>())?;
        dispatch_recv_counter.zero();
        let epoch_counter = CudaDeviceMemory::device(std::mem::size_of::<u32>())?;
        epoch_counter.zero();
        let current_epoch = CudaDeviceMemory::device(std::mem::size_of::<u32>())?;
        current_epoch.zero();

        let sync_ptrs = if host_sync_ptrs.is_empty() {
            None
        } else {
            Some(CudaDeviceMemory::from_vec(host_sync_ptrs)?)
        };
        let send_ptrs = if host_send_ptrs.is_empty() {
            None
        } else {
            Some(CudaDeviceMemory::from_vec(host_send_ptrs)?)
        };
        let recv_ptrs = if host_recv_ptrs.is_empty() {
            None
        } else {
            Some(CudaDeviceMemory::from_vec(host_recv_ptrs)?)
        };

        Ok(Self {
            expert_offsets,
            token_offset,
            combine_recv_position,
            token_counter,
            dispatch_send_counter,
            dispatch_recv_counter,
            epoch_counter,
            current_epoch,
            sync_counter,
            sync_ptrs,
            send_ptrs,
            recv_ptrs,
        })
    }

    fn get_sync_ptr(&mut self) -> *mut *mut u32 {
        self.sync_ptrs.as_mut().map_or(null_mut(), |p| p.get_mut_ptr())
    }

    fn get_recv_ptr(&mut self) -> *mut *mut c_void {
        self.recv_ptrs.as_mut().map_or(null_mut(), |p| p.get_mut_ptr())
    }

    fn get_send_ptr(&mut self) -> *mut *mut c_void {
        self.send_ptrs.as_mut().map_or(null_mut(), |p| p.get_mut_ptr())
    }
}

#[allow(dead_code)]
pub struct AllToAllContext {
    hidden_dim: usize,
    hidden_dim_scale: usize,
    in_elemsize: usize,
    out_elemsize: usize,
    out_dtype: ScalarType,
    scale_elemsize: usize,
    num_experts: usize,
    max_num_tokens: usize,
    max_recv_tokens: usize,
    max_tokens_per_expert: usize,
    expert_padding: usize,
    num_experts_per_token: usize,
    max_private_tokens: usize,
    rank: usize,
    dp_size: usize,
    node_size: usize,
    world_size: usize,
    device: u8,
    workspaces: Vec<DeviceWorkspace>,
    workers: Vec<Arc<WorkerState>>,
    threads: Vec<JoinHandle<()>>,
    slot_pool: Arc<SlotPool>,
    next_generation: AtomicU64,
    slot_states: Vec<Mutex<SlotGenerationState>>,
    num_blocks: usize,
}

impl AllToAllContext {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_dim: usize,
        hidden_dim_scale: usize,
        in_elemsize: usize,
        out_elemsize: usize,
        out_dtype: ScalarType,
        scale_elemsize: usize,
        max_num_tokens: usize,
        max_recv_tokens: usize,
        max_tokens_per_expert: usize,
        max_private_tokens: usize,
        num_experts: usize,
        expert_padding: usize,
        num_experts_per_token: usize,
        rank: usize,
        dp_size: usize,
        node_size: usize,
        world_size: usize,
        num_routed_ptrs: Vec<*mut u32>,
        num_routed_mrs: Vec<MemoryRegionHandle>,
        send_buffer_ptrs: Vec<*mut c_void>,
        send_buffer_mrs: Vec<MemoryRegionHandle>,
        recv_buffer_ptrs: Vec<*mut c_void>,
        recv_buffer_mrs: Vec<MemoryRegionHandle>,
        sync_ptrs: Vec<Vec<u64>>,
        send_ptrs: Vec<Vec<u64>>,
        recv_ptrs: Vec<Vec<u64>>,
        node_route_count_ptrs: Vec<Vec<u64>>,
        node_route_epoch_ptrs: Vec<Vec<u64>>,
        device: u8,
        imm_base: u32,
        rank_handles: Vec<Vec<AllToAllRankHandle>>,
        transfer_engine: Arc<TransferEngine>,
        worker_cpu: Option<u16>,
        num_slots: usize,
    ) -> Result<Self> {
        // Start the all-to-all worker thread.
        for (i, peer) in rank_handles.first().into_iter().flatten().enumerate() {
            tracing::info!("Rank#{} Peer#{}: {}", rank, i, peer.address);
        }

        let num_slots = num_slots.max(1);
        if num_routed_ptrs.len() != num_slots || num_routed_mrs.len() != num_slots {
            return Err(anyhow!(
                "Expected {} num_routed buffers, got {} ptrs and {} MRs",
                num_slots,
                num_routed_ptrs.len(),
                num_routed_mrs.len()
            ));
        }
        if send_buffer_ptrs.len() != num_slots
            || send_buffer_mrs.len() != num_slots
            || recv_buffer_ptrs.len() != num_slots
            || recv_buffer_mrs.len() != num_slots
        {
            return Err(anyhow!(
                "Expected {} payload buffers, got send {} ptrs/{} MRs and recv {} ptrs/{} MRs",
                num_slots,
                send_buffer_ptrs.len(),
                send_buffer_mrs.len(),
                recv_buffer_ptrs.len(),
                recv_buffer_mrs.len()
            ));
        }
        if rank_handles.len() != num_slots {
            return Err(anyhow!(
                "Expected {} rank handle sets, got {}",
                num_slots,
                rank_handles.len()
            ));
        }
        if sync_ptrs.len() != num_slots
            || send_ptrs.len() != num_slots
            || recv_ptrs.len() != num_slots
        {
            return Err(anyhow!(
                "Expected {} NVLink pointer sets, got sync {} send {} recv {}",
                num_slots,
                sync_ptrs.len(),
                send_ptrs.len(),
                recv_ptrs.len()
            ));
        }
        if node_route_count_ptrs.len() != num_slots
            || node_route_epoch_ptrs.len() != num_slots
        {
            return Err(anyhow!(
                "Expected {} node route pointer sets, got counts {} epochs {}",
                num_slots,
                node_route_count_ptrs.len(),
                node_route_epoch_ptrs.len()
            ));
        }
        let slot_pool = Arc::new(SlotPool::new(num_slots));

        let mut workers = Vec::with_capacity(num_slots);
        let mut threads = Vec::with_capacity(num_slots);
        let mut workspaces = Vec::with_capacity(num_slots);

        for slot_idx in 0..num_slots {
            cudaSetDevice(device.into())?;
            let workspace = DeviceWorkspace::new(
                num_experts,
                max_num_tokens,
                num_experts_per_token,
                &sync_ptrs[slot_idx],
                &send_ptrs[slot_idx],
                &recv_ptrs[slot_idx],
            )?;

            let slot_imm_base = imm_base + (slot_idx as u32) * 5;
            let worker: Arc<WorkerState> = Arc::new(WorkerState::new(
                slot_idx,
                slot_pool.clone(),
                hidden_dim,
                hidden_dim_scale,
                in_elemsize,
                out_elemsize,
                scale_elemsize,
                max_num_tokens,
                max_recv_tokens,
                max_tokens_per_expert,
                max_private_tokens,
                num_experts,
                expert_padding,
                num_experts_per_token,
                rank,
                dp_size,
                node_size,
                world_size,
                num_routed_ptrs[slot_idx],
                num_routed_mrs[slot_idx],
                send_buffer_ptrs[slot_idx],
                send_buffer_mrs[slot_idx],
                recv_buffer_ptrs[slot_idx],
                recv_buffer_mrs[slot_idx],
                node_route_count_ptrs[slot_idx]
                    .iter()
                    .map(|ptr| *ptr as usize)
                    .collect(),
                node_route_epoch_ptrs[slot_idx]
                    .iter()
                    .map(|ptr| *ptr as usize)
                    .collect(),
                device,
                slot_imm_base,
                rank_handles[slot_idx].clone(),
                transfer_engine.clone(),
            )?);

            // Create the worker thread.
            let (init_tx, init_rx) = oneshot::channel();
            let thread_worker = worker.clone();
            let thread = std::thread::Builder::new()
                .name(format!("p2p_all_to_all Worker[{slot_idx}]"))
                .spawn(move || {
                    // Pin to the desired CPU.
                    tracing::info!(
                        "Running worker slot {} for cuda:{}",
                        slot_idx,
                        device
                    );
                    if let Some(cpu) = worker_cpu {
                        if let Err(e) = pin_cpu(cpu.into()) {
                            tracing::info!("Failed to pin CPU {}: {:?}", cpu, e);
                        }
                        tracing::info!(
                            "Pinned worker slot {} for cuda:{} to CPU {}",
                            slot_idx,
                            device,
                            cpu
                        );
                    }

                    // Block until the worker is fully initialized.
                    if init_tx.send(()).is_err() {
                        panic!("Failed to send initialization signal");
                    } else {
                        tracing::info!(
                            "Initialized worker slot {} for cuda:{}",
                            slot_idx,
                            device
                        );
                    }

                    // Main loop.
                    thread_worker.main_loop();
                    tracing::info!(
                        "Stopping worker slot {} for cuda:{}",
                        slot_idx,
                        device
                    );
                })
                .expect("Failed to spawn p2p_all_to_all Worker thread");
            init_rx.recv()?;

            workers.push(worker);
            threads.push(thread);
            workspaces.push(workspace);
        }

        let num_blocks = cudaGetNumSMs(device)?;

        // Build the context.
        Ok(Self {
            hidden_dim,
            hidden_dim_scale,
            in_elemsize,
            out_elemsize,
            out_dtype,
            scale_elemsize,
            num_experts,
            max_num_tokens,
            max_recv_tokens,
            max_tokens_per_expert,
            expert_padding,
            num_experts_per_token,
            max_private_tokens,
            rank,
            dp_size,
            node_size,
            world_size,
            device,
            workspaces,
            workers,
            threads,
            slot_pool,
            next_generation: AtomicU64::new(1),
            slot_states: (0..num_slots)
                .map(|_| Mutex::new(SlotGenerationState::Idle))
                .collect(),
            num_blocks,
        })
    }

    pub fn destroy(&mut self) -> Result<()> {
        // Stop all work on the worker thread.
        tracing::info!("Stopping worker thread for cuda:{}", self.device);

        for worker in &self.workers {
            worker.stop();
        }
        for thread in self.threads.drain(..) {
            if thread.join().is_err() {
                return Err(anyhow!("Failed to join thread"));
            }
        }
        Ok(())
    }

    fn worker(&self, slot: usize) -> Result<&Arc<WorkerState>> {
        self.workers
            .get(slot)
            .ok_or_else(|| anyhow!("Invalid all-to-all slot {}", slot))
    }

    fn workspace_mut(&mut self, slot: usize) -> Result<&mut DeviceWorkspace> {
        self.workspaces
            .get_mut(slot)
            .ok_or_else(|| anyhow!("Invalid all-to-all slot {}", slot))
    }

    fn slot_state(&self, slot: usize) -> Result<&Mutex<SlotGenerationState>> {
        self.slot_states
            .get(slot)
            .ok_or_else(|| anyhow!("Invalid all-to-all slot {}", slot))
    }

    fn begin_dispatch_handle(
        &self,
        slot: usize,
        num_tokens: usize,
    ) -> Result<DispatchHandleState> {
        let generation = self.next_generation.fetch_add(1, Ordering::Relaxed);
        let mut state = self.slot_state(slot)?.lock().unwrap();
        match *state {
            SlotGenerationState::Idle => {
                *state = SlotGenerationState::DispatchSent { generation, num_tokens };
                Ok(DispatchHandleState { slot, generation })
            }
            other => Err(anyhow!(
                "All-to-all slot {slot} expected Idle before dispatch, found {other:?}"
            )),
        }
    }

    fn validate_dispatch_handle(
        &self,
        slot: usize,
        generation: u64,
        expected: &'static str,
    ) -> Result<usize> {
        let state = self.slot_state(slot)?.lock().unwrap();
        match (*state, expected) {
            (
                SlotGenerationState::DispatchSent {
                    generation: state_generation,
                    num_tokens,
                },
                "DispatchSent",
            )
            | (
                SlotGenerationState::DispatchReceived {
                    generation: state_generation,
                    num_tokens,
                },
                "DispatchReceived",
            )
            | (
                SlotGenerationState::CombineSent {
                    generation: state_generation,
                    num_tokens,
                },
                "CombineSent",
            ) if state_generation == generation => Ok(num_tokens),
            (state, _) => Err(anyhow!(
                "All-to-all stale or invalid handle for slot {slot}: \
                 expected {expected} generation {generation}, found {state:?}"
            )),
        }
    }

    fn validate_live_dispatch_handle(
        &self,
        slot: usize,
        generation: u64,
    ) -> Result<usize> {
        let state = self.slot_state(slot)?.lock().unwrap();
        match *state {
            SlotGenerationState::DispatchSent {
                generation: state_generation,
                num_tokens,
            }
            | SlotGenerationState::DispatchReceived {
                generation: state_generation,
                num_tokens,
            }
            | SlotGenerationState::CombineSent {
                generation: state_generation,
                num_tokens,
            } if state_generation == generation => Ok(num_tokens),
            current => Err(anyhow!(
                "All-to-all stale or invalid handle for slot {slot}: \
                 expected live dispatch generation {generation}, found {current:?}"
            )),
        }
    }

    fn transition_dispatch_received(&self, slot: usize, generation: u64) -> Result<()> {
        let mut state = self.slot_state(slot)?.lock().unwrap();
        match *state {
            SlotGenerationState::DispatchSent {
                generation: state_generation,
                num_tokens,
            } if state_generation == generation => {
                *state =
                    SlotGenerationState::DispatchReceived { generation, num_tokens };
                Ok(())
            }
            current => Err(anyhow!(
                "All-to-all stale or invalid dispatch recv for slot {slot}: \
                 generation {generation}, found {current:?}"
            )),
        }
    }

    fn transition_combine_sent(&self, slot: usize, generation: u64) -> Result<()> {
        let mut state = self.slot_state(slot)?.lock().unwrap();
        match *state {
            SlotGenerationState::DispatchReceived {
                generation: state_generation,
                num_tokens,
            } if state_generation == generation => {
                *state = SlotGenerationState::CombineSent { generation, num_tokens };
                Ok(())
            }
            current => Err(anyhow!(
                "All-to-all stale or invalid combine send for slot {slot}: \
                 generation {generation}, found {current:?}"
            )),
        }
    }

    fn release_handle(&self, slot: usize, generation: u64) -> Result<()> {
        let mut state = self.slot_state(slot)?.lock().unwrap();
        match *state {
            SlotGenerationState::CombineSent {
                generation: state_generation, ..
            } if state_generation == generation => {
                *state = SlotGenerationState::Idle;
                Ok(())
            }
            current => Err(anyhow!(
                "All-to-all stale or invalid combine recv for slot {slot}: \
                 generation {generation}, found {current:?}"
            )),
        }
    }
    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
    pub fn dispatch_send(
        &mut self,
        num_tokens: usize,
        x_ptr: *const c_void,
        x_stride: usize,
        x_scale_ptr: *const c_void,
        x_scale_stride_elem: usize,
        x_scale_stride_token: usize,
        indices: *const i32,
        indices_stride: usize,
        weights: *const f32,
        weights_stride: usize,
        bound_m_ptr: *const i32,
        stream: u64,
    ) -> Result<DispatchHandleState> {
        if num_tokens > self.max_num_tokens {
            return Err(anyhow!("Number of tokens exceeds maximum allowed"));
        }
        let slot = self.slot_pool.acquire();
        self.dispatch_send_on_reserved_slot(
            slot,
            num_tokens,
            x_ptr,
            x_stride,
            x_scale_ptr,
            x_scale_stride_elem,
            x_scale_stride_token,
            indices,
            indices_stride,
            weights,
            weights_stride,
            bound_m_ptr,
            stream,
        )?;
        self.begin_dispatch_handle(slot, num_tokens)
    }

    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
    pub fn dispatch_send_on_slot(
        &mut self,
        slot: usize,
        num_tokens: usize,
        x_ptr: *const c_void,
        x_stride: usize,
        x_scale_ptr: *const c_void,
        x_scale_stride_elem: usize,
        x_scale_stride_token: usize,
        indices: *const i32,
        indices_stride: usize,
        weights: *const f32,
        weights_stride: usize,
        bound_m_ptr: *const i32,
        stream: u64,
    ) -> Result<DispatchHandleState> {
        if num_tokens > self.max_num_tokens {
            return Err(anyhow!("Number of tokens exceeds maximum allowed"));
        }
        if !self.slot_pool.acquire_specific(slot) {
            return Err(anyhow!(
                "All-to-all slot {slot} is already in use or does not exist"
            ));
        }
        self.dispatch_send_on_reserved_slot(
            slot,
            num_tokens,
            x_ptr,
            x_stride,
            x_scale_ptr,
            x_scale_stride_elem,
            x_scale_stride_token,
            indices,
            indices_stride,
            weights,
            weights_stride,
            bound_m_ptr,
            stream,
        )?;
        self.begin_dispatch_handle(slot, num_tokens)
    }

    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
    fn dispatch_send_on_reserved_slot(
        &mut self,
        slot: usize,
        num_tokens: usize,
        x_ptr: *const c_void,
        x_stride: usize,
        x_scale_ptr: *const c_void,
        x_scale_stride_elem: usize,
        x_scale_stride_token: usize,
        indices: *const i32,
        indices_stride: usize,
        weights: *const f32,
        weights_stride: usize,
        bound_m_ptr: *const i32,
        stream: u64,
    ) -> Result<()> {
        if num_tokens > self.max_num_tokens {
            return Err(anyhow!("Number of tokens exceeds maximum allowed"));
        }
        let num_blocks = self.num_blocks;
        let hidden_dim = self.hidden_dim;
        let hidden_dim_scale = self.hidden_dim_scale;
        let num_experts = self.num_experts;
        let num_experts_per_token = self.num_experts_per_token;
        let max_private_tokens = self.max_private_tokens;
        let rank = self.rank;
        let dp_size = self.dp_size;
        let node_size = self.node_size;
        let world_size = self.world_size;
        let in_elemsize = self.in_elemsize;
        let scale_elemsize = self.scale_elemsize;
        let worker = self.worker(slot)?.clone();
        let epoch = worker.epoch();
        let workspace = self.workspace_mut(slot)?;
        let trace = std::env::var_os("PPLX_GARDEN_TRACE").is_some();

        if trace {
            eprintln!(
                "PPLX dispatch_send rank={} slot={} epoch={} launching kernel num_tokens={} stream={}",
                rank, slot, epoch, num_tokens, stream,
            );
        }

        cuda_check!(a2a_kernels::a2a_dispatch_send(
            num_blocks,
            hidden_dim,
            hidden_dim_scale,
            num_experts,
            num_experts_per_token,
            max_private_tokens,
            rank,
            dp_size,
            node_size,
            world_size,
            num_tokens,
            bound_m_ptr,
            x_ptr as *const u8,
            in_elemsize,
            x_stride,
            x_scale_ptr as *const u8,
            scale_elemsize,
            x_scale_stride_elem,
            x_scale_stride_token,
            indices,
            indices_stride,
            weights,
            weights_stride,
            workspace.token_offset.get_mut_ptr(),
            worker.buffers.num_routed_ptr,
            workspace.expert_offsets.get_mut_ptr(),
            workspace.combine_recv_position.get_mut_ptr(),
            worker.slot.dispatch_route_done.get_device_ptr(),
            worker.slot.dispatch_send_done.get_device_ptr(),
            worker.slot.tx_ready.get_device_ptr(),
            worker.buffers.send_buffer_ptr as *mut u8,
            workspace.dispatch_send_counter.get_mut_ptr(),
            workspace.sync_counter.get_mut_ptr(),
            workspace.get_sync_ptr(),
            workspace.get_recv_ptr() as *mut *mut u8,
            workspace.epoch_counter.get_mut_ptr(),
            workspace.current_epoch.get_mut_ptr(),
            stream,
        ))
        .map_err(|e| anyhow!("a2a_dispatch_send slot {slot}: {e}"))?;
        if trace {
            eprintln!(
                "PPLX dispatch_send rank={} slot={} epoch={} kernel launch returned",
                rank, slot, epoch,
            );
        }

        if worker.failed() {
            return Err(anyhow!(
                "a2a_dispatch_send slot {slot}: fabric-lib transfer error"
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
    pub fn dispatch_recv(
        &mut self,
        slot: usize,
        generation: u64,
        out_num_tokens_ptr: *mut i32,
        out_x_ptr: *mut c_void,
        out_x_stride: usize,
        out_x_scale_ptr: *mut c_void,
        out_x_scale_stride_elem: usize,
        out_x_scale_stride_token: usize,
        stream: u64,
    ) -> Result<()> {
        self.validate_dispatch_handle(slot, generation, "DispatchSent")?;
        let num_blocks = self.num_blocks;
        let hidden_dim = self.hidden_dim;
        let hidden_dim_scale = self.hidden_dim_scale;
        let in_elemsize = self.in_elemsize;
        let scale_elemsize = self.scale_elemsize;
        let num_experts = self.num_experts;
        let rank = self.rank;
        let dp_size = self.dp_size;
        let node_size = self.node_size;
        let world_size = self.world_size;
        let worker = self.worker(slot)?.clone();
        let workspace = self.workspace_mut(slot)?;

        cuda_check!(a2a_kernels::a2a_dispatch_recv(
            num_blocks,
            hidden_dim,
            hidden_dim_scale,
            in_elemsize,
            scale_elemsize,
            num_experts,
            rank,
            dp_size,
            node_size,
            world_size,
            out_num_tokens_ptr,
            out_x_ptr as *mut u8,
            out_x_stride,
            out_x_scale_ptr as *mut u8,
            out_x_scale_stride_elem,
            out_x_scale_stride_token,
            worker.slot.tokens_per_expert.get_device_ptr(),
            worker.buffers.send_buffer_ptr as *mut u8,
            worker.buffers.recv_buffer_ptr as *mut u8,
            worker.slot.source_rank.get_device_ptr(),
            worker.slot.source_dispatch_offset.get_device_ptr(),
            worker.slot.padded_index.get_device_ptr(),
            worker.slot.source_token_index.get_device_ptr(),
            worker.buffers.num_routed_ptr,
            worker.slot.num_recv_tokens.get_device_ptr(),
            worker.slot.num_recv_tokens_ready.get_device_ptr(),
            worker.slot.dispatch_recv_flag.get_device_ptr(),
            worker.slot.dispatch_recv_done.get_device_ptr(),
            workspace.dispatch_recv_counter.get_mut_ptr(),
            workspace.sync_counter.get_mut_ptr(),
            workspace.get_sync_ptr(),
            workspace.get_send_ptr() as *mut *mut u8,
            workspace.current_epoch.get_mut_ptr(),
            stream,
        ))
        .map_err(|e| anyhow!("a2a_dispatch_recv slot {slot}: {e}"))?;

        if worker.failed() {
            return Err(anyhow!(
                "a2a_dispatch_recv slot {slot}: fabric-lib transfer error"
            ));
        }

        self.transition_dispatch_received(slot, generation)?;
        Ok(())
    }

    #[allow(
        unused_variables,
        clippy::too_many_arguments,
        clippy::not_unsafe_ptr_arg_deref
    )]
    pub fn combine_send(
        &mut self,
        slot: usize,
        generation: u64,
        expert_x_ptr: *const c_void,
        expert_x_stride: usize,
        stream: u64,
    ) -> Result<()> {
        self.validate_dispatch_handle(slot, generation, "DispatchReceived")?;
        let num_blocks = self.num_blocks;
        let hidden_dim = self.hidden_dim;
        let out_elemsize = self.out_elemsize;
        let rank = self.rank;
        let node_size = self.node_size;
        let dp_size = self.dp_size;
        let worker = self.worker(slot)?.clone();
        let epoch = worker.epoch();
        let workspace = self.workspace_mut(slot)?;
        let trace = std::env::var_os("PPLX_GARDEN_TRACE").is_some();

        if trace {
            eprintln!(
                "PPLX combine_send rank={} slot={} epoch={} launching kernel",
                rank, slot, epoch
            );
        }

        cuda_check!(a2a_kernels::a2a_combine_send(
            num_blocks,
            hidden_dim,
            out_elemsize,
            rank,
            node_size,
            dp_size,
            expert_x_ptr as *const u8,
            expert_x_stride,
            worker.slot.tx_ready.get_device_ptr(),
            worker.buffers.send_buffer_ptr as *mut u8,
            worker.buffers.recv_buffer_ptr as *mut u8,
            worker.slot.source_rank.get_device_ptr(),
            worker.slot.combine_send_offset.get_device_ptr(),
            worker.slot.padded_index.get_device_ptr(),
            worker.slot.num_recv_tokens.get_device_ptr(),
            worker.slot.combine_send_done.get_device_ptr(),
            workspace.token_counter.get_mut_ptr(),
            workspace.sync_counter.get_mut_ptr(),
            workspace.get_sync_ptr(),
            workspace.get_recv_ptr() as *mut *mut u8,
            workspace.current_epoch.get_mut_ptr(),
            stream,
        ))
        .map_err(|e| anyhow!("a2a_combine_send slot {slot}: {e}"))?;
        if trace {
            eprintln!(
                "PPLX combine_send rank={} slot={} epoch={} kernel launch returned",
                rank, slot, epoch
            );
        }

        if worker.failed() {
            return Err(anyhow!(
                "a2a_combine_send slot {slot}: fabric-lib transfer error"
            ));
        }

        self.transition_combine_sent(slot, generation)?;
        Ok(())
    }

    #[allow(
        unused_variables,
        clippy::too_many_arguments,
        clippy::not_unsafe_ptr_arg_deref
    )]
    pub fn combine_recv(
        &mut self,
        slot: usize,
        generation: u64,
        num_tokens: usize,
        num_recv_tokens: usize,
        expert_y_dtype: ScalarType,
        out_tokens_ptr: *mut c_void,
        out_tokens_stride: usize,
        indices_ptr: *const i32,
        indices_stride: usize,
        weights_ptr: *const f32,
        weights_stride: usize,
        bound_m_ptr: *const i32,
        accumulate: bool,
        stream: u64,
    ) -> Result<()> {
        let expected_num_tokens =
            self.validate_dispatch_handle(slot, generation, "CombineSent")?;
        if expected_num_tokens != num_tokens {
            return Err(anyhow!(
                "All-to-all combine recv token count mismatch for slot {slot}: \
                 handle has {expected_num_tokens}, call has {num_tokens}"
            ));
        }
        let num_blocks = self.num_blocks;
        let hidden_dim = self.hidden_dim;
        let out_elemsize = self.out_elemsize;
        let out_dtype = self.out_dtype;
        let num_experts = self.num_experts;
        let num_experts_per_token = self.num_experts_per_token;
        let max_recv_tokens = self.max_recv_tokens;
        let rank = self.rank;
        let node_size = self.node_size;
        let world_size = self.world_size;
        let worker = self.worker(slot)?.clone();
        let epoch = worker.epoch();
        let workspace = self.workspace_mut(slot)?;
        if std::env::var_os("PPLX_GARDEN_TRACE").is_some() {
            eprintln!(
                "PPLX combine_recv rank={} slot={} epoch={} num_tokens={} num_recv_tokens={} num_experts={} num_experts_per_token={} indices_ptr={:?}",
                rank,
                slot,
                epoch,
                num_tokens,
                num_recv_tokens,
                num_experts,
                num_experts_per_token,
                indices_ptr,
            );
            eprintln!(
                "PPLX combine_recv ptrs rank={} slot={} epoch={} recv_buffer={:?} padded_index={:?} combine_send_offset={:?} source_rank={:?} max_recv_tokens={}",
                rank,
                slot,
                epoch,
                worker.buffers.recv_buffer_ptr,
                worker.slot.padded_index.get_device_ptr(),
                worker.slot.combine_send_offset.get_device_ptr(),
                worker.slot.source_rank.get_device_ptr(),
                max_recv_tokens,
            );
        }

        cuda_check!(a2a_kernels::a2a_combine_recv(
            num_blocks,
            hidden_dim,
            out_elemsize,
            expert_y_dtype,
            out_dtype,
            num_experts,
            num_experts_per_token,
            rank,
            node_size,
            world_size,
            num_tokens,
            num_recv_tokens,
            max_recv_tokens,
            bound_m_ptr,
            indices_ptr,
            indices_stride,
            weights_ptr,
            weights_stride,
            out_tokens_ptr as *mut u8,
            out_tokens_stride,
            accumulate,
            worker.buffers.recv_buffer_ptr as *mut u8,
            workspace.combine_recv_position.get_mut_ptr(),
            worker.slot.combine_recv_flag.get_device_ptr(),
            worker.slot.combine_recv_done.get_device_ptr(),
            workspace.sync_counter.get_mut_ptr(),
            workspace.get_sync_ptr(),
            workspace.current_epoch.get_mut_ptr(),
            stream,
        ))
        .map_err(|e| anyhow!("a2a_combine_recv slot {slot}: {e}"))?;

        if worker.failed() {
            return Err(anyhow!(
                "a2a_combine_recv slot {slot}: fabric-lib transfer error"
            ));
        }

        self.release_handle(slot, generation)?;
        Ok(())
    }

    pub fn low_latency_workspace_layout(&self) -> LowLatencyWorkspaceLayout {
        let num_ep_groups = self.world_size / self.dp_size;
        let num_local_experts = self.num_experts.div_ceil(num_ep_groups);

        let mut tensors = Vec::new();
        let mut total_bytes = 0;
        add_workspace_tensor(
            &mut tensors,
            &mut total_bytes,
            "expert_num_tokens",
            vec![num_local_experts],
            "int32",
            ScalarType::I32.element_size(),
        );
        add_workspace_tensor(
            &mut tensors,
            &mut total_bytes,
            "expert_x",
            vec![num_local_experts, self.max_tokens_per_expert, self.hidden_dim],
            "activation",
            self.in_elemsize,
        );
        add_workspace_tensor(
            &mut tensors,
            &mut total_bytes,
            "indices",
            vec![self.max_num_tokens, self.num_experts_per_token],
            "uint32",
            ScalarType::U32.element_size(),
        );
        add_workspace_tensor(
            &mut tensors,
            &mut total_bytes,
            "weights",
            vec![self.max_num_tokens, self.num_experts_per_token],
            "float32",
            ScalarType::F32.element_size(),
        );
        add_workspace_tensor(
            &mut tensors,
            &mut total_bytes,
            "dp_x",
            vec![self.max_num_tokens, self.hidden_dim],
            "activation",
            self.in_elemsize,
        );
        if self.scale_elemsize != 0 {
            add_workspace_tensor(
                &mut tensors,
                &mut total_bytes,
                "expert_x_scale",
                vec![
                    num_local_experts,
                    self.max_tokens_per_expert,
                    self.hidden_dim_scale,
                ],
                "scale",
                self.scale_elemsize,
            );
            add_workspace_tensor(
                &mut tensors,
                &mut total_bytes,
                "dp_x_scale",
                vec![self.max_num_tokens, self.hidden_dim_scale],
                "scale",
                self.scale_elemsize,
            );
        }

        LowLatencyWorkspaceLayout {
            alignment: LOW_LATENCY_WORKSPACE_ALIGNMENT,
            total_bytes: align_up(total_bytes, LOW_LATENCY_WORKSPACE_ALIGNMENT),
            tensors,
        }
    }

    pub fn get_perf_stats(&self) -> AllToAllPerfStats {
        let mut stats = AllToAllPerfStats {
            local_dispatch_bytes: 0,
            nvlink_dispatch_bytes: 0,
            network_dispatch_bytes: 0,
            local_combine_bytes: 0,
            nvlink_combine_bytes: 0,
            network_combine_bytes: 0,
            peer_dispatch_bytes: vec![0; self.world_size],
            peer_combine_bytes: vec![0; self.world_size],
            wait_dispatch_route_ns: 0,
            route_exchange_ns: 0,
            process_routing_ns: 0,
            wait_dispatch_send_ns: 0,
            dispatch_transfer_wait_ns: 0,
            wait_dispatch_recv_ns: 0,
            dispatch_barrier_ns: 0,
            wait_combine_send_ns: 0,
            combine_transfer_wait_ns: 0,
            wait_combine_recv_ns: 0,
            combine_barrier_ns: 0,
        };
        for worker in &self.workers {
            stats.local_dispatch_bytes +=
                worker.accumulated_local_dispatch_bytes.load(Ordering::Relaxed);
            stats.nvlink_dispatch_bytes +=
                worker.accumulated_nvlink_dispatch_bytes.load(Ordering::Relaxed);
            stats.network_dispatch_bytes +=
                worker.accumulated_network_dispatch_bytes.load(Ordering::Relaxed);
            stats.local_combine_bytes +=
                worker.accumulated_local_combine_bytes.load(Ordering::Relaxed);
            stats.nvlink_combine_bytes +=
                worker.accumulated_nvlink_combine_bytes.load(Ordering::Relaxed);
            stats.network_combine_bytes +=
                worker.accumulated_network_combine_bytes.load(Ordering::Relaxed);
            for (dst, value) in stats
                .peer_dispatch_bytes
                .iter_mut()
                .zip(worker.peer_dispatch_bytes.iter())
            {
                *dst += value.load(Ordering::Relaxed);
            }
            for (dst, value) in stats
                .peer_combine_bytes
                .iter_mut()
                .zip(worker.peer_combine_bytes.iter())
            {
                *dst += value.load(Ordering::Relaxed);
            }
            stats.wait_dispatch_route_ns +=
                worker.accumulated_wait_dispatch_route_ns.load(Ordering::Relaxed);
            stats.route_exchange_ns +=
                worker.accumulated_route_exchange_ns.load(Ordering::Relaxed);
            stats.process_routing_ns +=
                worker.accumulated_process_routing_ns.load(Ordering::Relaxed);
            stats.wait_dispatch_send_ns +=
                worker.accumulated_wait_dispatch_send_ns.load(Ordering::Relaxed);
            stats.dispatch_transfer_wait_ns +=
                worker.accumulated_dispatch_transfer_wait_ns.load(Ordering::Relaxed);
            stats.wait_dispatch_recv_ns +=
                worker.accumulated_wait_dispatch_recv_ns.load(Ordering::Relaxed);
            stats.dispatch_barrier_ns +=
                worker.accumulated_dispatch_barrier_ns.load(Ordering::Relaxed);
            stats.wait_combine_send_ns +=
                worker.accumulated_wait_combine_send_ns.load(Ordering::Relaxed);
            stats.combine_transfer_wait_ns +=
                worker.accumulated_combine_transfer_wait_ns.load(Ordering::Relaxed);
            stats.wait_combine_recv_ns +=
                worker.accumulated_wait_combine_recv_ns.load(Ordering::Relaxed);
            stats.combine_barrier_ns +=
                worker.accumulated_combine_barrier_ns.load(Ordering::Relaxed);
        }
        stats
    }

    pub fn uses_node_route_exchange(&self) -> bool {
        self.workers.first().is_some_and(|worker| worker.has_node_route_exchange())
    }

    pub fn debug_low_latency_route_layout_plan(
        &self,
        num_routed: Vec<Vec<u32>>,
    ) -> Result<LowLatencyRouteLayoutPlan> {
        let num_ep_groups = self.world_size / self.dp_size;
        if num_routed.len() != num_ep_groups {
            return Err(anyhow!(
                "Expected {} source-group route rows, got {}",
                num_ep_groups,
                num_routed.len()
            ));
        }
        for (source_group, counts) in num_routed.iter().enumerate() {
            if counts.len() != self.num_experts {
                return Err(anyhow!(
                    "Expected {} expert counts for source group {}, got {}",
                    self.num_experts,
                    source_group,
                    counts.len()
                ));
            }
        }

        let dp_group = self.rank / self.dp_size;
        let dp_rank = self.rank % self.dp_size;
        let plan = compute_low_latency_route_layout_plan(
            dp_group,
            dp_rank,
            self.dp_size,
            self.node_size,
            self.world_size,
            self.num_experts,
            self.expert_padding,
            self.max_tokens_per_expert,
            self.max_private_tokens,
            1,
            |source_group, expert| num_routed[source_group][expert],
        );
        Ok(plan)
    }

    pub fn debug_low_latency_route_layout_plan_for_handle(
        &self,
        slot: usize,
        generation: u64,
    ) -> Result<LowLatencyRouteLayoutPlan> {
        self.validate_live_dispatch_handle(slot, generation)?;
        let worker = self.worker(slot)?;
        let num_ep_groups = self.world_size / self.dp_size;
        let dp_group = self.rank / self.dp_size;
        let dp_rank = self.rank % self.dp_size;
        let mut plan = compute_low_latency_route_layout_plan(
            dp_group,
            dp_rank,
            self.dp_size,
            self.node_size,
            self.world_size,
            self.num_experts,
            self.expert_padding,
            self.max_tokens_per_expert,
            self.max_private_tokens,
            1,
            |source_group, expert| {
                debug_assert!(source_group < num_ep_groups);
                worker.get_num_routed(source_group, expert)
            },
        );
        plan.layout_range = (0..plan.layout_range.len())
            .map(|index| worker.slot.layout_range.get(index))
            .collect();
        plan.source_token_index = plan
            .final_index
            .iter()
            .map(|index| worker.slot.source_token_index.get(*index as usize))
            .collect();
        Ok(plan)
    }
}

pub struct AllToAllPerfStats {
    pub local_dispatch_bytes: u64,
    pub nvlink_dispatch_bytes: u64,
    pub network_dispatch_bytes: u64,
    pub local_combine_bytes: u64,
    pub nvlink_combine_bytes: u64,
    pub network_combine_bytes: u64,
    pub peer_dispatch_bytes: Vec<u64>,
    pub peer_combine_bytes: Vec<u64>,
    pub wait_dispatch_route_ns: u64,
    pub route_exchange_ns: u64,
    pub process_routing_ns: u64,
    pub wait_dispatch_send_ns: u64,
    pub dispatch_transfer_wait_ns: u64,
    pub wait_dispatch_recv_ns: u64,
    pub dispatch_barrier_ns: u64,
    pub wait_combine_send_ns: u64,
    pub combine_transfer_wait_ns: u64,
    pub wait_combine_recv_ns: u64,
    pub combine_barrier_ns: u64,
}

impl Drop for AllToAllContext {
    fn drop(&mut self) {
        let _ = self.destroy();
    }
}

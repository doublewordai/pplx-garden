use std::{
    collections::VecDeque,
    ffi::c_void,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use cuda_lib::{
    CudaDeviceId, Device,
    gdr::{GdrCopyContext, GdrEpoch, GdrFlag, GdrVec},
};
use fabric_lib::{
    RdmaEngine, TransferEngine,
    api::{
        BarrierTransferRequest, DomainAddress, DomainGroupRouting, GdrCounter,
        GroupTransferRouting, ImmCounter, MemoryRegionDescriptor, MemoryRegionHandle,
        ScatterTarget, ScatterTransferRequest, TransferRequest,
    },
};
use nvtx::{range_end, range_start};

use crate::a2a_handles::AllToAllRankHandle;

fn compute_padded_offsets(
    tokens_per_expert: &[u32],
    expert_padding: usize,
    max_tokens_per_expert: usize,
) -> Vec<u32> {
    let mut padded_offset = Vec::with_capacity(tokens_per_expert.len());
    let mut base_expert_offset = 0;
    for (local_expert, count) in tokens_per_expert.iter().enumerate() {
        if max_tokens_per_expert > 0 {
            padded_offset.push((local_expert * max_tokens_per_expert) as u32);
        } else {
            let padded_count =
                (*count as usize).div_ceil(expert_padding) * expert_padding;
            padded_offset.push(base_expert_offset);
            base_expert_offset += padded_count as u32;
        }
    }
    padded_offset
}

fn pack_layout_range(count: u32, offset: u32) -> u64 {
    (count as u64) | ((offset as u64) << 32)
}

#[derive(Debug, PartialEq)]
struct ReceiveRoutePlan {
    tokens_from_group: Vec<u32>,
    src_group_offset: Vec<u32>,
    dst_group_offset: Vec<u32>,
    tokens_per_expert: Vec<u32>,
    num_recv_tokens: usize,
    num_recv_efa_tokens: usize,
    num_recv_tx: u32,
    source_dispatch_offset: Vec<u32>,
    combine_send_offset: Vec<u32>,
    source_rank: Vec<u32>,
    padded_index: Vec<u32>,
    tokens_to_rank: Vec<u32>,
    dispatch_src_offset: Vec<u32>,
    layout_range: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LowLatencyRouteLayoutPlan {
    pub source_group_order: Vec<u32>,
    pub source_rank: Vec<u32>,
    pub source_group: Vec<u32>,
    pub final_index: Vec<u32>,
    pub tokens_per_source_group_per_local_expert: Vec<Vec<u32>>,
    pub tokens_per_expert: Vec<u32>,
    pub layout_range: Vec<u64>,
    pub num_recv_tokens: usize,
}

#[derive(Debug, PartialEq)]
struct RemoteTransferPlan {
    peer_rank: usize,
    length_tokens: usize,
    src_token_offset: u64,
    dst_token_offset: u64,
}

#[allow(clippy::too_many_arguments)]
fn compute_initial_dispatch_transfer_plans(
    dp_group: usize,
    dp_rank: usize,
    dp_size: usize,
    node_size: usize,
    world_size: usize,
    num_experts: usize,
    max_private_tokens: usize,
    mut get_num_routed: impl FnMut(usize, usize) -> u32,
) -> Vec<RemoteTransferPlan> {
    let num_dp_groups = world_size / dp_size;
    let experts_per_rank = num_experts.div_ceil(num_dp_groups);
    let rank = dp_group * dp_size + dp_rank;
    let rank_node = rank / node_size;

    let mut tokens_per_rank = vec![0; world_size];
    let mut rank_offset = vec![0; world_size];
    let mut offset = 0;
    for peer_group in 0..num_dp_groups {
        let peer_rank = peer_group * dp_size + dp_rank;
        let first_expert = peer_group * experts_per_rank;
        let last_expert = (first_expert + experts_per_rank).min(num_experts);

        let mut tokens_on_rank = 0;
        for expert in first_expert..last_expert {
            tokens_on_rank += get_num_routed(dp_group, expert) as usize;
        }
        tokens_per_rank[peer_rank] = tokens_on_rank;
        rank_offset[peer_rank] = offset;
        offset += tokens_on_rank;
    }

    let mut transfers = Vec::with_capacity(world_size - 1);
    for peer_node in 1..(world_size / node_size) {
        for index in (dp_rank..node_size).step_by(dp_size) {
            let peer_rank = ((rank_node + peer_node) * node_size + index) % world_size;
            transfers.push(RemoteTransferPlan {
                peer_rank,
                length_tokens: tokens_per_rank[peer_rank].min(max_private_tokens),
                src_token_offset: rank_offset[peer_rank] as u64,
                dst_token_offset: (dp_group * max_private_tokens) as u64,
            });
        }
    }

    transfers
}

#[allow(clippy::too_many_arguments)]
fn compute_receive_route_plan(
    dp_group: usize,
    dp_rank: usize,
    dp_size: usize,
    node_size: usize,
    world_size: usize,
    num_experts: usize,
    expert_padding: usize,
    max_tokens_per_expert: usize,
    max_private_tokens: usize,
    nets_per_gpu: u32,
    mut get_num_routed: impl FnMut(usize, usize) -> u32,
) -> ReceiveRoutePlan {
    let num_dp_groups = world_size / dp_size;
    let experts_per_rank = num_experts.div_ceil(num_dp_groups);
    let first_local_expert = dp_group * experts_per_rank;
    let last_local_expert = (first_local_expert + experts_per_rank).min(num_experts);
    let num_local_experts = last_local_expert - first_local_expert;
    let rank = dp_group * dp_size + dp_rank;
    let rank_node = rank / node_size;
    let groups_per_node = node_size / dp_size;
    let num_nodes = world_size / node_size;

    let mut tokens_from_group = vec![0; num_dp_groups];
    let mut src_group_offset = vec![0; num_dp_groups];
    let mut dst_group_offset = vec![0; num_dp_groups];
    let mut tokens_per_expert = vec![0; experts_per_rank];
    let mut num_recv_tokens = 0;
    let mut num_recv_tx = 0;
    let mut source_expert_offset = vec![0; num_experts];
    let mut tokens_to_rank: Vec<u32> = vec![0; world_size];
    let mut dispatch_src_offset = vec![0; world_size];

    let mut rank_offset = 0;
    for src_group in 0..num_dp_groups {
        let group_node = src_group / groups_per_node;
        let mut num_tokens = 0usize;
        let source_rank = src_group * dp_size + dp_rank;
        dispatch_src_offset[source_rank] = rank_offset;

        let first_expert = src_group * experts_per_rank;
        let last_expert = (first_expert + experts_per_rank).min(num_experts);
        for expert in first_expert..last_expert {
            let n = get_num_routed(dp_group, expert);
            source_expert_offset[expert] = rank_offset;
            tokens_to_rank[source_rank] += n;
            rank_offset += n;
        }

        let mut offset = 0;
        for expert in 0..first_local_expert {
            offset += get_num_routed(src_group, expert);
        }
        dst_group_offset[src_group] = offset;

        for expert in first_local_expert..last_local_expert {
            let n = get_num_routed(src_group, expert);
            num_tokens += n as usize;
            tokens_per_expert[expert - first_local_expert] += n;
        }
        tokens_from_group[src_group] += num_tokens as u32;
        src_group_offset[src_group] = num_recv_tokens as u32;
        num_recv_tokens += num_tokens;

        if src_group != dp_group && group_node != rank_node {
            // Private buffer scatter shards by peers; overflow scatter shards by bytes.
            num_recv_tx += 1;
            if num_tokens > max_private_tokens {
                num_recv_tx += nets_per_gpu;
            }
        }
    }

    let padded_offset = compute_padded_offsets(
        &tokens_per_expert,
        expert_padding,
        max_tokens_per_expert,
    );
    let base_offset = (max_private_tokens * num_dp_groups) as u64;

    let mut source_dispatch_offset = vec![0; num_recv_tokens];
    let mut combine_send_offset = vec![0; num_recv_tokens];
    let mut source_rank = vec![0; num_recv_tokens];
    let mut padded_index = vec![0u32; num_recv_tokens];
    let mut layout_range = vec![0u64; num_local_experts * num_dp_groups];

    let mut last = 0;
    let mut src_dispatch_count = vec![0; num_dp_groups];
    let mut src_combine_count = vec![0; num_dp_groups];
    let mut expert_count = vec![0usize; num_local_experts];

    let mut route_group = |peer_group: usize| {
        let mut num_routed = 0;
        for expert in first_local_expert..last_local_expert {
            let private_offset = (max_private_tokens * peer_group) as u32;
            let routed = get_num_routed(peer_group, expert);
            num_routed += routed as usize;

            let local_expert = expert - first_local_expert;
            layout_range[local_expert * num_dp_groups + peer_group] =
                pack_layout_range(routed, expert_count[local_expert] as u32);
            let src_offset = src_group_offset[peer_group];
            let dst_offset = dst_group_offset[peer_group];
            let peer_rank = peer_group * dp_size + dp_rank;
            for _ in 0..routed {
                if peer_rank == rank {
                    let local_offset = source_expert_offset[expert];
                    source_expert_offset[expert] += 1;
                    source_dispatch_offset[last] = local_offset;
                    combine_send_offset[last] = local_offset;
                } else {
                    let index_on_rank = src_dispatch_count[peer_group];
                    src_dispatch_count[peer_group] += 1;

                    if (index_on_rank as usize) < max_private_tokens {
                        source_dispatch_offset[last] = private_offset + index_on_rank;
                    } else if peer_rank / node_size == rank_node {
                        source_dispatch_offset[last] =
                            (dst_offset + index_on_rank) | (1 << 31);
                    } else {
                        source_dispatch_offset[last] = src_offset
                            + index_on_rank
                            + (base_offset as u32 - max_private_tokens as u32);
                    }
                }

                if peer_rank != rank || dp_size > 1 {
                    let combine_index = src_combine_count[peer_group];
                    src_combine_count[peer_group] += 1;
                    if peer_rank / node_size == rank_node {
                        combine_send_offset[last] = dst_offset + combine_index;
                    } else {
                        combine_send_offset[last] = src_offset + combine_index;
                    }
                }
                source_rank[last] = peer_rank as u32;
                padded_index[last] =
                    padded_offset[local_expert] + (expert_count[local_expert] as u32);
                expert_count[local_expert] += 1;
                last += 1;
            }
        }
        num_routed
    };

    let mut num_recv_efa_tokens = 0;
    for i in 1..num_nodes {
        for j in 0..groups_per_node {
            num_recv_efa_tokens +=
                route_group((rank_node + i) % num_nodes * groups_per_node + j);
        }
    }
    for local_group in 1..groups_per_node {
        route_group(
            rank_node * groups_per_node + (dp_group + local_group) % groups_per_node,
        );
    }
    route_group(dp_group);

    ReceiveRoutePlan {
        tokens_from_group,
        src_group_offset,
        dst_group_offset,
        tokens_per_expert,
        num_recv_tokens,
        num_recv_efa_tokens,
        num_recv_tx,
        source_dispatch_offset,
        combine_send_offset,
        source_rank,
        padded_index,
        tokens_to_rank,
        dispatch_src_offset,
        layout_range,
    }
}

fn ordered_source_groups(
    dp_group: usize,
    rank_node: usize,
    groups_per_node: usize,
    num_nodes: usize,
) -> Vec<usize> {
    let mut source_groups = Vec::new();
    for node_offset in 1..num_nodes {
        let node = (rank_node + node_offset) % num_nodes;
        for group_in_node in 0..groups_per_node {
            source_groups.push(node * groups_per_node + group_in_node);
        }
    }
    for local_group_offset in 1..groups_per_node {
        source_groups.push(
            rank_node * groups_per_node
                + (dp_group + local_group_offset) % groups_per_node,
        );
    }
    source_groups.push(dp_group);
    source_groups
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_low_latency_route_layout_plan(
    dp_group: usize,
    dp_rank: usize,
    dp_size: usize,
    node_size: usize,
    world_size: usize,
    num_experts: usize,
    expert_padding: usize,
    max_tokens_per_expert: usize,
    max_private_tokens: usize,
    nets_per_gpu: u32,
    mut get_num_routed: impl FnMut(usize, usize) -> u32,
) -> LowLatencyRouteLayoutPlan {
    let num_dp_groups = world_size / dp_size;
    let experts_per_rank = num_experts.div_ceil(num_dp_groups);
    let first_local_expert = dp_group * experts_per_rank;
    let last_local_expert = (first_local_expert + experts_per_rank).min(num_experts);
    let rank = dp_group * dp_size + dp_rank;
    let rank_node = rank / node_size;
    let groups_per_node = node_size / dp_size;
    let num_nodes = world_size / node_size;

    let mut num_routed = vec![vec![0u32; num_experts]; num_dp_groups];
    for (source_group, row) in num_routed.iter_mut().enumerate() {
        for (expert, count) in row.iter_mut().enumerate() {
            *count = get_num_routed(source_group, expert);
        }
    }

    let plan = compute_receive_route_plan(
        dp_group,
        dp_rank,
        dp_size,
        node_size,
        world_size,
        num_experts,
        expert_padding,
        max_tokens_per_expert,
        max_private_tokens,
        nets_per_gpu,
        |source_group, expert| num_routed[source_group][expert],
    );
    let source_group_order =
        ordered_source_groups(dp_group, rank_node, groups_per_node, num_nodes);
    let tokens_per_source_group_per_local_expert = (0..num_dp_groups)
        .map(|source_group| {
            (first_local_expert..last_local_expert)
                .map(|expert| num_routed[source_group][expert])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    LowLatencyRouteLayoutPlan {
        source_group_order: source_group_order
            .into_iter()
            .map(|source_group| source_group as u32)
            .collect(),
        source_group: plan
            .source_rank
            .iter()
            .map(|rank| *rank / dp_size as u32)
            .collect(),
        source_rank: plan.source_rank,
        final_index: plan.padded_index,
        tokens_per_source_group_per_local_expert,
        tokens_per_expert: plan.tokens_per_expert,
        layout_range: plan.layout_range,
        num_recv_tokens: plan.num_recv_tokens,
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_remote_dispatch_transfer_plans(
    plan: &ReceiveRoutePlan,
    dp_group: usize,
    dp_rank: usize,
    dp_size: usize,
    node_size: usize,
    world_size: usize,
    num_experts: usize,
    max_private_tokens: usize,
    mut get_num_routed: impl FnMut(usize, usize) -> u32,
) -> Vec<RemoteTransferPlan> {
    let num_dp_groups = world_size / dp_size;
    let experts_per_rank = num_experts.div_ceil(num_dp_groups);
    let rank = dp_group * dp_size + dp_rank;
    let rank_node = rank / node_size;
    let base_offset = (max_private_tokens * num_dp_groups) as u64;

    let mut transfers = Vec::with_capacity(world_size - 1);
    for peer_node in 1..(world_size / node_size) {
        for index in (dp_rank..node_size).step_by(dp_size) {
            let peer_rank = ((rank_node + peer_node) * node_size + index) % world_size;
            let num_tokens = plan.tokens_to_rank[peer_rank] as usize;
            if num_tokens <= max_private_tokens {
                continue;
            }

            let peer_group = peer_rank / dp_size;
            let first_expert = peer_group * experts_per_rank;
            let last_expert = (first_expert + experts_per_rank).min(num_experts);

            let mut dst_token_offset = 0;
            for src_group in 0..dp_group {
                for expert in first_expert..last_expert {
                    dst_token_offset += get_num_routed(src_group, expert) as u64;
                }
            }

            transfers.push(RemoteTransferPlan {
                peer_rank,
                length_tokens: num_tokens - max_private_tokens,
                src_token_offset: max_private_tokens as u64
                    + plan.dispatch_src_offset[peer_rank] as u64,
                dst_token_offset: base_offset + dst_token_offset,
            });
        }
    }

    transfers
}

fn compute_remote_combine_transfer_plans(
    plan: &ReceiveRoutePlan,
    dp_group: usize,
    dp_rank: usize,
    dp_size: usize,
    node_size: usize,
    world_size: usize,
) -> Vec<RemoteTransferPlan> {
    let rank = dp_group * dp_size + dp_rank;
    let rank_node = rank / node_size;
    let groups_per_node = node_size / dp_size;
    let num_dp_groups = world_size / dp_size;

    let mut transfers = Vec::with_capacity(world_size - 1);
    for peer_node in 1..(world_size / node_size) {
        for index in 0..groups_per_node {
            let peer_group =
                ((rank_node + peer_node) * groups_per_node + index) % num_dp_groups;
            transfers.push(RemoteTransferPlan {
                peer_rank: peer_group * dp_size + dp_rank,
                length_tokens: plan.tokens_from_group[peer_group] as usize,
                src_token_offset: plan.src_group_offset[peer_group] as u64,
                dst_token_offset: plan.dst_group_offset[peer_group] as u64,
            });
        }
    }

    transfers
}

pub(crate) struct WorkerBuffers {
    pub(crate) num_routed_ptr: *mut u32,
    pub(crate) send_buffer_ptr: *mut c_void,
    pub(crate) recv_buffer_ptr: *mut c_void,
}

unsafe impl Send for WorkerBuffers {}
unsafe impl Sync for WorkerBuffers {}

pub(crate) struct SlotPool {
    state: Mutex<SlotPoolState>,
    condvar: Condvar,
}

struct SlotPoolState {
    free_slots: VecDeque<usize>,
    in_use: Vec<bool>,
}

impl SlotPool {
    pub(crate) fn new(num_slots: usize) -> Self {
        Self {
            state: Mutex::new(SlotPoolState {
                free_slots: (0..num_slots).collect(),
                in_use: vec![false; num_slots],
            }),
            condvar: Condvar::new(),
        }
    }

    pub(crate) fn acquire(&self) -> usize {
        let mut state = self.state.lock().unwrap();
        loop {
            if let Some(slot) = state.free_slots.pop_front() {
                state.in_use[slot] = true;
                return slot;
            }
            state = self.condvar.wait(state).unwrap();
        }
    }

    pub(crate) fn acquire_specific(&self, slot: usize) -> bool {
        let mut state = self.state.lock().unwrap();
        if slot >= state.in_use.len() {
            return false;
        }
        while state.in_use[slot] {
            state = self.condvar.wait(state).unwrap();
        }
        state.in_use[slot] = true;
        if let Some(index) =
            state.free_slots.iter().position(|free_slot| *free_slot == slot)
        {
            state.free_slots.remove(index);
        }
        true
    }

    pub(crate) fn release(&self, slot: usize) {
        let mut state = self.state.lock().unwrap();
        let Some(slot_in_use) = state.in_use.get_mut(slot) else {
            return;
        };
        if !*slot_in_use {
            return;
        }
        *slot_in_use = false;
        if !state.free_slots.contains(&slot) {
            state.free_slots.push_back(slot);
            self.condvar.notify_one();
        }
    }
}

#[allow(dead_code)]
pub(crate) struct MicrobatchSlot {
    pub(crate) dispatch_route_done: GdrEpoch,
    pub(crate) dispatch_send_done: GdrEpoch,
    pub(crate) dispatch_recv_done: GdrEpoch,
    pub(crate) combine_send_done: GdrEpoch,
    pub(crate) combine_recv_done: GdrEpoch,
    pub(crate) tokens_per_expert: GdrVec<u32>,
    pub(crate) source_dispatch_offset: GdrVec<u32>,
    pub(crate) combine_send_offset: GdrVec<u32>,
    pub(crate) source_rank: GdrVec<u32>,
    pub(crate) padded_index: GdrVec<u32>,
    pub(crate) layout_range: GdrVec<u64>,
    pub(crate) num_recv_tokens: GdrVec<u32>,
    pub(crate) num_recv_tokens_ready: GdrEpoch,
    pub(crate) tx_ready: GdrFlag,
    pub(crate) dispatch_recv_flag: Arc<GdrFlag>,
    pub(crate) combine_recv_flag: Arc<GdrFlag>,
}

impl MicrobatchSlot {
    fn new(
        gdr_context: &GdrCopyContext,
        num_local_experts: usize,
        num_ep_groups: usize,
        max_recv_tokens: usize,
    ) -> Result<Self> {
        let dispatch_route_done = GdrEpoch::new(gdr_context)?;
        let dispatch_send_done = GdrEpoch::new(gdr_context)?;
        let dispatch_recv_done = GdrEpoch::new(gdr_context)?;
        let combine_send_done = GdrEpoch::new(gdr_context)?;
        let combine_recv_done = GdrEpoch::new(gdr_context)?;
        let num_recv_tokens_ready = GdrEpoch::new(gdr_context)?;
        let tx_ready = GdrFlag::new(gdr_context)?;
        let dispatch_recv_flag = Arc::new(GdrFlag::new(gdr_context)?);
        let combine_recv_flag = Arc::new(GdrFlag::new(gdr_context)?);

        let tokens_per_expert = GdrVec::new(gdr_context, num_local_experts)?;
        let source_rank = GdrVec::new(gdr_context, max_recv_tokens)?;
        let source_dispatch_offset = GdrVec::new(gdr_context, max_recv_tokens)?;
        let combine_send_offset = GdrVec::new(gdr_context, max_recv_tokens)?;
        let padded_index = GdrVec::new(gdr_context, max_recv_tokens)?;
        let layout_range = GdrVec::new(gdr_context, num_local_experts * num_ep_groups)?;
        let num_recv_tokens = GdrVec::new(gdr_context, 3)?;

        num_recv_tokens.copy(&[0u32, 0u32, 0u32]);
        dispatch_route_done.set(0);
        dispatch_send_done.set(0);
        dispatch_recv_done.set(0);
        combine_send_done.set(0);
        combine_recv_done.set(0);
        num_recv_tokens_ready.set(0);
        tx_ready.set(true);
        dispatch_recv_flag.set(false);
        combine_recv_flag.set(false);

        Ok(Self {
            dispatch_route_done,
            dispatch_send_done,
            dispatch_recv_done,
            combine_send_done,
            combine_recv_done,
            tokens_per_expert,
            source_dispatch_offset,
            combine_send_offset,
            source_rank,
            padded_index,
            layout_range,
            num_recv_tokens,
            num_recv_tokens_ready,
            tx_ready,
            dispatch_recv_flag,
            combine_recv_flag,
        })
    }

    fn stop(&self, epoch: u32) {
        self.dispatch_route_done.set(epoch);
        self.dispatch_send_done.set(epoch);
        self.dispatch_recv_done.set(epoch);
        self.combine_send_done.set(epoch);
        self.combine_recv_done.set(epoch);
    }
}

pub(crate) struct WorkerState {
    slot_idx: usize,
    slot_pool: Arc<SlotPool>,
    transfer_engine: Arc<TransferEngine>,
    max_num_tokens: usize,
    max_recv_tokens: usize,
    max_tokens_per_expert: usize,
    max_private_tokens: usize,
    hidden_dim: usize,
    hidden_dim_scale: usize,
    in_elemsize: usize,
    out_elemsize: usize,
    scale_elemsize: usize,
    num_experts: usize,
    num_experts_per_token: usize,
    expert_padding: usize,
    rank: usize,
    dp_rank: usize,
    dp_group: usize,
    dp_size: usize,
    node_size: usize,
    world_size: usize,
    rank_handles: Vec<AllToAllRankHandle>,
    stop_flag: AtomicBool,
    device: u8,
    dispatch_imm: u32,
    combine_imm: u32,
    route_imm: u32,
    dispatch_barrier_imm: u32,
    combine_barrier_imm: u32,
    num_routed_mr: MemoryRegionHandle,
    send_buffer_mr: MemoryRegionHandle,
    recv_buffer_mr: MemoryRegionHandle,
    pub(crate) buffers: WorkerBuffers,
    pub(crate) slot: MicrobatchSlot,
    epoch: AtomicU32,
    route_counter: ImmCounter,
    dispatch_counter: GdrCounter,
    combine_counter: ImmCounter,
    dispatch_barrier_counter: ImmCounter,
    combine_barrier_counter: ImmCounter,
    tx_counter: Arc<AtomicI64>,
    err_counter: Arc<AtomicI64>,
    route_write_op: TransferRequest,
    dispatch_barrier_write_op: TransferRequest,
    combine_barrier_write_op: TransferRequest,
    node_route_count_ptrs: Vec<usize>,
    node_route_epoch_ptrs: Vec<usize>,
    pub(crate) accumulated_local_dispatch_bytes: AtomicU64,
    pub(crate) accumulated_nvlink_dispatch_bytes: AtomicU64,
    pub(crate) accumulated_network_dispatch_bytes: AtomicU64,
    pub(crate) accumulated_local_combine_bytes: AtomicU64,
    pub(crate) accumulated_nvlink_combine_bytes: AtomicU64,
    pub(crate) accumulated_network_combine_bytes: AtomicU64,
    pub(crate) peer_dispatch_bytes: Vec<AtomicU64>,
    pub(crate) peer_combine_bytes: Vec<AtomicU64>,
    pub(crate) accumulated_wait_dispatch_route_ns: AtomicU64,
    pub(crate) accumulated_route_exchange_ns: AtomicU64,
    pub(crate) accumulated_process_routing_ns: AtomicU64,
    pub(crate) accumulated_wait_dispatch_send_ns: AtomicU64,
    pub(crate) accumulated_dispatch_transfer_wait_ns: AtomicU64,
    pub(crate) accumulated_wait_dispatch_recv_ns: AtomicU64,
    pub(crate) accumulated_dispatch_barrier_ns: AtomicU64,
    pub(crate) accumulated_wait_combine_send_ns: AtomicU64,
    pub(crate) accumulated_combine_transfer_wait_ns: AtomicU64,
    pub(crate) accumulated_wait_combine_recv_ns: AtomicU64,
    pub(crate) accumulated_combine_barrier_ns: AtomicU64,
}

#[derive(Debug)]
struct RoutingInfo {
    num_recv_tx: u32,
    dispatch_ranges: Arc<Vec<ScatterTarget>>,
    combine_ranges: Arc<Vec<ScatterTarget>>,
}

impl WorkerState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        slot_idx: usize,
        slot_pool: Arc<SlotPool>,
        hidden_dim: usize,
        hidden_dim_scale: usize,
        in_elemsize: usize,
        out_elemsize: usize,
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
        num_routed_ptr: *mut u32,
        num_routed_mr: MemoryRegionHandle,
        send_buffer_ptr: *mut c_void,
        send_buffer_mr: MemoryRegionHandle,
        recv_buffer_ptr: *mut c_void,
        recv_buffer_mr: MemoryRegionHandle,
        node_route_count_ptrs: Vec<usize>,
        node_route_epoch_ptrs: Vec<usize>,
        device: u8,
        imm_base: u32,
        rank_handles: Vec<AllToAllRankHandle>,
        transfer_engine: Arc<TransferEngine>,
    ) -> Result<Self> {
        let dp_rank = rank % dp_size;
        let dp_group = rank / dp_size;
        let num_ep_groups = world_size / dp_size;
        let num_local_experts = num_experts.div_ceil(num_ep_groups);

        let gdr_context = GdrCopyContext::new()?;

        let slot = MicrobatchSlot::new(
            &gdr_context,
            num_local_experts,
            num_ep_groups,
            max_recv_tokens,
        )?;
        // Set up the immediate counters.
        let route_imm = imm_base;
        let route_counter = transfer_engine.get_imm_counter(route_imm);
        let dispatch_imm = imm_base + 1;
        let dispatch_counter = transfer_engine
            .get_gdr_counter(dispatch_imm, slot.dispatch_recv_flag.clone());
        let combine_imm = imm_base + 2;
        let combine_counter = transfer_engine.get_imm_counter(combine_imm);
        let dispatch_barrier_imm = imm_base + 3;
        let dispatch_barrier_counter =
            transfer_engine.get_imm_counter(dispatch_barrier_imm);
        let combine_barrier_imm = imm_base + 4;
        let combine_barrier_counter =
            transfer_engine.get_imm_counter(combine_barrier_imm);

        // Prepare the re-usable command to send out routing info.
        let route_write_op = {
            // Send the expert counts, plus one, over to all peers.
            let mut dsts = Vec::with_capacity(world_size - 1);
            let mut addrs = Vec::with_capacity(world_size - 1);
            for i in 1..world_size {
                // Do not transfer to self.
                let peer_rank = (rank + i) % world_size;
                let peer_group = peer_rank / dp_size;
                if peer_group == dp_group || peer_rank % dp_size != dp_rank {
                    continue;
                }

                let dst_mr = rank_handles[peer_rank].num_routed_desc.clone();
                let length: u64 = (num_experts * std::mem::size_of::<u32>()) as u64;
                let offset: u64 = (dp_group as u64) * length;
                addrs.push(extract_addrs(&dst_mr));
                dsts.push(ScatterTarget {
                    length,
                    src_offset: offset,
                    dst_offset: offset,
                    dst_mr,
                })
            }
            let dst_handle = transfer_engine
                .add_peer_group(addrs, Device::Cuda(CudaDeviceId(device)))
                .context("Failed to add_peer_group for route_write_op")?;

            TransferRequest::Scatter(ScatterTransferRequest {
                src_mr: num_routed_mr,
                dst_handle: Some(dst_handle),
                dsts: Arc::new(dsts),
                imm_data: Some(route_imm),
                domain: GroupTransferRouting::AllDomainsShardPeers,
            })
        };

        let (dispatch_barrier_write_op, combine_barrier_write_op) = {
            // Send an immedate to all peer ranks.
            let mut dst_mrs = Vec::with_capacity(world_size - 1);
            for i in 1..world_size {
                let peer_rank = (rank + i) % world_size;
                if peer_rank == rank {
                    continue;
                }
                dst_mrs.push(rank_handles[peer_rank].recv_buffer_desc.clone());
            }
            let dispatch = TransferRequest::Barrier(BarrierTransferRequest {
                imm_data: dispatch_barrier_imm,
                dst_mrs: dst_mrs.clone(),
                domain: DomainGroupRouting::Pinned { domain_idx: 0 },
            });
            let combine = TransferRequest::Barrier(BarrierTransferRequest {
                imm_data: combine_barrier_imm,
                dst_mrs,
                domain: DomainGroupRouting::Pinned { domain_idx: 0 },
            });
            (dispatch, combine)
        };

        Ok(WorkerState {
            slot_idx,
            slot_pool,
            transfer_engine: transfer_engine.clone(),
            max_num_tokens,
            max_recv_tokens,
            max_tokens_per_expert,
            max_private_tokens,
            hidden_dim,
            hidden_dim_scale,
            in_elemsize,
            out_elemsize,
            scale_elemsize,
            num_experts,
            num_experts_per_token,
            expert_padding,
            rank,
            dp_rank,
            dp_group,
            dp_size,
            node_size,
            world_size,
            rank_handles,
            stop_flag: AtomicBool::new(false),
            device,
            dispatch_imm,
            combine_imm,
            route_imm,
            dispatch_barrier_imm,
            combine_barrier_imm,
            buffers: WorkerBuffers { num_routed_ptr, send_buffer_ptr, recv_buffer_ptr },
            num_routed_mr,
            send_buffer_mr,
            recv_buffer_mr,
            slot,
            epoch: AtomicU32::new(1),
            route_counter,
            dispatch_counter,
            combine_counter,
            dispatch_barrier_counter,
            combine_barrier_counter,
            tx_counter: Arc::new(AtomicI64::new(0)),
            err_counter: Arc::new(AtomicI64::new(0)),
            route_write_op,
            dispatch_barrier_write_op,
            combine_barrier_write_op,
            node_route_count_ptrs,
            node_route_epoch_ptrs,
            accumulated_local_dispatch_bytes: AtomicU64::new(0),
            accumulated_nvlink_dispatch_bytes: AtomicU64::new(0),
            accumulated_network_dispatch_bytes: AtomicU64::new(0),
            accumulated_local_combine_bytes: AtomicU64::new(0),
            accumulated_nvlink_combine_bytes: AtomicU64::new(0),
            accumulated_network_combine_bytes: AtomicU64::new(0),
            peer_dispatch_bytes: (0..world_size).map(|_| AtomicU64::new(0)).collect(),
            peer_combine_bytes: (0..world_size).map(|_| AtomicU64::new(0)).collect(),
            accumulated_wait_dispatch_route_ns: AtomicU64::new(0),
            accumulated_route_exchange_ns: AtomicU64::new(0),
            accumulated_process_routing_ns: AtomicU64::new(0),
            accumulated_wait_dispatch_send_ns: AtomicU64::new(0),
            accumulated_dispatch_transfer_wait_ns: AtomicU64::new(0),
            accumulated_wait_dispatch_recv_ns: AtomicU64::new(0),
            accumulated_dispatch_barrier_ns: AtomicU64::new(0),
            accumulated_wait_combine_send_ns: AtomicU64::new(0),
            accumulated_combine_transfer_wait_ns: AtomicU64::new(0),
            accumulated_wait_combine_recv_ns: AtomicU64::new(0),
            accumulated_combine_barrier_ns: AtomicU64::new(0),
        })
    }

    pub(crate) fn is_running(&self) -> bool {
        !self.stop_flag.load(Ordering::Relaxed)
    }

    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        self.slot.stop(self.epoch.load(Ordering::Relaxed));
        self.slot.tx_ready.set(true);
    }

    pub(crate) fn get_num_routed(&self, dp_group: usize, expert: usize) -> u32 {
        assert!(dp_group < self.world_size / self.dp_size);
        assert!(expert < self.num_experts);
        unsafe {
            AtomicU32::from_ptr(
                self.buffers.num_routed_ptr.add(dp_group * self.num_experts + expert),
            )
        }
        .load(Ordering::Relaxed)
    }

    pub(crate) fn has_node_route_exchange(&self) -> bool {
        self.world_size == self.node_size
            && self.node_route_count_ptrs.len() == self.world_size
            && self.node_route_epoch_ptrs.len() == self.world_size
    }

    fn publish_local_node_route_counts(&self, epoch: u32) {
        debug_assert!(self.has_node_route_exchange());
        let local_counts = unsafe {
            self.buffers.num_routed_ptr.add(self.dp_group * self.num_experts)
        };
        let local_route_counts = self.node_route_count_ptrs[self.rank] as *mut u32;
        unsafe {
            std::ptr::copy_nonoverlapping(
                local_counts,
                local_route_counts,
                self.num_experts,
            );
            AtomicU32::from_ptr(self.node_route_epoch_ptrs[self.rank] as *mut u32)
                .store(epoch, Ordering::Release);
        }
    }

    fn wait_node_route_counts(&self, epoch: u32) -> bool {
        debug_assert!(self.has_node_route_exchange());
        let num_dp_groups = self.world_size / self.dp_size;
        for peer_group in 0..num_dp_groups {
            let peer_rank = peer_group * self.dp_size + self.dp_rank;
            let peer_epoch = unsafe {
                AtomicU32::from_ptr(self.node_route_epoch_ptrs[peer_rank] as *mut u32)
            };
            while peer_epoch.load(Ordering::Acquire) != epoch {
                if !self.is_running() {
                    return false;
                }
                std::hint::spin_loop();
            }

            let peer_route_counts = self.node_route_count_ptrs[peer_rank] as *mut u32;
            let local_row = unsafe {
                self.buffers.num_routed_ptr.add(peer_group * self.num_experts)
            };
            unsafe {
                std::ptr::copy_nonoverlapping(
                    peer_route_counts,
                    local_row,
                    self.num_experts,
                );
            }
        }
        true
    }

    fn get_dispatch_token_dim(&self) -> usize {
        let token_dim = (self.hidden_dim * self.in_elemsize).div_ceil(16) * 16;
        let scale_dim = (self.hidden_dim_scale * self.scale_elemsize).div_ceil(16) * 16;
        token_dim + scale_dim + 16
    }

    fn get_combine_token_dim(&self) -> usize {
        (self.hidden_dim * self.out_elemsize).div_ceil(16) * 16
    }

    pub fn main_loop(&self) {
        // Worker thread main loop.
        while self.is_running() {
            let step_dispatch_range = range_start!("p2p_all_to_all");
            self.step();
            range_end!(step_dispatch_range);
        }
    }

    pub fn failed(&self) -> bool {
        self.err_counter.load(Ordering::Relaxed) != 0
    }

    fn wait_imm(&self, counter: &ImmCounter, target: u32) -> bool {
        counter.wait_while(target, || self.is_running())
    }

    pub fn epoch(&self) -> u32 {
        self.epoch.load(Ordering::Acquire)
    }

    fn wait_epoch(&self, flag: &GdrEpoch, epoch: u32) -> bool {
        flag.wait_for(epoch, || self.is_running())
    }

    fn wait_epoch_trace(
        &self,
        name: &str,
        flag: &GdrEpoch,
        epoch: u32,
        trace: bool,
    ) -> bool {
        let mut last_report = Instant::now();
        while flag.get() != epoch {
            if !self.is_running() {
                return false;
            }
            if trace && last_report.elapsed() >= Duration::from_secs(1) {
                eprintln!(
                    "PPLX worker rank={} slot={} epoch={} waiting {} observed={}",
                    self.rank,
                    self.slot_idx,
                    epoch,
                    name,
                    flag.get()
                );
                last_report = Instant::now();
            }
            std::hint::spin_loop();
        }
        true
    }

    fn add_elapsed_ns(counter: &AtomicU64, start: Instant) {
        let ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        counter.fetch_add(ns, Ordering::Relaxed);
    }

    fn step(&self) {
        let epoch = self.epoch();
        let trace = std::env::var_os("PPLX_GARDEN_TRACE").is_some();

        // Wait for the device to copy the routing info to the host.
        let wait_dispatch_route_start = Instant::now();
        if !self.wait_epoch_trace(
            "dispatch_route_done",
            &self.slot.dispatch_route_done,
            epoch,
            trace,
        ) {
            return;
        }
        Self::add_elapsed_ns(
            &self.accumulated_wait_dispatch_route_ns,
            wait_dispatch_route_start,
        );
        if !self.is_running() {
            return;
        }

        let use_node_route_exchange = self.has_node_route_exchange();
        if use_node_route_exchange {
            self.publish_local_node_route_counts(epoch);
        } else {
            // Start exchanging routing info.
            self.transfer_engine
                .submit_transfer_atomic(
                    self.route_write_op.clone(),
                    self.tx_counter.clone(),
                    self.err_counter.clone(),
                )
                .unwrap();
        }

        // Wait for the dispatch kernel to copy tokens into send buffers.
        let wait_dispatch_send_start = Instant::now();
        if !self.wait_epoch_trace(
            "dispatch_send_done",
            &self.slot.dispatch_send_done,
            epoch,
            trace,
        ) {
            return;
        }
        Self::add_elapsed_ns(
            &self.accumulated_wait_dispatch_send_ns,
            wait_dispatch_send_start,
        );
        self.slot.tx_ready.set(false);
        if trace {
            eprintln!(
                "PPLX worker rank={} slot={} epoch={} dispatch_send_done observed; tx_ready cleared",
                self.rank, self.slot_idx, epoch
            );
        }
        if !self.is_running() {
            return;
        }

        // Trigger transfers into private recv buffers.
        let num_private_ranges = self.dispatch_initial_routes();
        if trace {
            eprintln!(
                "PPLX worker rank={} slot={} epoch={} dispatch_initial_routes num_private_ranges={}",
                self.rank, self.slot_idx, epoch, num_private_ranges
            );
        }

        // Wait for the routing information to arrive and aggregate it.
        let num_dp_groups = (self.world_size / self.dp_size) as u32;
        if trace {
            eprintln!(
                "PPLX worker rank={} slot={} epoch={} waiting route imm target={} world_size={} dp_size={} node_size={}",
                self.rank,
                self.slot_idx,
                epoch,
                num_dp_groups - 1,
                self.world_size,
                self.dp_size,
                self.node_size
            );
        }
        let route_exchange_start = Instant::now();
        if use_node_route_exchange {
            if !self.wait_node_route_counts(epoch) {
                return;
            }
        } else {
            if !self.wait_imm(&self.route_counter, num_dp_groups - 1) {
                return;
            }
        }
        Self::add_elapsed_ns(&self.accumulated_route_exchange_ns, route_exchange_start);
        let process_routing_start = Instant::now();
        let route = self.process_routing_info(epoch);
        Self::add_elapsed_ns(
            &self.accumulated_process_routing_ns,
            process_routing_start,
        );
        if trace {
            eprintln!(
                "PPLX worker rank={} slot={} epoch={} route processed num_recv_tx={} dispatch_ranges={} combine_ranges={}",
                self.rank,
                self.slot_idx,
                epoch,
                route.num_recv_tx,
                route.dispatch_ranges.len(),
                route.combine_ranges.len()
            );
        }

        // Register a callback to wait for the expected number of immediates.
        let num_shards = self.transfer_engine.nets_per_gpu().get() as u32;
        let num_dispatch_tx = 1
            + if num_private_ranges == 0 { 0 } else { 1 }
            + if route.dispatch_ranges.is_empty() { 0 } else { 1 };
        let num_combine_tx = if route.combine_ranges.is_empty() { 0 } else { 1 };
        let num_remote_nodes = self.world_size / self.node_size - 1;
        let groups_per_node = self.node_size / self.dp_size;
        let num_combine_imm = (num_remote_nodes * groups_per_node) as u32 * num_shards;

        // Dispatch stage.
        {
            let dispatch_range = range_start!("dispatch");

            if !route.dispatch_ranges.is_empty() {
                self.transfer_engine
                    .submit_transfer_atomic(
                        TransferRequest::Scatter(ScatterTransferRequest {
                            src_mr: self.send_buffer_mr,
                            dst_handle: None,
                            dsts: route.dispatch_ranges,
                            imm_data: Some(self.dispatch_imm),
                            domain: GroupTransferRouting::AllDomainsShardBytes,
                        }),
                        self.tx_counter.clone(),
                        self.err_counter.clone(),
                    )
                    .unwrap();
            }

            // Wait for the dispatch counter to settle and signal the kernel.
            if trace {
                eprintln!(
                    "PPLX worker rank={} slot={} epoch={} waiting dispatch counter target={}",
                    self.rank, self.slot_idx, epoch, route.num_recv_tx
                );
            }
            let dispatch_transfer_wait_start = Instant::now();
            self.dispatch_counter.wait(route.num_recv_tx);
            Self::add_elapsed_ns(
                &self.accumulated_dispatch_transfer_wait_ns,
                dispatch_transfer_wait_start,
            );
            if trace {
                eprintln!(
                    "PPLX worker rank={} slot={} epoch={} dispatch counter observed",
                    self.rank, self.slot_idx, epoch
                );
            }

            // Wait for the dispatch kernel to complete. It is triggered once
            // the immediate counter reaches zero by setting the dispatch recv flag.
            if trace {
                eprintln!(
                    "PPLX worker rank={} slot={} epoch={} waiting dispatch_recv_done",
                    self.rank, self.slot_idx, epoch
                );
            }
            let wait_dispatch_recv_start = Instant::now();
            if !self.wait_epoch(&self.slot.dispatch_recv_done, epoch) {
                return;
            }
            Self::add_elapsed_ns(
                &self.accumulated_wait_dispatch_recv_ns,
                wait_dispatch_recv_start,
            );
            if trace {
                eprintln!(
                    "PPLX worker rank={} slot={} epoch={} dispatch_recv_done observed",
                    self.rank, self.slot_idx, epoch
                );
            }

            range_end!(dispatch_range);
        }
        if self.world_size == self.node_size {
            // Single-node payload lifetime is guarded by the CUDA kernels'
            // NVLink sync-counter protocol. Avoid the fabric barrier entirely
            // and release the send buffer for the next dispatch here.
            self.slot.tx_ready.set(true);
        } else {
            let dispatch_barrier_start = Instant::now();
            if !self.barrier(
                "dispatch",
                self.dispatch_barrier_write_op.clone(),
                &self.dispatch_barrier_counter,
                num_dispatch_tx,
            ) {
                return;
            }
            Self::add_elapsed_ns(
                &self.accumulated_dispatch_barrier_ns,
                dispatch_barrier_start,
            );
        }
        if trace {
            eprintln!(
                "PPLX worker rank={} slot={} epoch={} dispatch barrier complete num_dispatch_tx={}",
                self.rank, self.slot_idx, epoch, num_dispatch_tx
            );
        }

        // Combine stage.
        {
            if trace {
                eprintln!(
                    "PPLX worker rank={} slot={} epoch={} waiting combine_send_done num_combine_imm={} combine_ranges={}",
                    self.rank,
                    self.slot_idx,
                    epoch,
                    num_combine_imm,
                    route.combine_ranges.len()
                );
            }
            let wait_combine_send_start = Instant::now();
            if !self.wait_epoch(&self.slot.combine_send_done, epoch) {
                return;
            }
            Self::add_elapsed_ns(
                &self.accumulated_wait_combine_send_ns,
                wait_combine_send_start,
            );
            if !self.is_running() {
                return;
            }
            if trace {
                eprintln!(
                    "PPLX worker rank={} slot={} epoch={} combine_send_done observed",
                    self.rank, self.slot_idx, epoch
                );
            }

            // Sent the tokens.
            let combine_range = range_start!("combine");

            if !route.combine_ranges.is_empty() {
                self.transfer_engine
                    .submit_transfer_atomic(
                        TransferRequest::Scatter(ScatterTransferRequest {
                            src_mr: self.send_buffer_mr,
                            dst_handle: None,
                            dsts: route.combine_ranges,
                            imm_data: Some(self.combine_imm),
                            domain: GroupTransferRouting::AllDomainsShardBytes,
                        }),
                        self.tx_counter.clone(),
                        self.err_counter.clone(),
                    )
                    .unwrap();
            }

            // Wait for all remote writes to complete.
            if trace {
                eprintln!(
                    "PPLX worker rank={} slot={} epoch={} waiting combine imm target={}",
                    self.rank, self.slot_idx, epoch, num_combine_imm
                );
            }
            let combine_transfer_wait_start = Instant::now();
            if !self.wait_imm(&self.combine_counter, num_combine_imm) {
                return;
            }
            Self::add_elapsed_ns(
                &self.accumulated_combine_transfer_wait_ns,
                combine_transfer_wait_start,
            );
            self.slot.combine_recv_flag.set(true);
            if trace {
                eprintln!(
                    "PPLX worker rank={} slot={} epoch={} combine imm observed; recv flag set",
                    self.rank, self.slot_idx, epoch
                );
            }

            // Let the recv phase output the tokens and proceed forward.
            if trace {
                eprintln!(
                    "PPLX worker rank={} slot={} epoch={} waiting combine_recv_done",
                    self.rank, self.slot_idx, epoch
                );
            }
            let wait_combine_recv_start = Instant::now();
            if !self.wait_epoch(&self.slot.combine_recv_done, epoch) {
                return;
            }
            Self::add_elapsed_ns(
                &self.accumulated_wait_combine_recv_ns,
                wait_combine_recv_start,
            );
            if trace {
                eprintln!(
                    "PPLX worker rank={} slot={} epoch={} combine_recv_done observed",
                    self.rank, self.slot_idx, epoch
                );
            }

            range_end!(combine_range);
        }

        if self.world_size == self.node_size {
            self.slot.tx_ready.set(true);
        } else {
            let combine_barrier_start = Instant::now();
            if !self.barrier(
                "combine",
                self.combine_barrier_write_op.clone(),
                &self.combine_barrier_counter,
                num_combine_tx,
            ) {
                return;
            }
            Self::add_elapsed_ns(
                &self.accumulated_combine_barrier_ns,
                combine_barrier_start,
            );
        }
        if trace {
            eprintln!(
                "PPLX worker rank={} slot={} epoch={} combine barrier complete; releasing slot",
                self.rank, self.slot_idx, epoch
            );
        }
        self.epoch.fetch_add(1, Ordering::Release);
        self.slot_pool.release(self.slot_idx);
    }

    fn dispatch_initial_routes(&self) -> usize {
        let token_dim = self.get_dispatch_token_dim() as u64;
        let dsts: Vec<ScatterTarget> = compute_initial_dispatch_transfer_plans(
            self.dp_group,
            self.dp_rank,
            self.dp_size,
            self.node_size,
            self.world_size,
            self.num_experts,
            self.max_private_tokens,
            |dp_group, expert| self.get_num_routed(dp_group, expert),
        )
        .into_iter()
        .map(|transfer| ScatterTarget {
            length: transfer.length_tokens as u64 * token_dim,
            src_offset: transfer.src_token_offset * token_dim,
            dst_offset: transfer.dst_token_offset * token_dim,
            dst_mr: self.rank_handles[transfer.peer_rank].recv_buffer_desc.clone(),
        })
        .collect();

        if dsts.is_empty() {
            return 0;
        }

        let num_ranges = dsts.len();
        self.transfer_engine
            .submit_transfer_atomic(
                TransferRequest::Scatter(ScatterTransferRequest {
                    src_mr: self.send_buffer_mr,
                    dst_handle: None,
                    dsts: Arc::new(dsts),
                    imm_data: Some(self.dispatch_imm),
                    domain: GroupTransferRouting::AllDomainsShardPeers,
                }),
                self.tx_counter.clone(),
                self.err_counter.clone(),
            )
            .unwrap();

        num_ranges
    }

    #[allow(clippy::needless_range_loop)]
    fn process_routing_info(&self, epoch: u32) -> RoutingInfo {
        // Determine counts on the current rank.
        let process_routing_info_range = range_start!("process_routing_info");

        let num_dp_groups = self.world_size / self.dp_size;
        let rank_node = self.rank / self.node_size;

        let nets_per_gpu = self.transfer_engine.nets_per_gpu().get() as u32;
        let plan = compute_receive_route_plan(
            self.dp_group,
            self.dp_rank,
            self.dp_size,
            self.node_size,
            self.world_size,
            self.num_experts,
            self.expert_padding,
            self.max_tokens_per_expert,
            self.max_private_tokens,
            nets_per_gpu,
            |dp_group, expert| self.get_num_routed(dp_group, expert),
        );
        self.slot.tokens_per_expert.copy(&plan.tokens_per_expert);

        // Copy the buffers to the device.
        self.slot.padded_index.copy(&plan.padded_index);
        self.slot.source_rank.copy(&plan.source_rank);
        self.slot.source_dispatch_offset.copy(&plan.source_dispatch_offset);
        self.slot.combine_send_offset.copy(&plan.combine_send_offset);
        self.slot.layout_range.copy(&plan.layout_range);
        self.slot.num_recv_tokens.copy(&[
            plan.num_recv_tokens as u32,
            plan.num_recv_efa_tokens as u32,
            0,
        ]);
        self.slot.num_recv_tokens_ready.set(epoch);

        // Prepare the dispatch commands, beyond the private recv buffers.
        let token_dim = self.get_dispatch_token_dim() as u64;
        let dispatch_ranges = compute_remote_dispatch_transfer_plans(
            &plan,
            self.dp_group,
            self.dp_rank,
            self.dp_size,
            self.node_size,
            self.world_size,
            self.num_experts,
            self.max_private_tokens,
            |dp_group, expert| self.get_num_routed(dp_group, expert),
        )
        .into_iter()
        .map(|transfer| ScatterTarget {
            length: transfer.length_tokens as u64 * token_dim,
            src_offset: transfer.src_token_offset * token_dim,
            dst_offset: transfer.dst_token_offset * token_dim,
            dst_mr: self.rank_handles[transfer.peer_rank].recv_buffer_desc.clone(),
        })
        .collect();

        // Prepare the combine commands to remote nodes over EFA.
        let token_dim = self.get_combine_token_dim() as u64;
        let combine_ranges = compute_remote_combine_transfer_plans(
            &plan,
            self.dp_group,
            self.dp_rank,
            self.dp_size,
            self.node_size,
            self.world_size,
        )
        .into_iter()
        .map(|transfer| ScatterTarget {
            length: transfer.length_tokens as u64 * token_dim,
            src_offset: transfer.src_token_offset * token_dim,
            dst_offset: transfer.dst_token_offset * token_dim,
            dst_mr: self.rank_handles[transfer.peer_rank].recv_buffer_desc.clone(),
        })
        .collect();

        // Telemetry counters.
        for peer_rank in 0..self.world_size {
            let dispatch_tokens = plan.tokens_to_rank[peer_rank] as u64;
            let dispatch_bytes = dispatch_tokens * self.get_dispatch_token_dim() as u64;
            if dispatch_bytes > 0 {
                self.peer_dispatch_bytes[peer_rank]
                    .fetch_add(dispatch_bytes, Ordering::Relaxed);
                if peer_rank == self.rank {
                    self.accumulated_local_dispatch_bytes
                        .fetch_add(dispatch_bytes, Ordering::Relaxed);
                } else if peer_rank / self.node_size == rank_node {
                    self.accumulated_nvlink_dispatch_bytes
                        .fetch_add(dispatch_bytes, Ordering::Relaxed);
                } else {
                    self.accumulated_network_dispatch_bytes
                        .fetch_add(dispatch_bytes, Ordering::Relaxed);
                }
            }
        }

        for peer_group in 0..num_dp_groups {
            let peer_rank = peer_group * self.dp_size + self.dp_rank;
            let combine_tokens = plan.tokens_from_group[peer_group] as u64;
            let combine_bytes = combine_tokens * self.get_combine_token_dim() as u64;
            if combine_bytes > 0 {
                self.peer_combine_bytes[peer_rank]
                    .fetch_add(combine_bytes, Ordering::Relaxed);
                if peer_rank == self.rank {
                    self.accumulated_local_combine_bytes
                        .fetch_add(combine_bytes, Ordering::Relaxed);
                } else if peer_rank / self.node_size == rank_node {
                    self.accumulated_nvlink_combine_bytes
                        .fetch_add(combine_bytes, Ordering::Relaxed);
                } else {
                    self.accumulated_network_combine_bytes
                        .fetch_add(combine_bytes, Ordering::Relaxed);
                }
            }
        }

        range_end!(process_routing_info_range);

        RoutingInfo {
            num_recv_tx: plan.num_recv_tx,
            dispatch_ranges: Arc::new(dispatch_ranges),
            combine_ranges: Arc::new(combine_ranges),
        }
    }

    fn barrier(
        &self,
        label: &str,
        request: TransferRequest,
        imm_counter: &ImmCounter,
        num_tx: u32,
    ) -> bool {
        let barrier = range_start!("barrier");
        let trace = std::env::var_os("PPLX_GARDEN_TRACE").is_some();
        if trace {
            eprintln!(
                "PPLX worker rank={} slot={} barrier={} submitting num_tx={}",
                self.rank, self.slot_idx, label, num_tx
            );
        }

        self.transfer_engine
            .submit_transfer_atomic(
                request,
                self.tx_counter.clone(),
                self.err_counter.clone(),
            )
            .unwrap();

        // Wait for all payloads to be received.
        if trace {
            eprintln!(
                "PPLX worker rank={} slot={} barrier={} waiting imm target={}",
                self.rank,
                self.slot_idx,
                label,
                self.world_size - 1
            );
        }
        if !self.wait_imm(imm_counter, (self.world_size - 1) as u32) {
            range_end!(barrier);
            return false;
        }
        if trace {
            eprintln!(
                "PPLX worker rank={} slot={} barrier={} imm observed",
                self.rank, self.slot_idx, label
            );
        }

        // Wait for the sends to complete.
        let num_tx_total = num_tx as i64 + 1;
        let old = self.tx_counter.fetch_sub(num_tx_total, Ordering::Relaxed);
        if trace {
            eprintln!(
                "PPLX worker rank={} slot={} barrier={} waiting tx_counter old={} subtract={}",
                self.rank, self.slot_idx, label, old, num_tx_total
            );
        }
        if old < num_tx_total {
            while self.tx_counter.load(Ordering::Relaxed) < 0 {
                if !self.is_running() {
                    range_end!(barrier);
                    return false;
                }
                std::thread::yield_now();
            }
        }
        self.slot.tx_ready.set(true);
        if trace {
            eprintln!(
                "PPLX worker rank={} slot={} barrier={} tx_ready set",
                self.rank, self.slot_idx, label
            );
        }

        range_end!(barrier);
        true
    }
}

fn extract_addrs(
    dst_mr: &MemoryRegionDescriptor,
) -> fabric_lib::api::SmallVec<DomainAddress> {
    dst_mr.addr_rkey_list.iter().map(|(addr, _)| addr.clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::{
        RemoteTransferPlan, compute_initial_dispatch_transfer_plans,
        compute_padded_offsets, compute_receive_route_plan,
        compute_remote_combine_transfer_plans, compute_remote_dispatch_transfer_plans,
    };

    #[test]
    fn compact_offsets_respect_expert_padding() {
        assert_eq!(compute_padded_offsets(&[3, 0, 5], 4, 0), vec![0, 4, 4]);
        assert_eq!(compute_padded_offsets(&[1, 4, 5], 4, 0), vec![0, 4, 8]);
    }

    #[test]
    fn batched_offsets_use_fixed_expert_stride() {
        assert_eq!(compute_padded_offsets(&[3, 0, 5], 4, 8), vec![0, 8, 16]);
        assert_eq!(compute_padded_offsets(&[1, 4, 5], 1, 128), vec![0, 128, 256]);
    }

    #[test]
    fn slot_pool_tracks_explicit_and_automatic_slots() {
        let pool = super::SlotPool::new(2);

        assert!(pool.acquire_specific(1));
        assert!(!pool.acquire_specific(2));
        assert_eq!(pool.acquire(), 0);

        pool.release(1);
        assert!(pool.acquire_specific(1));
        pool.release(1);
        pool.release(1);

        assert_eq!(pool.acquire(), 1);
    }

    #[test]
    fn receive_route_plan_orders_remote_local_then_self() {
        let num_routed = [
            [1, 2, 5, 1, 4, 0, 2, 3],
            [3, 0, 0, 0, 0, 0, 0, 0],
            [2, 1, 0, 0, 0, 0, 0, 0],
            [0, 4, 0, 0, 0, 0, 0, 0],
        ];

        let plan = compute_receive_route_plan(
            0,
            0,
            1,
            2,
            4,
            8,
            1,
            8,
            2,
            2,
            |dp_group, expert| num_routed[dp_group][expert],
        );

        assert_eq!(plan.tokens_per_expert, vec![6, 7]);
        assert_eq!(plan.tokens_from_group, vec![3, 3, 3, 4]);
        assert_eq!(plan.tokens_to_rank, vec![3, 6, 4, 5]);
        assert_eq!(plan.num_recv_tokens, 13);
        assert_eq!(plan.num_recv_efa_tokens, 7);
        assert_eq!(plan.num_recv_tx, 6);

        assert_eq!(plan.source_rank, vec![2, 2, 2, 3, 3, 3, 3, 1, 1, 1, 0, 0, 0]);
        assert_eq!(
            plan.source_dispatch_offset,
            vec![4, 5, 14, 6, 7, 17, 18, 2, 3, (1 << 31) | 2, 0, 1, 2]
        );
        assert_eq!(
            plan.combine_send_offset,
            vec![6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 0, 1, 2]
        );
        assert_eq!(plan.padded_index, vec![0, 1, 8, 9, 10, 11, 12, 2, 3, 4, 5, 13, 14]);
        assert_eq!(
            plan.layout_range,
            vec![
                super::pack_layout_range(1, 5),
                super::pack_layout_range(3, 2),
                super::pack_layout_range(2, 0),
                super::pack_layout_range(0, 2),
                super::pack_layout_range(2, 5),
                super::pack_layout_range(0, 5),
                super::pack_layout_range(1, 0),
                super::pack_layout_range(4, 1),
            ]
        );

        assert_eq!(
            compute_initial_dispatch_transfer_plans(
                0,
                0,
                1,
                2,
                4,
                8,
                2,
                |dp_group, expert| { num_routed[dp_group][expert] }
            ),
            vec![
                RemoteTransferPlan {
                    peer_rank: 2,
                    length_tokens: 2,
                    src_token_offset: 9,
                    dst_token_offset: 0,
                },
                RemoteTransferPlan {
                    peer_rank: 3,
                    length_tokens: 2,
                    src_token_offset: 13,
                    dst_token_offset: 0,
                },
            ]
        );
        assert_eq!(
            compute_remote_dispatch_transfer_plans(
                &plan,
                0,
                0,
                1,
                2,
                4,
                8,
                2,
                |dp_group, expert| { num_routed[dp_group][expert] }
            ),
            vec![
                RemoteTransferPlan {
                    peer_rank: 2,
                    length_tokens: 2,
                    src_token_offset: 11,
                    dst_token_offset: 8,
                },
                RemoteTransferPlan {
                    peer_rank: 3,
                    length_tokens: 3,
                    src_token_offset: 15,
                    dst_token_offset: 8,
                },
            ]
        );
        assert_eq!(
            compute_remote_combine_transfer_plans(&plan, 0, 0, 1, 2, 4),
            vec![
                RemoteTransferPlan {
                    peer_rank: 2,
                    length_tokens: 3,
                    src_token_offset: 6,
                    dst_token_offset: 0,
                },
                RemoteTransferPlan {
                    peer_rank: 3,
                    length_tokens: 4,
                    src_token_offset: 9,
                    dst_token_offset: 0,
                },
            ]
        );
    }
}

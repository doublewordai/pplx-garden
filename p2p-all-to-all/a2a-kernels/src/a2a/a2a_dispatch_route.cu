#include "a2a/a2a_kernels.h"
#include "core/device_utils.cuh"
#include "core/launch_utils.cuh"
#include "core/memory.cuh"

#include <cuda.h>
#include <cooperative_groups.h>
#include <nvtx3/nvToolsExt.h>

#include <cassert>
#include <cstdint>

using namespace rose;
using namespace rose::device;

template <size_t NUM_WARPS, typename NumExpertsPerTokenTy>
__global__ __launch_bounds__(NUM_WARPS * WARP_SIZE, 1) void a2a_dispatch_route_kernel(
    size_t num_experts,
    size_t num_experts_per_token,
    size_t rank,
    size_t dp_size,
    size_t world_size,
    size_t num_tokens,
    const int32_t * __restrict__ bound_m_ptr,
    const int32_t * __restrict__ indices,
    size_t indices_stride,
    uint32_t * __restrict__ token_offset,
    uint32_t * __restrict__ num_routed,
    uint32_t * __restrict__ expert_offsets,
    uint32_t * __restrict__ combine_recv_position,
    uint32_t * __restrict__ dispatch_route_done,
    uint32_t * __restrict__ epoch_counter,
    uint32_t * __restrict__ current_epoch
) {
    NumExpertsPerTokenTy num_experts_per_token_bound(num_experts_per_token);
    auto grid = cooperative_groups::this_grid();

    extern __shared__ uint32_t shared_memory[];
    const size_t warp_id = threadIdx.x / WARP_SIZE;
    const size_t lane_id = get_lane_id();

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *current_epoch = atomicAdd(epoch_counter, 1) + 1;
    }
    grid.sync();
    const uint32_t epoch = *current_epoch;

    const size_t dp_group = rank / dp_size;
    const size_t expert_parallel_size = world_size / dp_size;
    const size_t experts_per_rank = ceil_div<size_t>(num_experts, expert_parallel_size);
    const size_t num_send_tokens = bound_m_ptr ? *bound_m_ptr : num_tokens;

    uint32_t *tokens_per_expert = shared_memory;
    for (uint32_t i = threadIdx.x; i < num_experts; i += blockDim.x) {
        tokens_per_expert[i] = 0;
    }
    __syncthreads();

    for (uint32_t expert = threadIdx.x; expert < num_experts; expert += blockDim.x) {
        uint32_t count = 0;
        for (uint32_t token = 0; token < num_send_tokens; ++token) {
            #pragma unroll
            for (uint32_t index = 0; index < num_experts_per_token_bound; ++index) {
                const uint32_t route = __ldg(&indices[token * indices_stride + index]);
                if (route == expert) {
                    token_offset[token * num_experts_per_token_bound + index] = count++;
                }
            }
        }
        tokens_per_expert[expert] = count;
    }
    __syncthreads();

    const uint32_t i = threadIdx.x;
    const uint32_t num_warps = ceil_div<size_t>(num_experts, WARP_SIZE);
    uint32_t *expert_sums = shared_memory;

    uint32_t expert_offset = 0;
    uint32_t *local_num_routed = num_routed + dp_group * num_experts;
    if (i < num_experts) {
        expert_offset = tokens_per_expert[i];
        local_num_routed[i] = expert_offset;
    }
    __syncthreads();

    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        unsigned warp_sum_expert = __shfl_up_sync(0xFFFFFFFF, expert_offset, offset);
        if (lane_id >= offset) {
            expert_offset += warp_sum_expert;
        }
    }
    if (lane_id == WARP_SIZE - 1) {
        expert_sums[warp_id] = expert_offset;
    }
    __syncthreads();

    if (warp_id == 0) {
        uint32_t total_expert_sum = (lane_id < num_warps) ? expert_sums[lane_id] : 0;
        for (int offset = 1; offset < num_warps; offset <<= 1) {
            unsigned warp_sum = __shfl_up_sync(0xFFFFFFFF, total_expert_sum, offset);
            if (lane_id >= offset) {
                total_expert_sum += warp_sum;
            }
        }
        if (lane_id < num_warps) {
            expert_sums[lane_id] = total_expert_sum;
        }
    }
    __syncthreads();

    if (i < num_experts) {
        if (warp_id > 0) {
            expert_offsets[i] = expert_sums[warp_id - 1] + expert_offset;
        } else {
            expert_offsets[i] = expert_offset;
        }
    }
    __syncthreads();

    for (uint32_t token = threadIdx.x; token < num_send_tokens; token += blockDim.x) {
        #pragma unroll
        for (uint32_t index = 0; index < num_experts_per_token_bound; ++index) {
            const uint32_t expert = __ldg(&indices[token * indices_stride + index]);
            const uint32_t offset = token_offset[token * num_experts_per_token_bound + index];
            const uint32_t position = (expert > 0 ? expert_offsets[expert - 1] : 0) + offset;
            combine_recv_position[token * num_experts_per_token_bound + index] = position;
        }
    }
    __syncthreads();

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        st_mmio_u32(dispatch_route_done, epoch);
    }
}

int a2a_kernels::a2a_dispatch_route(
    size_t num_experts,
    size_t num_experts_per_token,
    size_t rank,
    size_t dp_size,
    size_t world_size,
    size_t num_tokens,
    const int32_t *bound_m_ptr,
    const int32_t *indices,
    size_t indices_stride,
    uint32_t *token_offset,
    uint32_t *num_routed,
    uint32_t *expert_offsets,
    uint32_t *combine_recv_position,
    uint32_t *dispatch_route_done,
    uint32_t *epoch_counter,
    uint32_t *current_epoch,
    uint64_t stream
) {
    constexpr size_t NUM_WARPS = 16;
    constexpr size_t NUM_THREADS = NUM_WARPS * WARP_SIZE;

    dim3 dimGrid(1, 1, 1);
    dim3 dimBlock(NUM_THREADS, 1, 1);

    assert(num_experts <= NUM_THREADS);

    void *args[] = {
        &num_experts,
        &num_experts_per_token,
        &rank,
        &dp_size,
        &world_size,
        &num_tokens,
        &bound_m_ptr,
        &indices,
        &indices_stride,
        &token_offset,
        &num_routed,
        &expert_offsets,
        &combine_recv_position,
        &dispatch_route_done,
        &epoch_counter,
        &current_epoch,
    };

    const size_t shared_memory = std::max(num_experts, NUM_WARPS) * sizeof(uint32_t);

    nvtxRangePush("dispatch_route");
    cudaError_t status;
    LAUNCH_NUM_EXPERTS_PER_TOKEN(num_experts_per_token, NumExpertsPerToken, {
        status = cudaLaunchCooperativeKernel(
            (void *)&a2a_dispatch_route_kernel<NUM_WARPS, NumExpertsPerToken>,
            dimGrid,
            dimBlock,
            args,
            shared_memory,
            (cudaStream_t)stream
        );
    });
    nvtxRangePop();
    return status;
}

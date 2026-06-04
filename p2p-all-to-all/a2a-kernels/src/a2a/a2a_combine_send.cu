#include "a2a/a2a_kernels.h"
#include "core/memory.cuh"
#include "core/device_utils.cuh"
#include "core/combine_utils.cuh"
#include "core/launch_utils.cuh"

#include <cuda.h>
#include <cooperative_groups.h>
#include <nvtx3/nvToolsExt.h>

#include <cassert>
#include <cstdint>

#include <type_traits>

using namespace rose;
using namespace rose::device;


__global__ __launch_bounds__(WARP_SIZE, 1) void a2a_wait_tx_ready_kernel(
    uint8_t * __restrict__ tx_ready,
    const uint32_t * __restrict__ num_recv_tokens_ptr
) {
    if (threadIdx.x == 0) {
        const unsigned num_efa_tokens = __ldg(num_recv_tokens_ptr + 1);
        if (num_efa_tokens != 0) {
            while (ld_mmio_b8(tx_ready) == 0);
        }
    }
}


template <unsigned NUM_WARPS, unsigned NODE_SIZE, unsigned DP_SIZE, typename TokenDim>
__global__ __launch_bounds__(NUM_WARPS * WARP_SIZE, 1) void a2a_combine_send_kernel(
    const size_t token_dim,
    const size_t rank,
    const std::byte * __restrict__ expert_x_ptr,
    size_t expert_x_stride,
    uint8_t * __restrict__ tx_ready,
    std::byte * __restrict__ send_buffer,
    std::byte * __restrict__ recv_buffer,
    uint32_t * __restrict__ source_rank_by_final_index,
    uint32_t * __restrict__ combine_send_offset,
    uint32_t * __restrict__ padded_index,
    const uint32_t * __restrict__ num_recv_tokens_ptr,
    uint32_t * __restrict__ combine_send_done,
    uint32_t * __restrict__ token_counter,
    uint32_t * __restrict__ sync_counter,
    uint32_t ** __restrict__ sync_ptrs,
    std::byte **recv_ptrs,
    uint32_t * __restrict__ current_epoch
) {
    TokenDim token_bound(token_dim);
    constexpr size_t NUM_THREADS = NUM_WARPS * WARP_SIZE;

    constexpr size_t NUM_STAGES = 8;
    struct Stage {
        uint32_t offset;
        uint32_t index;
        uint32_t rank;
    };
    __shared__ Stage shared_stages[NUM_STAGES];
    Stage local_stages[NUM_STAGES];

    // Local copy of peer recv ptrs. The pointer table is only populated when
    // an intra-node peer group exists; with NODE_SIZE == 1, local writes target
    // this rank's own recv buffer directly.
    std::byte *recv_ptrs_local[NODE_SIZE];
    if constexpr (NODE_SIZE > 1) {
        #pragma unroll
        for (unsigned i = 0; i < NODE_SIZE; i++) {
            recv_ptrs_local[i] = recv_ptrs[i];
        }
    }

    auto grid = cooperative_groups::this_grid();
    const unsigned rank_node = rank / NODE_SIZE;
    const unsigned warp_id = threadIdx.x / WARP_SIZE;
    const unsigned lane_id = get_lane_id();
    const uint32_t epoch = *current_epoch;

    const unsigned num_recv_tokens = __ldg(num_recv_tokens_ptr);
    const unsigned num_efa_tokens = __ldg(num_recv_tokens_ptr + 1);

    // Pick a token to send.
    unsigned token = blockIdx.x;

    // Synchronization counter.
    auto counter = *sync_counter;

    // Wait for all transactions using the send buffer to finish before writing
    // to it. Local-node combine copies do not touch the send buffer, so do not
    // block on tx_ready unless there is fabric-bound combine payload.
    if (warp_id == 0) {
        if (elect_one_sync()) {
            if (num_efa_tokens != 0) {
                while (ld_mmio_b8(tx_ready) == 0);
            }
        }
    } else if (warp_id == 1) {
        if constexpr (NODE_SIZE > 1) {
            auto local_rank = rank % NODE_SIZE;
            if (lane_id < NODE_SIZE) {
                auto *flag = &sync_ptrs[local_rank][lane_id];
                while (ld_volatile_u32(flag) != counter);
            }
        }
    } else if (warp_id == 2) {
        unsigned next_token = token + lane_id * gridDim.x;
        if (next_token < num_recv_tokens && lane_id < NUM_STAGES) {
            auto index = padded_index[next_token];
            shared_stages[lane_id].offset = combine_send_offset[next_token];
            shared_stages[lane_id].index = index;
            shared_stages[lane_id].rank = source_rank_by_final_index[index];
        }
    }
    __syncthreads();

    auto shared_to_local = [&](unsigned count) {
        #pragma unroll(NUM_STAGES)
        for (unsigned s = 0; s < NUM_STAGES; s++) {
            local_stages[s] = shared_stages[s];
        }
        __syncthreads();
    };

    shared_to_local(num_efa_tokens);

    unsigned num_local_efa_tokens = 0;
    while (token < num_efa_tokens) {
        // Fetch the next batch.
        unsigned next_token = token + (NUM_STAGES + threadIdx.x) * gridDim.x;
        if (threadIdx.x < NUM_STAGES && next_token < num_efa_tokens) {
            auto index = padded_index[next_token];
            shared_stages[threadIdx.x].offset = combine_send_offset[next_token];
            shared_stages[threadIdx.x].index = index;
            shared_stages[threadIdx.x].rank = source_rank_by_final_index[index];
        }
        __syncthreads();

        // Pipelined copy.
        uint4 values[NUM_STAGES];
        for (unsigned i = threadIdx.x; i * sizeof(uint4) < token_bound; i += NUM_THREADS) {
            #pragma unroll(NUM_STAGES)
            for (unsigned s = 0; s < NUM_STAGES && token + s * gridDim.x < num_efa_tokens; s++) {
                auto *ptr = (uint4*)(expert_x_ptr + expert_x_stride * local_stages[s].index);
                values[s] = ld_global_nc_uint4(&ptr[i]);
            }

            #pragma unroll(NUM_STAGES)
            for (unsigned s = 0; s < NUM_STAGES && token + s * gridDim.x < num_efa_tokens; s++) {
                unsigned offset = local_stages[s].offset;
                auto token_rank = local_stages[s].rank;
                auto token_node = token_rank / NODE_SIZE;
                if (token_node != rank_node) {
                    auto *x_token_dst = (uint4*)(send_buffer + offset * token_bound);
                    st_global_nc_uint4(&x_token_dst[i], values[s]);
                }
            }
        }

        #pragma unroll(NUM_STAGES)
        for (unsigned s = 0; s < NUM_STAGES && token < num_efa_tokens; s++) {
            if (token < num_efa_tokens) {
                num_local_efa_tokens++;
            }
            token += gridDim.x;
        }

        shared_to_local(num_efa_tokens);
    }

    grid.sync();

    if (num_efa_tokens != 0 && blockIdx.x == 0 && threadIdx.x == 0) {
        st_mmio_u32(combine_send_done, epoch);
    }

    token = blockIdx.x;

    if (warp_id == 0) {
        unsigned next_token = token + lane_id * gridDim.x;
        if (next_token < num_recv_tokens && lane_id < NUM_STAGES) {
            auto index = padded_index[next_token];
            shared_stages[lane_id].offset = combine_send_offset[next_token];
            shared_stages[lane_id].index = index;
            shared_stages[lane_id].rank = source_rank_by_final_index[index];
        }
    }
    __syncthreads();

    shared_to_local(num_recv_tokens);

    grid.sync();

    while (token < num_recv_tokens) {
        // Fetch the next batch.
        unsigned next_token = token + (NUM_STAGES + threadIdx.x) * gridDim.x;
        if (threadIdx.x < NUM_STAGES && next_token < num_recv_tokens) {
            auto index = padded_index[next_token];
            shared_stages[threadIdx.x].offset = combine_send_offset[next_token];
            shared_stages[threadIdx.x].index = index;
            shared_stages[threadIdx.x].rank = source_rank_by_final_index[index];
        }
        __syncthreads();

        // Pipelined copy.
        uint4 values[NUM_STAGES];
        for (unsigned i = threadIdx.x; i * sizeof(uint4) < token_bound; i += NUM_THREADS) {
            #pragma unroll(NUM_STAGES)
            for (unsigned s = 0; s < NUM_STAGES && token + s * gridDim.x < num_recv_tokens; s++) {
                auto *ptr = (uint4*)(expert_x_ptr + expert_x_stride * local_stages[s].index);
                values[s] = ld_global_nc_uint4(&ptr[i]);
            }

            #pragma unroll(NUM_STAGES)
            for (unsigned s = 0; s < NUM_STAGES && token + s * gridDim.x < num_recv_tokens; s++) {
                unsigned offset = local_stages[s].offset;
                auto token_rank = local_stages[s].rank;
                auto token_node = token_rank / NODE_SIZE;
                if (token_node == rank_node) {
                    auto token_peer = token_rank % NODE_SIZE;
                    std::byte *dst_base;
                    if constexpr (NODE_SIZE > 1) {
                        dst_base = recv_ptrs_local[token_peer];
                    } else {
                        dst_base = recv_buffer;
                    }
                    auto *x_token_dst = (uint4*)(dst_base + offset * token_bound);
                    st_global_nc_uint4(&x_token_dst[i], values[s]);
                }
            }
        }

        #pragma unroll(NUM_STAGES)
        for (unsigned s = 0; s < NUM_STAGES && token < num_recv_tokens; s++) {
            token += gridDim.x;
        }

        shared_to_local(num_recv_tokens);
    }

    grid.sync();

    if (blockIdx.x == 0) {
        if (warp_id == 0) {
            if (elect_one_sync()) {
                *sync_counter = counter + 1;
                *token_counter = 0;
                st_mmio_u32(combine_send_done, epoch);
                if (num_efa_tokens != 0) {
                    *tx_ready = 0;
                }
            }
        } else if (warp_id == 1) {
            if constexpr (NODE_SIZE > 1) {
                auto local_rank = rank % NODE_SIZE;
                if (lane_id < NODE_SIZE) {
                    st_release_u32(&sync_ptrs[lane_id][local_rank + NODE_SIZE], counter + 1);
                }
            }
        }
    }
}


int a2a_kernels::a2a_combine_send(
    size_t num_blocks,
    size_t hidden_dim,
    size_t x_elemsize,
    size_t rank,
    size_t node_size,
    size_t dp_size,
    const uint8_t *expert_x_ptr,
    size_t expert_x_stride,
    uint8_t *tx_ready,
    uint8_t *send_buffer,
    uint8_t *recv_buffer,
    uint32_t *source_rank_by_final_index,
    uint32_t *combine_send_offset,
    uint32_t *padded_index,
    uint32_t *num_recv_tokens_ptr,
    uint32_t *combine_send_done,
    uint32_t *token_counter,
    uint32_t *sync_counter,
    uint32_t **sync_ptrs,
    uint8_t **recv_ptrs,
    uint32_t *current_epoch,
    uint64_t stream
) {
    const size_t token_dim = round_up<size_t>(hidden_dim * x_elemsize, sizeof(int4));

    void *args[] = {
        const_cast<size_t *>(&token_dim),
        &rank,
        &expert_x_ptr,
        &expert_x_stride,
        &tx_ready,
        &send_buffer,
        &recv_buffer,
        &source_rank_by_final_index,
        &combine_send_offset,
        &padded_index,
        &num_recv_tokens_ptr,
        &combine_send_done,
        &token_counter,
        &sync_counter,
        &sync_ptrs,
        &recv_ptrs,
        &current_epoch,
    };

    dim3 dimGrid(num_blocks, 1, 1);

    cudaError_t status;

    void *wait_args[] = {
        &tx_ready,
        &num_recv_tokens_ptr,
    };

    nvtxRangePush("combine_send");
    status = cudaLaunchKernel(
        (void *)&a2a_wait_tx_ready_kernel,
        dim3(1, 1, 1),
        dim3(WARP_SIZE, 1, 1),
        wait_args,
        0,
        (cudaStream_t)stream
    );
    if (status != cudaSuccess) {
        nvtxRangePop();
        return status;
    }

    LAUNCH_DP_SIZE(dp_size, DP_SIZE, {
        LAUNCH_WORLD_SIZE(node_size, NODE_SIZE, {
            LAUNCH_TOKEN_DIM_COMBINE(token_dim, TokenDim, {
                constexpr size_t NUM_WARPS = 16;
                dim3 dimBlock(NUM_WARPS * WARP_SIZE, 1, 1);
                status = cudaLaunchCooperativeKernel(
                    (void *)&a2a_combine_send_kernel<NUM_WARPS, NODE_SIZE, DP_SIZE, TokenDim>,
                    dimGrid,
                    dimBlock,
                    args,
                    0,
                    (cudaStream_t)stream
                );
            });
        });
    });
    nvtxRangePop();
    return status;
}

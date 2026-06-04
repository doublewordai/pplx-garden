from typing import Optional

import torch

from pplx_garden.utils.math import round_up
from tests.p2p_all_to_all.data import RankTestData


def _ordered_source_groups(
    *,
    rank: int,
    dp_size: int,
    node_size: int,
    world_size: int,
) -> list[int]:
    dp_group = rank // dp_size
    rank_node = rank // node_size
    groups_per_node = node_size // dp_size
    num_nodes = world_size // node_size

    source_groups = []
    for node_offset in range(1, num_nodes):
        node = (rank_node + node_offset) % num_nodes
        for group_in_node in range(groups_per_node):
            source_groups.append(node * groups_per_node + group_in_node)

    for local_group_offset in range(1, groups_per_node):
        source_groups.append(
            rank_node * groups_per_node
            + (dp_group + local_group_offset) % groups_per_node
        )

    source_groups.append(dp_group)
    return source_groups


def _pack_layout_range(count: int, offset: int) -> int:
    return count | (offset << 32)


def expected_canonical_batched_experts_route_plan(
    *,
    rank_data: list[RankTestData],
    first_expert: int,
    num_local_experts: int,
    rank: int,
    dp_size: int,
    node_size: int,
    world_size: int,
    max_tokens_per_expert: int,
) -> dict[str, list[int] | list[list[int]] | int]:
    """Return the canonical source order and flat BatchedExperts indices."""

    source_group_order = _ordered_source_groups(
        rank=rank,
        dp_size=dp_size,
        node_size=node_size,
        world_size=world_size,
    )
    tokens_per_source_group_per_local_expert = [
        [
            int(
                rank_data[source_group]
                .expected_num_tokens[first_expert + local_expert]
                .item()
            )
            for local_expert in range(num_local_experts)
        ]
        for source_group in range(len(rank_data))
    ]

    source_rank: list[int] = []
    source_group: list[int] = []
    final_index: list[int] = []
    tokens_per_expert = [0 for _ in range(num_local_experts)]
    num_source_groups = len(rank_data)
    layout_range = [0 for _ in range(num_local_experts * num_source_groups)]
    for route_source_group in source_group_order:
        route_source_rank = route_source_group * dp_size + rank % dp_size
        for local_expert in range(num_local_experts):
            count = tokens_per_source_group_per_local_expert[
                route_source_group
            ][local_expert]
            layout_range[local_expert * num_source_groups + route_source_group] = (
                _pack_layout_range(count, tokens_per_expert[local_expert])
            )
            for _ in range(count):
                source_rank.append(route_source_rank)
                source_group.append(route_source_group)
                final_index.append(
                    local_expert * max_tokens_per_expert
                    + tokens_per_expert[local_expert]
                )
                tokens_per_expert[local_expert] += 1

    return {
        "source_group_order": source_group_order,
        "source_rank": source_rank,
        "source_group": source_group,
        "final_index": final_index,
        "tokens_per_source_group_per_local_expert": (
            tokens_per_source_group_per_local_expert
        ),
        "tokens_per_expert": tokens_per_expert,
        "layout_range": layout_range,
        "num_recv_tokens": len(final_index),
    }


def assert_canonical_batched_experts_layout(
    *,
    out_expert_x: torch.Tensor,
    out_expert_x_scale: Optional[torch.Tensor],
    expert_num_tokens: torch.Tensor,
    rank_data: list[RankTestData],
    first_expert: int,
    num_local_experts: int,
    rank: int,
    dp_size: int,
    node_size: int,
    world_size: int,
    expert_padding: int,
    max_tokens_per_expert: Optional[int],
) -> None:
    """Assert the public dispatch layout consumed by batched MoE kernels."""

    expected_counts = torch.sum(
        torch.stack([data.expected_num_tokens for data in rank_data], dim=0),
        dim=0,
        dtype=torch.int32,
    )
    expected_local_counts = expected_counts[
        first_expert : first_expert + num_local_experts
    ].to(expert_num_tokens.device)
    torch.testing.assert_close(expert_num_tokens, expected_local_counts)

    expected_tokens: list[list[torch.Tensor]] = [
        [] for _ in range(num_local_experts)
    ]
    expected_scales: Optional[list[list[torch.Tensor]]]
    if out_expert_x_scale is None:
        expected_scales = None
    else:
        expected_scales = [[] for _ in range(num_local_experts)]

    for source_group in _ordered_source_groups(
        rank=rank,
        dp_size=dp_size,
        node_size=node_size,
        world_size=world_size,
    ):
        source = rank_data[source_group]
        for token_idx in range(source.indices.shape[0]):
            for topk_idx in range(source.indices.shape[1]):
                expert = int(source.indices[token_idx, topk_idx].item())
                if not first_expert <= expert < first_expert + num_local_experts:
                    continue

                local_expert = expert - first_expert
                expected_tokens[local_expert].append(source.dp_x[token_idx])
                if expected_scales is not None:
                    assert source.dp_x_scale is not None
                    expected_scales[local_expert].append(source.dp_x_scale[token_idx])

    flat_out_expert_x: torch.Tensor
    flat_out_expert_x_scale: Optional[torch.Tensor]
    if out_expert_x.ndim == 3:
        assert max_tokens_per_expert is not None
        flat_out_expert_x = out_expert_x.reshape(-1, out_expert_x.shape[-1])
        flat_out_expert_x_scale = (
            None
            if out_expert_x_scale is None
            else out_expert_x_scale.reshape(-1, out_expert_x_scale.shape[-1])
        )
        starts = [
            local_expert * max_tokens_per_expert
            for local_expert in range(num_local_experts)
        ]
    else:
        flat_out_expert_x = out_expert_x
        flat_out_expert_x_scale = out_expert_x_scale
        starts = []
        offset = 0
        for local_expert in range(num_local_experts):
            starts.append(offset)
            offset += round_up(len(expected_tokens[local_expert]), expert_padding)

    for local_expert, tokens in enumerate(expected_tokens):
        start = starts[local_expert]
        end = start + len(tokens)
        if tokens:
            expected_x = torch.stack(tokens)
            torch.testing.assert_close(
                flat_out_expert_x[start:end],
                expected_x,
                rtol=0,
                atol=0,
            )

            if expected_scales is not None:
                assert flat_out_expert_x_scale is not None
                scales = expected_scales[local_expert]
                assert scales
                expected_x_scale = torch.stack(scales)
                torch.testing.assert_close(
                    flat_out_expert_x_scale[start:end],
                    expected_x_scale,
                    rtol=0,
                    atol=0,
                )
        else:
            assert start == end

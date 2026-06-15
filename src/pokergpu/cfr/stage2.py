from __future__ import annotations

from dataclasses import dataclass

from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.tree.public_tree import NodeType, PublicTree


@dataclass(slots=True, frozen=True)
class AggregateProbSumResult:
    node_reach_sum: tuple[float, ...]
    leaf_node_ids: tuple[int, ...]
    leaf_reach_sum: tuple[float, ...]
    leaf_batch_rows: tuple[tuple[int, float], ...]


def aggregate_prob_sum(
    tree: PublicTree,
    forward: ForwardProfileResult,
) -> AggregateProbSumResult:
    if tree.node_count != len(forward.node_reach):
        raise ValueError("tree and forward pass must cover the same number of nodes")

    leaf_node_ids = tuple(
        node_index
        for node_index, node_type in enumerate(tree.node_types)
        if node_type is NodeType.LEAF
    )
    leaf_reach_sum = tuple(forward.node_reach[node_index] for node_index in leaf_node_ids)
    leaf_batch_rows = tuple(
        (node_index, forward.node_reach[node_index]) for node_index in leaf_node_ids
    )

    return AggregateProbSumResult(
        node_reach_sum=forward.node_reach,
        leaf_node_ids=leaf_node_ids,
        leaf_reach_sum=leaf_reach_sum,
        leaf_batch_rows=leaf_batch_rows,
    )

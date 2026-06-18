from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.cfr.stage2 import AggregateProbSumResult
from pokergpu.cfr.stage3 import OpponentReachResult
from pokergpu.cfr.stage4 import ShowdownEquityResult
from pokergpu.cfr.infosets import build_dense_infoset_table
from pokergpu.tree.public_tree import NodeId
from pokergpu.tree.public_tree import NodeType
from pokergpu.tree.public_tree import PublicTree


@dataclass(slots=True, frozen=True)
class BackwardCFVResult:
    node_values: NDArray[np.float64]
    infoset_values: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.node_values.ndim != 1:
            raise ValueError("node values must be one-dimensional")
        if self.infoset_values.ndim != 1:
            raise ValueError("infoset values must be one-dimensional")
        if self.node_values.dtype != np.float64:
            raise ValueError("node values must use float64")
        if self.infoset_values.dtype != np.float64:
            raise ValueError("infoset values must use float64")


@dataclass(slots=True, frozen=True)
class BackwardCFVInput:
    tree: PublicTree
    forward: ForwardProfileResult
    aggregate: AggregateProbSumResult
    opponent_reach: OpponentReachResult
    showdown: ShowdownEquityResult
    leaf_values: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.tree.node_count != len(self.forward.node_reach):
            raise ValueError("tree and forward pass must cover the same number of nodes")
        if self.tree.node_count != len(self.aggregate.node_aggregate.reach):
            raise ValueError("tree and aggregate result must cover the same number of nodes")
        if self.tree.node_count != len(self.opponent_reach.node_opponent_reach):
            raise ValueError("tree and opponent reach result must cover the same number of nodes")
        if self.tree.node_count != len(self.showdown.node_showdown_equity):
            raise ValueError("tree and showdown result must cover the same number of nodes")
        if self.leaf_values.ndim != 1:
            raise ValueError("leaf values must be one-dimensional")
        if self.leaf_values.dtype != np.float64:
            raise ValueError("leaf values must use float64")


def backward_cfv(stage6_input: BackwardCFVInput) -> BackwardCFVResult:
    tree = stage6_input.tree
    node_values = np.zeros(tree.node_count, dtype=np.float64)
    leaf_node_ids = stage6_input.aggregate.leaf_node_ids

    if len(leaf_node_ids) != len(stage6_input.leaf_values):
        raise ValueError("leaf node ids and leaf values must align")

    for leaf_index, node_id in enumerate(leaf_node_ids):
        node_values[int(node_id)] = float(stage6_input.leaf_values[leaf_index])

    for node_index in range(tree.node_count - 1, -1, -1):
        node_type = tree.node_types[node_index]
        if node_type is NodeType.LEAF:
            continue
        if node_type is NodeType.TERMINAL:
            node_values[node_index] = float(stage6_input.showdown.node_showdown_equity[node_index])
            continue
        if node_type is NodeType.CHANCE:
            node_values[node_index] = _combine_chance_node(tree, node_index, node_values)
            continue
        node_values[node_index] = _combine_player_node(
            tree,
            node_index,
            node_values,
            stage6_input.forward.action_reach,
        )

    infoset_table = build_dense_infoset_table(tree)
    infoset_values = np.zeros(infoset_table.infoset_count, dtype=np.float64)
    infoset_totals = np.zeros(infoset_table.infoset_count, dtype=np.float64)
    for node_index, infoset_id in enumerate(tree.infoset_ids):
        if infoset_id is None:
            continue
        infoset_index = int(infoset_id)
        weight = float(stage6_input.opponent_reach.node_opponent_share[node_index])
        infoset_values[infoset_index] += weight * node_values[node_index]
        infoset_totals[infoset_index] += weight

    for infoset_index in range(infoset_values.shape[0]):
        total = infoset_totals[infoset_index]
        if total > 0.0:
            infoset_values[infoset_index] /= total

    return BackwardCFVResult(node_values=node_values, infoset_values=infoset_values)


def _combine_player_node(
    tree: PublicTree,
    node_index: int,
    node_values: NDArray[np.float64],
    action_reach: tuple[tuple[float, ...], ...],
) -> float:
    child_links = tree.child_links(NodeId(node_index))
    if not child_links:
        raise ValueError("player nodes must have children")
    strategy = action_reach[node_index]
    if len(strategy) != len(child_links):
        raise ValueError("action reach must match the node branching factor")
    total = 0.0
    for action_index, link in enumerate(child_links):
        total += float(strategy[action_index]) * float(node_values[int(link.child)])
    return total


def _combine_chance_node(
    tree: PublicTree,
    node_index: int,
    node_values: NDArray[np.float64],
) -> float:
    child_links = tree.child_links(NodeId(node_index))
    if not child_links:
        raise ValueError("chance nodes must have children")
    total = 0.0
    for link in child_links:
        if link.chance_prob is None:
            raise ValueError("chance children must define probabilities")
        total += float(link.chance_prob) * float(node_values[int(link.child)])
    return total

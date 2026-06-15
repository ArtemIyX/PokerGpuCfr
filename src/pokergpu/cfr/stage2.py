from __future__ import annotations

from dataclasses import dataclass

from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.core.board import Board, Street
from pokergpu.tree.public_tree import NodeType, PublicTree


@dataclass(slots=True, frozen=True)
class LeafFeatures:
    reach: float
    share: float
    street: int
    board_size: int


@dataclass(slots=True, frozen=True)
class LeafBatchRow:
    node_id: int
    reach: float
    features: LeafFeatures


@dataclass(slots=True, frozen=True)
class LeafBatchInput:
    rows: tuple[LeafBatchRow, ...]


@dataclass(slots=True, frozen=True)
class AggregateProbSumResult:
    node_reach_sum: tuple[float, ...]
    leaf_node_ids: tuple[int, ...]
    leaf_reach_sum: tuple[float, ...]
    leaf_batch: LeafBatchInput


def aggregate_prob_sum(
    tree: PublicTree,
    forward: ForwardProfileResult,
    board: Board | None = None,
) -> AggregateProbSumResult:
    if tree.node_count != len(forward.node_reach):
        raise ValueError("tree and forward pass must cover the same number of nodes")

    leaf_node_ids = tuple(
        node_index
        for node_index, node_type in enumerate(tree.node_types)
        if node_type is NodeType.LEAF
    )
    leaf_reach_sum = tuple(forward.node_reach[node_index] for node_index in leaf_node_ids)
    total_leaf_reach = sum(leaf_reach_sum)
    board_street = board.street if board is not None else Street.PREFLOP
    street = _street_code(board_street)
    board_size = len(board.cards) if board is not None else 0
    leaf_batch = LeafBatchInput(
        rows=tuple(
            LeafBatchRow(
                node_id=node_index,
                reach=forward.node_reach[node_index],
                features=LeafFeatures(
                    reach=forward.node_reach[node_index],
                    share=_safe_share(forward.node_reach[node_index], total_leaf_reach),
                    street=street,
                    board_size=board_size,
                ),
            )
            for node_index in leaf_node_ids
        )
    )

    return AggregateProbSumResult(
        node_reach_sum=forward.node_reach,
        leaf_node_ids=leaf_node_ids,
        leaf_reach_sum=leaf_reach_sum,
        leaf_batch=leaf_batch,
    )


def _safe_share(value: float, total: float) -> float:
    if total <= 0.0:
        return 0.0
    return value / total


def _street_code(street: Street) -> int:
    if street is Street.PREFLOP:
        return 0
    if street is Street.FLOP:
        return 1
    if street is Street.TURN:
        return 2
    return 3

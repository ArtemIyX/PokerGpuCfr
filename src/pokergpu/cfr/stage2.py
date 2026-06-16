from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.cfr.leaf_eval import LEAF_EVAL_FEATURE_WIDTH
from pokergpu.cfr.leaf_eval import LeafEvalBatchInput
from pokergpu.abstraction.hands import private_hand_count, private_hand_mask
from pokergpu.core.cards import Rank, Suit
from pokergpu.core.board import Board, Street
from pokergpu.tree.public_tree import NodeType, PublicTree


@dataclass(slots=True, frozen=True)
class LeafBatchInput:
    node_ids: tuple[int, ...]
    reach: NDArray[np.float32]
    features: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.reach.ndim != 1:
            raise ValueError("leaf batch reach must be a 1D tensor")
        if self.reach.dtype != np.float32:
            raise ValueError("leaf batch reach must use float32")
        if self.features.ndim != 2:
            raise ValueError("leaf batch features must be a 2D tensor")
        if self.features.dtype != np.float32:
            raise ValueError("leaf batch features must use float32")
        if self.features.shape[0] != len(self.node_ids):
            raise ValueError("leaf batch feature rows must match node ids")
        if self.reach.shape[0] != len(self.node_ids):
            raise ValueError("leaf batch reach rows must match node ids")


@dataclass(slots=True, frozen=True)
class NodeCardAggregate:
    reach: tuple[float, ...]
    card_reach: NDArray[np.float64]
    hand_reach: NDArray[np.float64]

    def __post_init__(self) -> None:
        assert self.reach, "node reach cannot be empty"
        if self.card_reach.ndim != 2:
            raise ValueError("node card reach must be a 2D tensor")
        if self.hand_reach.ndim != 2:
            raise ValueError("node hand reach must be a 2D tensor")
        if self.card_reach.dtype != np.float64:
            raise ValueError("node card reach must use float64")
        if self.hand_reach.dtype != np.float64:
            raise ValueError("node hand reach must use float64")
        if self.card_reach.shape[0] != len(self.reach):
            raise ValueError("node card reach rows must match node reach")
        if self.hand_reach.shape[0] != len(self.reach):
            raise ValueError("node hand reach rows must match node reach")
        if self.card_reach.shape[1] != 52:
            raise ValueError("node card reach must have width 52")
        if self.hand_reach.shape[1] != private_hand_count():
            raise ValueError("node hand reach must match private hand count")


@dataclass(slots=True, frozen=True)
class AggregateProbSumResult:
    node_aggregate: NodeCardAggregate
    leaf_node_ids: tuple[int, ...]
    leaf_reach_sum: tuple[float, ...]
    leaf_batch: LeafBatchInput

    def __post_init__(self) -> None:
        assert len(self.leaf_node_ids) == len(self.leaf_reach_sum), "leaf ids and reach must align"
        if self.leaf_batch.node_ids != self.leaf_node_ids:
            raise ValueError("leaf batch node ids must match leaf node ids")
        if self.leaf_batch.reach.shape[0] != len(self.leaf_node_ids):
            raise ValueError("leaf batch reach must match leaf node ids")
        if self.leaf_batch.features.shape[0] != len(self.leaf_node_ids):
            raise ValueError("leaf batch features must match leaf node ids")


def aggregate_prob_sum(
    tree: PublicTree,
    forward: ForwardProfileResult,
    board: Board | None = None,
    max_workers: int | None = None,
) -> AggregateProbSumResult:
    if tree.node_count != len(forward.node_reach):
        raise ValueError("tree and forward pass must cover the same number of nodes")
    if tree.node_count == 0:
        raise ValueError("tree cannot be empty")

    del max_workers

    board_card_mask, board_card_vector, leaf_card_reach_vector = _board_card_features(board)
    node_card_reach = _build_node_card_reach(forward.node_reach, board_card_mask)
    node_hand_reach = _build_node_hand_reach(forward.node_reach, board)
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
    board_signature = _board_signature(board)
    leaf_node_id_tuple = tuple(leaf_node_ids)
    leaf_reach_array = np.asarray(leaf_reach_sum, dtype=np.float32)
    if leaf_reach_array.ndim != 1:
        raise ValueError("leaf reach must be one-dimensional")
    leaf_feature_rows = _build_leaf_feature_rows(
        leaf_node_ids=leaf_node_ids,
        node_reach=forward.node_reach,
        total_leaf_reach=total_leaf_reach,
        street=street,
        board_size=board_size,
        board_signature=board_signature,
        board_card_mask=board_card_mask,
        board_card_vector=board_card_vector,
        leaf_card_reach_vector=leaf_card_reach_vector,
    )
    if leaf_feature_rows.dtype != np.float32:
        raise ValueError("leaf feature rows must use float32")
    leaf_batch = LeafBatchInput(
        node_ids=leaf_node_id_tuple,
        reach=leaf_reach_array,
        features=leaf_feature_rows,
    )

    return AggregateProbSumResult(
        node_aggregate=NodeCardAggregate(
            reach=forward.node_reach,
            card_reach=node_card_reach,
            hand_reach=node_hand_reach,
        ),
        leaf_node_ids=leaf_node_ids,
        leaf_reach_sum=leaf_reach_sum,
        leaf_batch=leaf_batch,
    )


def build_leaf_eval_batch(leaf_batch: LeafBatchInput) -> LeafEvalBatchInput:
    return LeafEvalBatchInput(
        node_ids=leaf_batch.node_ids,
        features=leaf_batch.features,
    )


def _safe_share(value: float, total: float) -> float:
    if total <= 0.0:
        return 0.0
    return value / total


def _build_leaf_feature_rows(
    *,
    leaf_node_ids: tuple[int, ...],
    node_reach: tuple[float, ...],
    total_leaf_reach: float,
    street: int,
    board_size: int,
    board_signature: int,
    board_card_mask: tuple[bool, ...],
    board_card_vector: tuple[float, ...],
    leaf_card_reach_vector: tuple[float, ...],
) -> NDArray[np.float32]:
    leaf_count = len(leaf_node_ids)
    if leaf_count == 0:
        return np.zeros((0, LEAF_EVAL_FEATURE_WIDTH), dtype=np.float32)

    reach = np.asarray([node_reach[node_index] for node_index in leaf_node_ids], dtype=np.float32)
    share = np.asarray([_safe_share(node_reach[node_index], total_leaf_reach) for node_index in leaf_node_ids], dtype=np.float32)
    street_column = np.full((leaf_count, 1), np.float32(street), dtype=np.float32)
    board_size_column = np.full((leaf_count, 1), np.float32(board_size), dtype=np.float32)
    board_signature_column = np.full((leaf_count, 1), np.float32(board_signature), dtype=np.float32)
    board_mask = np.asarray([1.0 if blocked else 0.0 for blocked in board_card_mask], dtype=np.float32)
    board_vector = np.asarray(board_card_vector, dtype=np.float32)
    leaf_vector = np.asarray(leaf_card_reach_vector, dtype=np.float32)
    if board_mask.shape[0] != 52:
        raise ValueError("board mask must have width 52")
    if board_vector.shape[0] != 52:
        raise ValueError("board vector must have width 52")
    if leaf_vector.shape[0] != 52:
        raise ValueError("leaf card reach vector must have width 52")

    repeated_board_mask = np.broadcast_to(board_mask, (leaf_count, board_mask.shape[0]))
    repeated_board_vector = np.broadcast_to(board_vector, (leaf_count, board_vector.shape[0]))
    repeated_leaf_vector = np.broadcast_to(leaf_vector, (leaf_count, leaf_vector.shape[0]))
    rows = np.concatenate(
        (
            reach[:, None],
            share[:, None],
            street_column,
            board_size_column,
            board_signature_column,
            repeated_board_mask,
            repeated_board_vector,
            repeated_leaf_vector,
        ),
        axis=1,
    )
    if rows.shape[1] != LEAF_EVAL_FEATURE_WIDTH:
        raise ValueError("leaf eval feature row has an unexpected width")
    return rows.astype(np.float32, copy=False)


def _street_code(street: Street) -> int:
    if street is Street.PREFLOP:
        return 0
    if street is Street.FLOP:
        return 1
    if street is Street.TURN:
        return 2
    return 3


def _board_signature(board: Board | None) -> int:
    if board is None:
        return 0
    signature = 0
    for card in board.cards:
        rank_value = ord(card.rank.value[0])
        suit_value = ord(card.suit.value[0])
        signature = signature * 131 + rank_value * 17 + suit_value
    return signature


def _board_card_features(
    board: Board | None,
) -> tuple[tuple[bool, ...], tuple[float, ...], tuple[float, ...]]:
    mask = [False] * 52
    if board is None:
        empty_mask = tuple(mask)
        empty_vector = tuple(0.0 for _ in empty_mask)
        return empty_mask, empty_vector, empty_vector
    for card in board.cards:
        mask[_card_index(card.rank, card.suit)] = True
    mask_tuple = tuple(mask)
    board_vector = tuple(1.0 if present else 0.0 for present in mask_tuple)
    leaf_vector = _leaf_card_reach_vector(mask_tuple, len(board.cards))
    return mask_tuple, board_vector, leaf_vector


def _card_index(rank: Rank, suit: Suit) -> int:
    suit_index = {
        Suit.CLUBS: 0,
        Suit.DIAMONDS: 1,
        Suit.HEARTS: 2,
        Suit.SPADES: 3,
    }[suit]
    rank_index = {
        Rank.TWO: 0,
        Rank.THREE: 1,
        Rank.FOUR: 2,
        Rank.FIVE: 3,
        Rank.SIX: 4,
        Rank.SEVEN: 5,
        Rank.EIGHT: 6,
        Rank.NINE: 7,
        Rank.TEN: 8,
        Rank.JACK: 9,
        Rank.QUEEN: 10,
        Rank.KING: 11,
        Rank.ACE: 12,
    }[rank]
    return suit_index * 13 + rank_index


def _leaf_card_reach_vector(board_card_mask: tuple[bool, ...], board_size: int) -> tuple[float, ...]:
    live_card_count = 52 - board_size
    if live_card_count <= 0:
        return tuple(0.0 for _ in range(52))
    weight = 1.0 / live_card_count
    return tuple(0.0 if blocked else weight for blocked in board_card_mask)


def _build_node_hand_reach(
    node_reach: tuple[float, ...],
    board: Board | None,
) -> NDArray[np.float64]:
    node_reach_array = np.asarray(node_reach, dtype=np.float64)
    live_mask = private_hand_mask(board.cards if board is not None else ())
    live_count = int(live_mask.sum())
    if live_count <= 0:
        return np.zeros((len(node_reach), private_hand_count()), dtype=np.float64)
    weights = node_reach_array / np.float64(live_count)
    return np.where(live_mask[None, :], weights[:, None], np.float64(0.0))


def _build_node_card_reach(
    node_reach: tuple[float, ...],
    board_card_mask: tuple[bool, ...],
) -> NDArray[np.float64]:
    node_reach_array = np.asarray(node_reach, dtype=np.float64)
    live_mask = np.asarray([not blocked for blocked in board_card_mask], dtype=np.bool_)
    live_count = int(live_mask.sum())
    if live_count <= 0:
        return np.zeros((len(node_reach), 52), dtype=np.float64)
    weights = node_reach_array / np.float64(live_count)
    return np.where(live_mask[None, :], weights[:, None], np.float64(0.0))

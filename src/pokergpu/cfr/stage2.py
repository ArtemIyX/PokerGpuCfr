from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os

import numpy as np
from numpy.typing import NDArray

from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.cfr.leaf_eval import LEAF_EVAL_FEATURE_WIDTH
from pokergpu.cfr.leaf_eval import LeafEvalBatchInput
from pokergpu.abstraction.hands import private_hand_count, private_hand_mask
from pokergpu.core.cards import Rank, Suit
from pokergpu.core.board import Board, Street
from pokergpu.tree.public_tree import NodeType, PublicTree

try:
    from numba import njit, prange  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    njit = None
    prange = None


@dataclass(slots=True, frozen=True)
class LeafBatchInput:
    node_ids: tuple[int, ...]
    reach: NDArray[np.float32]
    features: NDArray[np.float32]

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_ids", tuple(self.node_ids))
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
        if len(self.reach) == 0:
            raise ValueError("node reach cannot be empty")
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
    leaf_reach_sum: NDArray[np.float32]
    leaf_batch: LeafBatchInput

    def __post_init__(self) -> None:
        object.__setattr__(self, "leaf_node_ids", tuple(self.leaf_node_ids))
        if self.leaf_reach_sum.ndim != 1:
            raise ValueError("leaf reach sum must be a 1D tensor")
        if self.leaf_reach_sum.dtype != np.float32:
            raise ValueError("leaf reach sum must use float32")
        if len(self.leaf_node_ids) != self.leaf_reach_sum.shape[0]:
            raise ValueError("leaf ids and reach must align")
        if self.leaf_batch.node_ids != self.leaf_node_ids:
            raise ValueError("leaf batch node ids must match leaf node ids")
        if self.leaf_batch.reach.shape[0] != len(self.leaf_node_ids):
            raise ValueError("leaf batch reach must match leaf node ids")
        if self.leaf_batch.features.shape[0] != len(self.leaf_node_ids):
            raise ValueError("leaf batch features must match leaf node ids")


@dataclass(slots=True, frozen=True)
class Stage2PreparedInput:
    node_count: int
    board_card_mask: NDArray[np.bool_]
    board_card_block: NDArray[np.float32]
    board_card_vector: NDArray[np.float32]
    leaf_card_reach_vector: NDArray[np.float32]
    live_hand_mask: NDArray[np.bool_]
    leaf_node_ids: tuple[int, ...]
    leaf_indices: NDArray[np.int64]
    leaf_count: int
    board_street: int
    board_size: int
    board_signature: int
    node_card_reach: NDArray[np.float64]
    node_hand_reach: NDArray[np.float64]
    leaf_batch_features: NDArray[np.float32]
    leaf_reach_sum: NDArray[np.float32]
    leaf_shares: NDArray[np.float32]
    node_reach: NDArray[np.float64]
    node_reach_tuple: tuple[float, ...]


def aggregate_prob_sum(
    tree: PublicTree,
    forward: ForwardProfileResult,
    board: Board | None = None,
    max_workers: int | None = None,
) -> AggregateProbSumResult:
    prepared = prepare_stage2_input(tree, board, forward)
    result = aggregate_prob_sum_prepacked(prepared, max_workers=max_workers)
    return AggregateProbSumResult(
        node_aggregate=result.node_aggregate,
        leaf_node_ids=result.leaf_node_ids,
        leaf_reach_sum=np.asarray(result.leaf_reach_sum, dtype=np.float32),
        leaf_batch=result.leaf_batch,
    )


def aggregate_prob_sum_prepacked(
    prepared: Stage2PreparedInput,
    max_workers: int | None = None,
) -> AggregateProbSumResult:
    if prepared.node_count == 0:
        raise ValueError("tree cannot be empty")
    if _can_use_numba_parallel(max_workers):
        board_street = prepared.board_street
        _fill_node_aggregates_numba(
            prepared.node_card_reach,
            prepared.node_hand_reach,
            prepared.node_reach,
            prepared.board_card_mask,
            prepared.live_hand_mask,
        )
        prepared.leaf_batch_features[:] = _build_leaf_feature_rows(
            leaf_indices=prepared.leaf_indices,
            node_reach=prepared.node_reach,
            leaf_shares=prepared.leaf_shares,
            street=board_street,
            board_size=prepared.board_size,
            board_signature=prepared.board_signature,
            board_card_mask=prepared.board_card_mask,
            board_card_vector=prepared.board_card_vector,
            leaf_card_reach_vector=prepared.leaf_card_reach_vector,
        )
        leaf_batch = LeafBatchInput(
            node_ids=prepared.leaf_node_ids,
            reach=prepared.leaf_reach_sum,
            features=prepared.leaf_batch_features,
        )
    else:
        if max_workers is None or max_workers <= 1 or prepared.node_count <= 1:
            _fill_node_card_reach(prepared.node_card_reach, prepared.node_reach, prepared.board_card_mask)
            _fill_node_hand_reach(prepared.node_hand_reach, prepared.node_reach, prepared.live_hand_mask)
        else:
            _fill_node_aggregates_parallel(
                node_card_reach=prepared.node_card_reach,
                node_hand_reach=prepared.node_hand_reach,
                node_reach=prepared.node_reach,
                board_card_mask=prepared.board_card_mask,
                live_hand_mask=prepared.live_hand_mask,
                max_workers=max_workers,
            )
        prepared.leaf_batch_features[:] = _build_leaf_feature_rows(
            leaf_indices=prepared.leaf_indices,
            node_reach=prepared.node_reach,
            leaf_shares=prepared.leaf_shares,
            street=prepared.board_street,
            board_size=prepared.board_size,
            board_signature=prepared.board_signature,
            board_card_mask=prepared.board_card_mask,
            board_card_vector=prepared.board_card_vector,
            leaf_card_reach_vector=prepared.leaf_card_reach_vector,
        )
        leaf_batch = LeafBatchInput(
            node_ids=prepared.leaf_node_ids,
            reach=prepared.leaf_reach_sum,
            features=prepared.leaf_batch_features,
        )

    return AggregateProbSumResult(
        node_aggregate=NodeCardAggregate(
            reach=prepared.node_reach_tuple,
            card_reach=prepared.node_card_reach,
            hand_reach=prepared.node_hand_reach,
        ),
        leaf_node_ids=prepared.leaf_node_ids,
        leaf_reach_sum=prepared.leaf_reach_sum,
        leaf_batch=leaf_batch,
    )


def prepare_stage2_input(
    tree: PublicTree,
    board: Board | None,
    forward: ForwardProfileResult,
) -> Stage2PreparedInput:
    if tree.node_count != len(forward.node_reach):
        raise ValueError("tree and forward pass must cover the same number of nodes")
    if tree.node_count == 0:
        raise ValueError("tree cannot be empty")

    board_card_mask, board_card_vector, leaf_card_reach_vector = _cached_board_card_features(_board_cache_key(board))
    board_card_mask_array = np.asarray(board_card_mask, dtype=np.bool_)
    board_card_block_array = np.where(board_card_mask_array, np.float32(0.0), np.float32(1.0))
    board_card_vector_array = np.asarray(board_card_vector, dtype=np.float32)
    leaf_card_reach_vector_array = np.asarray(leaf_card_reach_vector, dtype=np.float32)
    live_hand_mask = private_hand_mask(board.cards if board is not None else ())
    node_reach_array = np.asarray(forward.node_reach, dtype=np.float64)
    leaf_node_ids = _cached_leaf_node_ids(tree.node_types)
    leaf_indices = np.asarray(leaf_node_ids, dtype=np.int64)
    leaf_reach_sum_array = node_reach_array[leaf_indices]
    total_leaf_reach = float(np.sum(leaf_reach_sum_array, dtype=np.float64))
    leaf_shares = np.zeros_like(leaf_reach_sum_array, dtype=np.float32)
    if total_leaf_reach > 0.0:
        np.divide(leaf_reach_sum_array, np.float32(total_leaf_reach), out=leaf_shares)
    board_street = _street_code(board.street if board is not None else Street.PREFLOP)
    board_size = len(board.cards) if board is not None else 0
    board_signature = _board_signature(board)
    node_card_reach = np.empty((tree.node_count, 52), dtype=np.float64)
    node_hand_reach = np.empty((tree.node_count, private_hand_count()), dtype=np.float64)
    leaf_batch_features = np.empty((len(leaf_node_ids), LEAF_EVAL_FEATURE_WIDTH), dtype=np.float32)
    return Stage2PreparedInput(
        node_count=tree.node_count,
        board_card_mask=board_card_mask_array,
        board_card_block=board_card_block_array,
        board_card_vector=board_card_vector_array,
        leaf_card_reach_vector=leaf_card_reach_vector_array,
        live_hand_mask=live_hand_mask,
        leaf_node_ids=leaf_node_ids,
        leaf_indices=leaf_indices,
        leaf_count=len(leaf_node_ids),
        board_street=board_street,
        board_size=board_size,
        board_signature=board_signature,
        node_card_reach=node_card_reach,
        node_hand_reach=node_hand_reach,
        leaf_batch_features=leaf_batch_features,
        leaf_reach_sum=np.asarray(leaf_reach_sum_array, dtype=np.float32),
        leaf_shares=leaf_shares,
        node_reach=node_reach_array,
        node_reach_tuple=tuple(float(value) for value in node_reach_array),
    )


def build_leaf_eval_batch(leaf_batch: LeafBatchInput) -> LeafEvalBatchInput:
    return LeafEvalBatchInput(
        node_ids=leaf_batch.node_ids,
        features=leaf_batch.features,
    )


def _build_leaf_feature_rows(
    *,
    leaf_indices: NDArray[np.int64],
    node_reach: NDArray[np.float64],
    leaf_shares: NDArray[np.float32],
    street: int,
    board_size: int,
    board_signature: int,
    board_card_mask: NDArray[np.bool_],
    board_card_vector: NDArray[np.float32],
    leaf_card_reach_vector: NDArray[np.float32],
) -> NDArray[np.float32]:
    leaf_count = leaf_indices.shape[0]
    if leaf_count == 0:
        return np.zeros((0, LEAF_EVAL_FEATURE_WIDTH), dtype=np.float32)

    reach = np.asarray(node_reach[leaf_indices], dtype=np.float32)
    share = np.asarray(leaf_shares, dtype=np.float32)
    street_column = np.full((leaf_count, 1), np.float32(street), dtype=np.float32)
    board_size_column = np.full((leaf_count, 1), np.float32(board_size), dtype=np.float32)
    board_signature_column = np.full((leaf_count, 1), np.float32(board_signature), dtype=np.float32)
    leaf_index_column = np.asarray(leaf_indices, dtype=np.float32)[:, None] / np.float32(max(1, leaf_indices.size - 1))
    board_mask = np.where(board_card_mask, np.float32(1.0), np.float32(0.0))
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
            leaf_index_column,
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


def _board_cache_key(board: Board | None) -> tuple[int, tuple[tuple[str, str], ...]]:
    if board is None:
        return (0, ())
    return (
        _street_code(board.street),
        tuple((card.rank.value, card.suit.value) for card in board.cards),
    )


@lru_cache(maxsize=256)
def _cached_board_card_features(
    board_key: tuple[int, tuple[tuple[str, str], ...]],
) -> tuple[tuple[bool, ...], tuple[float, ...], tuple[float, ...]]:
    _, card_pairs = board_key
    if not card_pairs:
        mask = tuple([False] * 52)
        vector = tuple(0.0 for _ in range(52))
        return mask, vector, vector
    mask_list: list[bool] = [False] * 52
    for rank_value, suit_value in card_pairs:
        rank = _rank_from_value(rank_value)
        suit = _suit_from_value(suit_value)
        mask_list[_card_index(rank, suit)] = True
    mask_tuple = tuple(mask_list)
    board_vector = tuple(1.0 if present else 0.0 for present in mask_tuple)
    leaf_vector = _leaf_card_reach_vector(mask_tuple, len(card_pairs))
    return mask_tuple, board_vector, leaf_vector


@lru_cache(maxsize=32)
def _cached_leaf_node_ids(node_types: tuple[NodeType, ...]) -> tuple[int, ...]:
    return tuple(index for index, node_type in enumerate(node_types) if node_type is NodeType.LEAF)


def _rank_from_value(value: str) -> Rank:
    return Rank(value)


def _suit_from_value(value: str) -> Suit:
    return Suit(value)


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


def _fill_node_hand_reach(
    out: NDArray[np.float64],
    node_reach: NDArray[np.float64],
    live_mask: NDArray[np.bool_],
) -> None:
    live_count = int(live_mask.sum())
    if live_count <= 0:
        out.fill(0.0)
        return
    weights = node_reach / np.float64(live_count)
    out[:] = np.where(live_mask[None, :], weights[:, None], np.float64(0.0))


def _fill_node_card_reach(
    out: NDArray[np.float64],
    node_reach: NDArray[np.float64],
    board_card_mask: NDArray[np.bool_],
) -> None:
    live_count = int((~board_card_mask).sum())
    if live_count <= 0:
        out.fill(0.0)
        return
    weights = node_reach / np.float64(live_count)
    out[:] = np.where((~board_card_mask)[None, :], weights[:, None], np.float64(0.0))


def _fill_node_aggregates_parallel(
    *,
    node_card_reach: NDArray[np.float64],
    node_hand_reach: NDArray[np.float64],
    node_reach: NDArray[np.float64],
    board_card_mask: NDArray[np.bool_],
    live_hand_mask: NDArray[np.bool_],
    max_workers: int,
) -> None:
    node_count = len(node_reach)
    worker_count = min(max_workers, node_count)
    if worker_count <= 1:
        _fill_node_card_reach(node_card_reach, node_reach, board_card_mask)
        _fill_node_hand_reach(node_hand_reach, node_reach, live_hand_mask)
        return

    bounds = _chunk_bounds(node_count, worker_count)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        list(
            executor.map(
                lambda span: _fill_node_aggregate_chunk(
                    node_card_reach=node_card_reach,
                    node_hand_reach=node_hand_reach,
                    node_reach=node_reach,
                    board_card_mask=board_card_mask,
                    live_hand_mask=live_hand_mask,
                    start=span[0],
                    stop=span[1],
                ),
                bounds,
            )
        )


def _fill_node_aggregate_chunk(
    *,
    node_card_reach: NDArray[np.float64],
    node_hand_reach: NDArray[np.float64],
    node_reach: NDArray[np.float64],
    board_card_mask: NDArray[np.bool_],
    live_hand_mask: NDArray[np.bool_],
    start: int,
    stop: int,
) -> None:
    if start >= stop:
        return
    _fill_node_card_reach(
        node_card_reach[start:stop],
        node_reach[start:stop],
        board_card_mask,
    )
    _fill_node_hand_reach(
        node_hand_reach[start:stop],
        node_reach[start:stop],
        live_hand_mask,
    )


def _chunk_bounds(node_count: int, worker_count: int) -> tuple[tuple[int, int], ...]:
    chunk_size = (node_count + worker_count - 1) // worker_count
    return tuple(
        (start, min(start + chunk_size, node_count))
        for start in range(0, node_count, chunk_size)
    )


def _can_use_numba_parallel(max_workers: int | None) -> bool:
    return (
        os.environ.get("POKERGPU_STAGE2_NUMBA", "0") == "1"
        and njit is not None
        and prange is not None
        and (max_workers is None or max_workers > 1)
    )


if njit is not None and prange is not None:

    @njit(parallel=True, cache=True)  # type: ignore[untyped-decorator]
    def _fill_node_aggregates_numba(
        node_card_reach: NDArray[np.float64],
        node_hand_reach: NDArray[np.float64],
        node_reach: NDArray[np.float64],
        board_card_mask: NDArray[np.bool_],
        live_hand_mask: NDArray[np.bool_],
    ) -> None:
        node_count = node_reach.shape[0]
        card_live_count = 0
        for i in range(board_card_mask.shape[0]):
            if not board_card_mask[i]:
                card_live_count += 1
        hand_live_count = 0
        for i in range(live_hand_mask.shape[0]):
            if live_hand_mask[i]:
                hand_live_count += 1
        for node_index in prange(node_count):
            reach = node_reach[node_index]
            if card_live_count <= 0 or reach <= 0.0:
                for card_index in range(52):
                    node_card_reach[node_index, card_index] = 0.0
            else:
                weight = reach / card_live_count
                for card_index in range(52):
                    node_card_reach[node_index, card_index] = weight if not board_card_mask[card_index] else 0.0
            if hand_live_count <= 0 or reach <= 0.0:
                for hand_index in range(live_hand_mask.shape[0]):
                    node_hand_reach[node_index, hand_index] = 0.0
            else:
                hand_weight = reach / hand_live_count
                for hand_index in range(live_hand_mask.shape[0]):
                    node_hand_reach[node_index, hand_index] = hand_weight if live_hand_mask[hand_index] else 0.0


    @njit(parallel=True, cache=True)  # type: ignore[untyped-decorator]
    def _fill_leaf_features_numba(
        leaf_features: NDArray[np.float32],
        node_reach: NDArray[np.float64],
        leaf_shares: NDArray[np.float32],
        board_mask: NDArray[np.bool_],
        board_vector: NDArray[np.float32],
        leaf_vector: NDArray[np.float32],
        street: np.float32,
        board_size: np.float32,
        board_signature: np.float32,
        leaf_node_ids: NDArray[np.int64],
    ) -> None:
        leaf_count = leaf_node_ids.shape[0]
        for leaf_index in prange(leaf_count):
            node_index = leaf_node_ids[leaf_index]
            leaf_features[leaf_index, 0] = np.float32(node_reach[node_index])
            leaf_features[leaf_index, 1] = leaf_shares[leaf_index]
            leaf_features[leaf_index, 2] = street
            leaf_features[leaf_index, 3] = board_size
            leaf_features[leaf_index, 4] = board_signature
            offset = 5
            for i in range(52):
                leaf_features[leaf_index, offset + i] = board_mask[i]
            offset += 52
            for i in range(52):
                leaf_features[leaf_index, offset + i] = board_vector[i]
            offset += 52
            for i in range(52):
                leaf_features[leaf_index, offset + i] = leaf_vector[i]

else:

    def _fill_node_aggregates_numba(
        node_card_reach: NDArray[np.float64],
        node_hand_reach: NDArray[np.float64],
        node_reach: NDArray[np.float64],
        board_card_mask: NDArray[np.bool_],
        live_hand_mask: NDArray[np.bool_],
    ) -> None:
        raise RuntimeError("numba is not available")

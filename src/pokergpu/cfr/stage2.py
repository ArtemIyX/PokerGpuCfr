from __future__ import annotations

from dataclasses import dataclass

from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.core.cards import Rank, Suit
from pokergpu.core.board import Board, Street
from pokergpu.tree.public_tree import NodeType, PublicTree


@dataclass(slots=True, frozen=True)
class LeafFeatures:
    reach: float
    share: float
    street: int
    board_size: int
    board_signature: int
    board_card_mask: tuple[bool, ...]
    board_card_vector: tuple[float, ...]
    leaf_card_reach_vector: tuple[float, ...]


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
    board_signature = _board_signature(board)
    board_card_mask, board_card_vector = _board_card_features(board)
    leaf_card_reach_vector = _leaf_card_reach_vector(board_card_mask, board_size)
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
                    board_signature=board_signature,
                    board_card_mask=board_card_mask,
                    board_card_vector=board_card_vector,
                    leaf_card_reach_vector=leaf_card_reach_vector,
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


def _board_signature(board: Board | None) -> int:
    if board is None:
        return 0
    signature = 0
    for card in board.cards:
        rank_value = ord(card.rank.value[0])
        suit_value = ord(card.suit.value[0])
        signature = signature * 131 + rank_value * 17 + suit_value
    return signature


def _board_card_features(board: Board | None) -> tuple[tuple[bool, ...], tuple[float, ...]]:
    mask = [False] * 52
    if board is None:
        empty_mask = tuple(mask)
        return empty_mask, tuple(0.0 for _ in empty_mask)
    for card in board.cards:
        mask[_card_index(card.rank, card.suit)] = True
    mask_tuple = tuple(mask)
    return mask_tuple, tuple(1.0 if present else 0.0 for present in mask_tuple)


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


def _leaf_card_reach_vector(
    board_card_mask: tuple[bool, ...],
    board_size: int,
) -> tuple[float, ...]:
    live_card_count = 52 - board_size
    if live_card_count <= 0:
        return tuple(0.0 for _ in range(52))
    weight = 1.0 / live_card_count
    return tuple(0.0 if blocked else weight for blocked in board_card_mask)

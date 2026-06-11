from __future__ import annotations

from dataclasses import dataclass

from .board import Board
from .cards import Card, Suit


@dataclass(frozen=True, slots=True)
class BoardCanonicalization:
    board: Board
    canonical_key: str
    suit_map: dict[Suit, Suit]


def canonicalize_board(board: Board) -> BoardCanonicalization:
    if board.is_preflop:
        return BoardCanonicalization(board=board, canonical_key="", suit_map={})

    suit_order = _canonical_suit_order(board.cards)
    canonical_suits = (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES)
    suit_map = {
        original: canonical_suits[index]
        for index, original in enumerate(suit_order)
    }
    canonical_cards = tuple(
        Card(rank=card.rank, suit=suit_map[card.suit])
        for card in board.cards
    )
    canonical_board = Board(cards=canonical_cards)
    return BoardCanonicalization(
        board=canonical_board,
        canonical_key=str(canonical_board),
        suit_map=suit_map,
    )


def canonical_board_key(board: Board) -> str:
    return canonicalize_board(board).canonical_key


def canonicalize_cards(cards: tuple[Card, ...]) -> tuple[Card, ...]:
    if len(cards) == 0:
        return cards
    suit_order = _canonical_suit_order(cards)
    canonical_suits = (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES)
    suit_map = {
        original: canonical_suits[index]
        for index, original in enumerate(suit_order)
    }
    return tuple(Card(rank=card.rank, suit=suit_map[card.suit]) for card in cards)


def _canonical_suit_order(cards: tuple[Card, ...]) -> tuple[Suit, ...]:
    suit_stats: dict[Suit, tuple[int, int]] = {
        suit: (0, 0) for suit in Suit
    }
    for card in cards:
        count, best_rank = suit_stats[card.suit]
        suit_stats[card.suit] = (count + 1, max(best_rank, card.rank.order_value))
    ordered = sorted(
        Suit,
        key=lambda suit: (
            -suit_stats[suit][0],
            -suit_stats[suit][1],
            suit.value,
        ),
    )
    return tuple(ordered)

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum
from random import Random


class Suit(StrEnum):
    CLUBS = "c"
    DIAMONDS = "d"
    HEARTS = "h"
    SPADES = "s"

    @classmethod
    def from_char(cls, value: str) -> Suit:
        normalized = value.strip().lower()
        for suit in cls:
            if suit.value == normalized:
                return suit
        raise ValueError(f"invalid suit: {value!r}")


class Rank(StrEnum):
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "T"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"

    @property
    def order_value(self) -> int:
        return _RANK_ORDER[self]

    @classmethod
    def from_char(cls, value: str) -> Rank:
        normalized = value.strip().upper()
        for rank in cls:
            if rank.value == normalized:
                return rank
        raise ValueError(f"invalid rank: {value!r}")


_RANK_ORDER: dict[Rank, int] = {
    Rank.TWO: 2,
    Rank.THREE: 3,
    Rank.FOUR: 4,
    Rank.FIVE: 5,
    Rank.SIX: 6,
    Rank.SEVEN: 7,
    Rank.EIGHT: 8,
    Rank.NINE: 9,
    Rank.TEN: 10,
    Rank.JACK: 11,
    Rank.QUEEN: 12,
    Rank.KING: 13,
    Rank.ACE: 14,
}


@dataclass(slots=True, frozen=True, order=True)
class Card:
    rank: Rank
    suit: Suit

    @classmethod
    def from_str(cls, value: str) -> Card:
        normalized = value.strip()
        if len(normalized) != 2:
            raise ValueError(f"invalid card literal: {value!r}")
        return cls(
            rank=Rank.from_char(normalized[0]),
            suit=Suit.from_char(normalized[1]),
        )

    def __str__(self) -> str:
        return f"{self.rank.value}{self.suit.value}"


def card_from_str(value: str) -> Card:
    return Card.from_str(value)


def cards_from_str(value: str) -> tuple[Card, ...]:
    normalized = value.strip()
    if len(normalized) % 2 != 0:
        raise ValueError("card string must contain pairs of rank+suit")
    return tuple(
        Card.from_str(normalized[index : index + 2])
        for index in range(0, len(normalized), 2)
    )


def format_cards(cards: tuple[Card, ...] | list[Card]) -> str:
    return "".join(str(card) for card in cards)


def make_deck() -> tuple[Card, ...]:
    return tuple(Card(rank=rank, suit=suit) for suit in Suit for rank in Rank)


def shuffled_deck(rng: Random | None = None) -> list[Card]:
    deck = list(make_deck())
    if rng is None:
        random.shuffle(deck)
    else:
        rng.shuffle(deck)
    return deck

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

import numpy as np
from numpy.typing import NDArray

from pokergpu.core.cards import Card, make_deck

PrivateHandIndex = NewType("PrivateHandIndex", int)


@dataclass(frozen=True, slots=True, order=True)
class PrivateHand:
    first: Card
    second: Card

    def __post_init__(self) -> None:
        if self.first == self.second:
            raise ValueError("private hand must contain two distinct cards")
        if _CARD_TO_INDEX[self.first] >= _CARD_TO_INDEX[self.second]:
            raise ValueError("private hand cards must be in canonical order")

    @classmethod
    def from_cards(cls, first: Card, second: Card) -> PrivateHand:
        if first == second:
            raise ValueError("private hand must contain two distinct cards")
        if _CARD_TO_INDEX[first] < _CARD_TO_INDEX[second]:
            return cls(first=first, second=second)
        return cls(first=second, second=first)

    def contains(self, card: Card) -> bool:
        return self.first == card or self.second == card


_DECK: tuple[Card, ...] = make_deck()
_CARD_TO_INDEX: dict[Card, int] = {card: index for index, card in enumerate(_DECK)}
_PRIVATE_HANDS: tuple[PrivateHand, ...] = tuple(
    PrivateHand(first=_DECK[first_index], second=_DECK[second_index])
    for first_index in range(len(_DECK))
    for second_index in range(first_index + 1, len(_DECK))
)
_PRIVATE_HAND_TO_INDEX: dict[PrivateHand, PrivateHandIndex] = {
    hand: PrivateHandIndex(index) for index, hand in enumerate(_PRIVATE_HANDS)
}


def all_private_hands() -> tuple[PrivateHand, ...]:
    return _PRIVATE_HANDS


def private_hand_count() -> int:
    return len(_PRIVATE_HANDS)


def private_hand_index(first: Card, second: Card) -> PrivateHandIndex:
    return _PRIVATE_HAND_TO_INDEX[PrivateHand.from_cards(first, second)]


def private_hand_from_index(index: PrivateHandIndex) -> PrivateHand:
    index_value = int(index)
    if index_value < 0 or index_value >= len(_PRIVATE_HANDS):
        raise IndexError(f"private hand index out of range: {index_value}")
    return _PRIVATE_HANDS[index_value]


def private_hand_mask(dead_cards: tuple[Card, ...] | list[Card]) -> NDArray[np.bool_]:
    dead_card_set = set(dead_cards)
    if len(dead_card_set) != len(dead_cards):
        raise ValueError("dead cards must be unique")
    return np.array(
        [
            hand.first not in dead_card_set and hand.second not in dead_card_set
            for hand in _PRIVATE_HANDS
        ],
        dtype=np.bool_,
    )


@dataclass(frozen=True, slots=True)
class RangeVector:
    values: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.values.ndim != 1:
            raise ValueError("range vector must be one-dimensional")
        if self.values.shape[0] != private_hand_count():
            raise ValueError(
                f"range vector must have length {private_hand_count()}, "
                f"got {self.values.shape[0]}"
            )
        if np.any(self.values < 0):
            raise ValueError("range vector values must be non-negative")

    @classmethod
    def zeros(cls) -> RangeVector:
        return cls(np.zeros(private_hand_count(), dtype=np.float32))

    @classmethod
    def uniform(cls) -> RangeVector:
        value = np.float32(1.0 / private_hand_count())
        return cls(np.full(private_hand_count(), value, dtype=np.float32))

    @classmethod
    def from_values(cls, values: NDArray[np.float32] | list[float]) -> RangeVector:
        return cls(np.asarray(values, dtype=np.float32))

    def total_weight(self) -> float:
        return float(np.sum(self.values, dtype=np.float64))

    def normalized(self) -> RangeVector:
        total = self.total_weight()
        if total <= 0.0:
            raise ValueError("cannot normalize a zero-weight range")
        return RangeVector(self.values / np.float32(total))

    def masked(self, dead_cards: tuple[Card, ...] | list[Card]) -> RangeVector:
        mask = private_hand_mask(dead_cards)
        return RangeVector(np.where(mask, self.values, np.float32(0.0)))

    def normalized_masked(
        self,
        dead_cards: tuple[Card, ...] | list[Card],
    ) -> RangeVector:
        return self.masked(dead_cards).normalized()

    def weight(self, first: Card, second: Card) -> float:
        return float(self.values[int(private_hand_index(first, second))])

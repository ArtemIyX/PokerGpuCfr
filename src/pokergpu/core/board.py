from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .cards import Card, cards_from_str, format_cards


class Street(StrEnum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


_VALID_BOARD_SIZES: dict[int, Street] = {
    0: Street.PREFLOP,
    3: Street.FLOP,
    4: Street.TURN,
    5: Street.RIVER,
}


@dataclass(slots=True, frozen=True)
class Board:
    cards: tuple[Card, ...]

    def __post_init__(self) -> None:
        if len(self.cards) not in _VALID_BOARD_SIZES:
            raise ValueError("board must contain 0, 3, 4, or 5 cards")
        if len(set(self.cards)) != len(self.cards):
            raise ValueError("board cannot contain duplicate cards")

    @classmethod
    def from_str(cls, value: str) -> Board:
        return cls(cards=cards_from_str(value))

    @property
    def street(self) -> Street:
        return _VALID_BOARD_SIZES[len(self.cards)]

    @property
    def is_preflop(self) -> bool:
        return self.street is Street.PREFLOP

    @property
    def is_flop(self) -> bool:
        return self.street is Street.FLOP

    @property
    def is_turn(self) -> bool:
        return self.street is Street.TURN

    @property
    def is_river(self) -> bool:
        return self.street is Street.RIVER

    def __str__(self) -> str:
        return format_cards(self.cards)


def board_from_str(value: str) -> Board:
    return Board.from_str(value)

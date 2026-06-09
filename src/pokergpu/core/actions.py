from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .betting import Chips


class ActionType(StrEnum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


@dataclass(slots=True, frozen=True)
class Action:
    action_type: ActionType
    amount: Chips | None = None

    def __post_init__(self) -> None:
        if self.action_type in {ActionType.BET, ActionType.RAISE}:
            if self.amount is None or self.amount <= 0:
                raise ValueError("bet and raise actions require a positive amount")
        elif self.amount is not None and self.amount < 0:
            raise ValueError("action amount cannot be negative")

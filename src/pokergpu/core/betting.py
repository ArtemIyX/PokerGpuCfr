from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

Chips = NewType("Chips", int)
PlayerIndex = NewType("PlayerIndex", int)


def chips(value: int) -> Chips:
    if value < 0:
        raise ValueError("chips must be non-negative")
    return Chips(value)


@dataclass(slots=True, frozen=True)
class BlindStructure:
    small_blind: Chips
    big_blind: Chips
    ante: Chips = Chips(0)

    def __post_init__(self) -> None:
        if self.small_blind <= 0:
            raise ValueError("small_blind must be positive")
        if self.big_blind <= 0:
            raise ValueError("big_blind must be positive")
        if self.small_blind > self.big_blind:
            raise ValueError("small_blind cannot exceed big_blind")
        if self.ante < 0:
            raise ValueError("ante must be non-negative")


@dataclass(slots=True, frozen=True)
class PlayerStack:
    player: PlayerIndex
    stack: Chips

    def __post_init__(self) -> None:
        if self.player < 0:
            raise ValueError("player index must be non-negative")
        if self.stack < 0:
            raise ValueError("stack must be non-negative")


@dataclass(slots=True, frozen=True)
class Pot:
    amount: Chips = Chips(0)

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError("pot amount must be non-negative")

    def add(self, value: Chips) -> Pot:
        if value < 0:
            raise ValueError("pot increment must be non-negative")
        return Pot(amount=Chips(self.amount + value))


@dataclass(slots=True, frozen=True)
class PlayerBet:
    player: PlayerIndex
    committed: Chips = Chips(0)
    folded: bool = False
    all_in: bool = False

    def __post_init__(self) -> None:
        if self.player < 0:
            raise ValueError("player index must be non-negative")
        if self.committed < 0:
            raise ValueError("committed chips must be non-negative")
        if self.folded and self.all_in:
            raise ValueError("player cannot be folded and all-in")


@dataclass(slots=True, frozen=True)
class BettingRoundState:
    pot: Pot
    stacks: tuple[PlayerStack, ...]
    bets: tuple[PlayerBet, ...]
    blinds: BlindStructure
    to_act: PlayerIndex

    def __post_init__(self) -> None:
        if not self.stacks:
            raise ValueError("betting round requires at least one player stack")
        if len(self.stacks) != len(self.bets):
            raise ValueError("stacks and bets must have the same length")
        if self.to_act < 0:
            raise ValueError("to_act must be non-negative")

        stack_players = {stack.player for stack in self.stacks}
        bet_players = {bet.player for bet in self.bets}

        if len(stack_players) != len(self.stacks):
            raise ValueError("stack players must be unique")
        if len(bet_players) != len(self.bets):
            raise ValueError("bet players must be unique")

        if stack_players != bet_players:
            raise ValueError("stack players and bet players must match")
        if self.to_act not in stack_players:
            raise ValueError("to_act must reference an active player index")

    @property
    def player_count(self) -> int:
        return len(self.stacks)

    @property
    def highest_bet(self) -> Chips:
        return Chips(max((bet.committed for bet in self.bets), default=0))

    def amount_to_call(self, player: PlayerIndex) -> Chips:
        committed = next(
            (bet.committed for bet in self.bets if bet.player == player),
            None,
        )
        if committed is None:
            raise ValueError("unknown player")
        return Chips(max(0, self.highest_bet - committed))

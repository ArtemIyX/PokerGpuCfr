from __future__ import annotations

from dataclasses import dataclass

from .betting import BettingRoundState, Chips, PlayerIndex


@dataclass(slots=True, frozen=True)
class RaiseBounds:
    min_to: Chips
    max_to: Chips

    @property
    def is_forced_all_in_only(self) -> bool:
        return self.min_to >= self.max_to


def player_stack(state: BettingRoundState, player: PlayerIndex) -> Chips:
    stack = next(
        (entry.stack for entry in state.stacks if entry.player == player),
        None,
    )
    if stack is None:
        raise ValueError("unknown player")
    return stack


def player_committed(state: BettingRoundState, player: PlayerIndex) -> Chips:
    committed = next(
        (entry.committed for entry in state.bets if entry.player == player),
        None,
    )
    if committed is None:
        raise ValueError("unknown player")
    return committed


def stack_after_call(state: BettingRoundState, player: PlayerIndex) -> Chips:
    stack = player_stack(state, player)
    to_call = state.amount_to_call(player)
    return Chips(max(0, stack - to_call))


def max_raise_to(state: BettingRoundState, player: PlayerIndex) -> Chips:
    return Chips(player_committed(state, player) + player_stack(state, player))


def min_raise_increment(state: BettingRoundState) -> Chips:
    commitments = sorted((bet.committed for bet in state.bets), reverse=True)
    highest = commitments[0] if commitments else 0
    second_highest = commitments[1] if len(commitments) > 1 else 0
    increment = highest - second_highest
    if increment <= 0:
        return state.blinds.big_blind
    return Chips(increment)


def min_raise_to(state: BettingRoundState, player: PlayerIndex) -> Chips:
    highest = state.highest_bet
    minimum = Chips(highest + min_raise_increment(state))
    return Chips(min(max_raise_to(state, player), minimum))


def raise_bounds(state: BettingRoundState, player: PlayerIndex) -> RaiseBounds:
    return RaiseBounds(
        min_to=min_raise_to(state, player),
        max_to=max_raise_to(state, player),
    )

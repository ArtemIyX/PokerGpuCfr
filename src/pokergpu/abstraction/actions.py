from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pokergpu.core.actions import Action, ActionType
from pokergpu.core.betting import Chips
from pokergpu.core.legality import can_bet, can_call, can_check, can_fold, can_raise
from pokergpu.core.rules import max_raise_to, min_raise_to
from pokergpu.core.state import GameState


class ActionAbstraction(Protocol):
    def legal_actions(self, state: GameState) -> tuple[Action, ...]:
        ...


@dataclass(slots=True, frozen=True)
class BaselineActionAbstraction:
    bet_sizes: tuple[float, ...] = (1.0,)
    raise_to_multipliers: tuple[float, ...] = (1.0, 1.5)

    def legal_actions(self, state: GameState) -> tuple[Action, ...]:
        actions: list[Action] = []
        betting_state = state.betting_round

        if can_fold(betting_state):
            actions.append(Action(ActionType.FOLD))
        if can_check(betting_state):
            actions.append(Action(ActionType.CHECK))
        if can_call(betting_state):
            actions.append(Action(ActionType.CALL))

        if can_bet(betting_state):
            max_bet = betting_state.stacks[betting_state.to_act].stack
            for size in self.bet_sizes:
                amount = Chips(int(round(betting_state.blinds.big_blind * size)))
                clamped = Chips(
                    min(max_bet, max(betting_state.blinds.big_blind, amount))
                )
                candidate = Action(ActionType.BET, amount=clamped)
                if candidate not in actions:
                    actions.append(candidate)

        if can_raise(betting_state):
            minimum = min_raise_to(betting_state, betting_state.to_act)
            maximum = max_raise_to(betting_state, betting_state.to_act)
            span = maximum - minimum
            for multiplier in self.raise_to_multipliers:
                amount = Chips(int(round(minimum + span * max(0.0, multiplier - 1.0))))
                clamped = Chips(min(maximum, max(minimum, amount)))
                candidate = Action(ActionType.RAISE, amount=clamped)
                if candidate not in actions:
                    actions.append(candidate)

        return tuple(actions)

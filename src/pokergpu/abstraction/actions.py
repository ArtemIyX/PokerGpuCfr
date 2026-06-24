from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pokergpu.core.actions import Action, ActionType
from pokergpu.core.betting import Chips
from pokergpu.core.board import Street
from pokergpu.core.legality import can_bet, can_call, can_check, can_fold, can_raise
from pokergpu.core.rules import max_raise_to, min_raise_to
from pokergpu.core.state import GameState


class ActionAbstraction(Protocol):
    def legal_actions(self, state: GameState) -> tuple[Action, ...]:
        ...


@dataclass(slots=True, frozen=True)
class StreetActionTemplate:
    bet_sizes: tuple[float, ...]
    raise_to_multipliers: tuple[float, ...]


@dataclass(slots=True, frozen=True)
class AbstractionProfile:
    name: str
    street_templates: dict[Street, StreetActionTemplate]

    def template_for_street(self, street: Street) -> StreetActionTemplate:
        return self.street_templates[street]


def action_labels_for_street(
    profile: AbstractionProfile,
    street: Street,
    *,
    include_check: bool = True,
) -> tuple[str, ...]:
    template = profile.template_for_street(street)
    prefix = "" if street is Street.PREFLOP else f"{street.value}_"
    labels: list[str] = []
    if include_check:
        labels.append(f"{prefix}check")
    labels.append(f"{prefix}fold")
    labels.append(f"{prefix}call")
    labels.extend(f"{prefix}bet:{int(round(size * 100))}pct" for size in template.bet_sizes)
    labels.extend(
        f"{prefix}raise:{int(round(multiplier * 100))}pct_min"
        for multiplier in template.raise_to_multipliers
    )
    labels.append(f"{prefix}bet:allin")
    labels.append(f"{prefix}raise:allin")
    return tuple(labels)


def make_default_profile() -> AbstractionProfile:
    return AbstractionProfile(
        name="default",
        street_templates={
            Street.PREFLOP: StreetActionTemplate(
                bet_sizes=(0.5, 1.0),
                raise_to_multipliers=(1.0, 1.5),
            ),
            Street.FLOP: StreetActionTemplate(
                bet_sizes=(0.25, 0.5, 0.75, 1.0, 1.5),
                raise_to_multipliers=(1.0, 1.5, 2.0),
            ),
            Street.TURN: StreetActionTemplate(
                bet_sizes=(0.25, 0.5, 0.75, 1.0, 1.5),
                raise_to_multipliers=(1.0, 1.5, 2.0),
            ),
            Street.RIVER: StreetActionTemplate(
                bet_sizes=(0.25, 0.5, 0.75, 1.0, 1.5),
                raise_to_multipliers=(1.0, 1.5, 2.0),
            ),
        },
    )


def make_holdem_hu_profile() -> AbstractionProfile:
    return AbstractionProfile(
        name="holdem_hu",
        street_templates={
            Street.PREFLOP: StreetActionTemplate(
                bet_sizes=(0.25, 0.5, 0.75, 1.0, 1.5),
                raise_to_multipliers=(1.0, 1.5),
            ),
            Street.FLOP: StreetActionTemplate(
                bet_sizes=(0.25, 0.5, 0.75, 1.0),
                raise_to_multipliers=(1.0, 1.5, 2.0),
            ),
            Street.TURN: StreetActionTemplate(
                bet_sizes=(0.33, 0.5, 0.66, 1.0),
                raise_to_multipliers=(1.0, 1.5, 2.0),
            ),
            Street.RIVER: StreetActionTemplate(
                bet_sizes=(0.33, 0.66, 1.0, 1.5, 2.0),
                raise_to_multipliers=(1.0, 1.5),
            ),
        },
    )


def make_compact_profile() -> AbstractionProfile:
    return AbstractionProfile(
        name="compact",
        street_templates={
            street: StreetActionTemplate(
                bet_sizes=(1.0,),
                raise_to_multipliers=(1.0,),
            )
            for street in Street
        },
    )


@dataclass(slots=True, frozen=True)
class BaselineActionAbstraction:
    profile: AbstractionProfile = make_default_profile()

    def legal_actions(self, state: GameState) -> tuple[Action, ...]:
        actions: list[Action] = []
        betting_state = state.betting_round
        template = self.profile.template_for_street(state.current_street)

        if can_fold(betting_state):
            actions.append(Action(ActionType.FOLD))
        if can_check(betting_state):
            actions.append(Action(ActionType.CHECK))
        if can_call(betting_state):
            actions.append(Action(ActionType.CALL))

        if can_bet(betting_state):
            max_bet = next(
                stack.stack
                for stack in betting_state.stacks
                if stack.player == betting_state.to_act
            )
            for size in template.bet_sizes:
                amount = Chips(int(round(betting_state.pot.amount * size)))
                clamped = Chips(min(max_bet, max(betting_state.blinds.big_blind, amount)))
                _append_unique_action(actions, Action(ActionType.BET, amount=clamped))
            _append_unique_action(actions, Action(ActionType.BET, amount=max_bet))

        if can_raise(betting_state):
            minimum = min_raise_to(betting_state, betting_state.to_act)
            maximum = max_raise_to(betting_state, betting_state.to_act)
            for multiplier in template.raise_to_multipliers:
                amount = Chips(int(round(float(minimum) * max(1.0, multiplier))))
                clamped = Chips(min(maximum, max(minimum, amount)))
                _append_unique_action(actions, Action(ActionType.RAISE, amount=clamped))
            _append_unique_action(actions, Action(ActionType.RAISE, amount=maximum))

        return tuple(actions)


def _append_unique_action(actions: list[Action], action: Action) -> None:
    if action not in actions:
        actions.append(action)

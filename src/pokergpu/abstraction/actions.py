from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pokergpu.core.actions import Action, ActionType
from pokergpu.core.betting import BettingRoundState, Chips, PlayerIndex
from pokergpu.core.board import Street
from pokergpu.core.legality import can_bet, can_call, can_check, can_fold, can_raise
from pokergpu.core.rules import max_raise_to, min_raise_to, player_stack
from pokergpu.core.state import GameState


class ActionAbstraction(Protocol):
    def legal_actions(self, state: GameState) -> tuple[Action, ...]:
        ...

    def abstraction_id(self, state: GameState | None = None) -> str:
        ...


@dataclass(slots=True, frozen=True)
class StreetActionTemplate:
    bet_sizes: tuple[float, ...]
    raise_to_multipliers: tuple[float, ...]
    allow_all_in: bool = True


@dataclass(slots=True, frozen=True)
class AbstractionProfile:
    name: str
    version: str
    street_templates: dict[Street, StreetActionTemplate]
    position_overrides: dict[str, dict[Street, StreetActionTemplate]] | None = None

    def __post_init__(self) -> None:
        if self.position_overrides is None:
            object.__setattr__(self, "position_overrides", {})

    def template_for_street(
        self,
        street: Street,
        *,
        position_group: str | None = None,
    ) -> StreetActionTemplate:
        overrides = self.position_overrides or {}
        if position_group is not None:
            override = overrides.get(position_group, {}).get(street)
            if override is not None:
                return override
        return self.street_templates[street]

    def profile_id(self) -> str:
        return f"{self.name}:{self.version}"


def make_default_profile() -> AbstractionProfile:
    return AbstractionProfile(
        name="default",
        version="v2",
        street_templates={
            Street.PREFLOP: StreetActionTemplate(
                bet_sizes=(2.0, 2.5, 3.0, 4.0, 6.0),
                raise_to_multipliers=(1.0, 1.5, 2.0, 3.0),
            ),
            Street.FLOP: StreetActionTemplate(
                bet_sizes=(0.33, 0.5, 0.75, 1.25),
                raise_to_multipliers=(1.0, 1.5, 2.0),
            ),
            Street.TURN: StreetActionTemplate(
                bet_sizes=(0.5, 0.75, 1.25),
                raise_to_multipliers=(1.0, 1.5, 2.0),
            ),
            Street.RIVER: StreetActionTemplate(
                bet_sizes=(0.5, 0.75, 1.25),
                raise_to_multipliers=(1.0, 1.5, 2.0),
            ),
        },
        position_overrides={
            "early": {
                Street.PREFLOP: StreetActionTemplate(
                    bet_sizes=(2.0, 2.5, 3.0),
                    raise_to_multipliers=(1.0, 1.5, 2.0),
                )
            },
            "middle": {
                Street.PREFLOP: StreetActionTemplate(
                    bet_sizes=(2.0, 2.5, 3.0, 4.0),
                    raise_to_multipliers=(1.0, 1.5, 2.0, 3.0),
                )
            },
            "late": {
                Street.PREFLOP: StreetActionTemplate(
                    bet_sizes=(2.0, 2.5, 3.0, 4.0, 6.0),
                    raise_to_multipliers=(1.0, 1.5, 2.0, 3.0, 4.0),
                )
            },
            "blinds": {
                Street.PREFLOP: StreetActionTemplate(
                    bet_sizes=(2.5, 3.5, 5.0),
                    raise_to_multipliers=(1.0, 1.5, 2.5, 4.0),
                )
            },
        },
    )


def make_compact_profile() -> AbstractionProfile:
    return AbstractionProfile(
        name="compact",
        version="v2",
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

    def abstraction_id(self, state: GameState | None = None) -> str:
        if state is None:
            return self.profile.profile_id()
        return "|".join(
            (
                self.profile.profile_id(),
                state.current_street.value,
                self.position_group_for_state(state),
            )
        )

    def position_group_for_state(self, state: GameState) -> str:
        if state.current_street is not Street.PREFLOP:
            if self._is_in_position(state):
                return "ip"
            return "oop"

        ordered_players = self._preflop_action_order(state)
        acting_player = state.betting_round.to_act
        if len(ordered_players) >= 6:
            early = {ordered_players[0], ordered_players[1]}
            middle = {ordered_players[2], ordered_players[3]}
            late = {ordered_players[4]}
            blinds = {ordered_players[5]}
            if acting_player in early:
                return "early"
            if acting_player in middle:
                return "middle"
            if acting_player in late:
                return "late"
            if acting_player in blinds:
                return "blinds"
        return "late"

    def legal_actions(self, state: GameState) -> tuple[Action, ...]:
        actions: list[Action] = []
        betting_state = state.betting_round
        position_group = self.position_group_for_state(state)
        template = self.profile.template_for_street(
            state.current_street,
            position_group=position_group,
        )

        if can_fold(betting_state):
            actions.append(Action(ActionType.FOLD))
        if can_check(betting_state):
            actions.append(Action(ActionType.CHECK))
        if can_call(betting_state):
            actions.append(Action(ActionType.CALL))

        if can_bet(betting_state):
            actions.extend(
                self._build_bets(betting_state, template.bet_sizes)
            )

        if can_raise(betting_state):
            actions.extend(
                self._build_raises(betting_state, template.raise_to_multipliers)
            )

        return tuple(self._dedupe_sorted(actions))

    def _build_bets(
        self,
        betting_state: BettingRoundState,
        bet_sizes: tuple[float, ...],
    ) -> list[Action]:
        max_bet = player_stack(betting_state, betting_state.to_act)
        pot = betting_state.pot.amount
        big_blind = betting_state.blinds.big_blind
        actions: list[Action] = []
        for size in bet_sizes:
            target = Chips(int(round(pot * size)))
            clamped = Chips(min(max_bet, max(big_blind, target)))
            candidate = Action(ActionType.BET, amount=clamped)
            actions.append(candidate)
        return actions

    def _build_raises(
        self,
        betting_state: BettingRoundState,
        multipliers: tuple[float, ...],
    ) -> list[Action]:
        minimum = min_raise_to(betting_state, betting_state.to_act)
        maximum = max_raise_to(betting_state, betting_state.to_act)
        span = max(0, int(maximum - minimum))
        actions: list[Action] = []
        for multiplier in multipliers:
            target = Chips(int(round(int(minimum) + span * max(0.0, multiplier - 1.0))))
            clamped = Chips(min(maximum, max(minimum, target)))
            candidate = Action(ActionType.RAISE, amount=clamped)
            actions.append(candidate)
        return actions

    def _dedupe_sorted(self, actions: list[Action]) -> list[Action]:
        priority = {
            ActionType.FOLD: 0,
            ActionType.CHECK: 1,
            ActionType.CALL: 2,
            ActionType.BET: 3,
            ActionType.RAISE: 4,
        }
        seen: set[tuple[ActionType, Chips | None]] = set()
        ordered = sorted(
            actions,
            key=lambda action: (
                priority[action.action_type],
                int(action.amount or 0),
            ),
        )
        unique: list[Action] = []
        for action in ordered:
            key = (action.action_type, action.amount)
            if key in seen:
                continue
            seen.add(key)
            unique.append(action)
        return unique

    def _preflop_action_order(self, state: GameState) -> tuple[PlayerIndex, ...]:
        players = tuple(player.player for player in state.players)
        dealer_index = next(
            index for index, player in enumerate(players) if player == state.dealer
        )
        return players[dealer_index + 1 :] + players[: dealer_index + 1]

    def _is_in_position(self, state: GameState) -> bool:
        players = self._preflop_action_order(state)
        return state.betting_round.to_act in players[2:]

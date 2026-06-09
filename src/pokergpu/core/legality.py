from __future__ import annotations

from .actions import Action, ActionType
from .betting import BettingRoundState, Chips, PlayerIndex
from .rules import max_raise_to, min_raise_to, player_stack


def amount_to_call_for_acting_player(state: BettingRoundState) -> Chips:
    return state.amount_to_call(state.to_act)


def can_check(state: BettingRoundState) -> bool:
    return amount_to_call_for_acting_player(state) == 0


def can_call(state: BettingRoundState) -> bool:
    to_call = amount_to_call_for_acting_player(state)
    if to_call <= 0:
        return False
    return player_stack(state, state.to_act) > 0


def can_fold(state: BettingRoundState) -> bool:
    return amount_to_call_for_acting_player(state) > 0


def can_bet(state: BettingRoundState) -> bool:
    return can_check(state) and player_stack(state, state.to_act) > 0


def can_raise(state: BettingRoundState) -> bool:
    if amount_to_call_for_acting_player(state) <= 0:
        return False
    return max_raise_to(state, state.to_act) > state.highest_bet


def is_legal_action(
    state: BettingRoundState,
    action: Action,
    *,
    player: PlayerIndex | None = None,
) -> bool:
    acting_player = state.to_act if player is None else player
    if acting_player != state.to_act:
        raise ValueError("legality checks currently require the acting player")

    if action.action_type is ActionType.FOLD:
        return can_fold(state)
    if action.action_type is ActionType.CHECK:
        return can_check(state)
    if action.action_type is ActionType.CALL:
        return can_call(state)
    if action.action_type is ActionType.BET:
        if action.amount is None or not can_bet(state):
            return False
        minimum_bet = state.blinds.big_blind
        maximum_bet = player_stack(state, state.to_act)
        return minimum_bet <= action.amount <= maximum_bet
    if action.action_type is ActionType.RAISE:
        if action.amount is None or not can_raise(state):
            return False
        minimum_raise_to = min_raise_to(state, state.to_act)
        maximum_raise_to = max_raise_to(state, state.to_act)
        return minimum_raise_to <= action.amount <= maximum_raise_to
    return False

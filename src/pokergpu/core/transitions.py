from __future__ import annotations

from dataclasses import dataclass

from .actions import Action, ActionType
from .betting import BettingRoundState, Chips, PlayerBet, PlayerIndex, PlayerStack
from .legality import is_legal_action
from .state import GameState, HandPhase, PlayerState


@dataclass(slots=True, frozen=True)
class AppliedTransition:
    previous_state: GameState
    action: Action
    next_state: GameState


def apply_action(state: GameState, action: Action) -> GameState:
    return apply_action_with_record(state, action).next_state


def apply_action_with_record(state: GameState, action: Action) -> AppliedTransition:
    if state.phase is not HandPhase.IN_PROGRESS:
        raise ValueError("cannot apply action to non-active hand")
    if not is_legal_action(state.betting_round, action):
        raise ValueError("illegal action for current state")

    player = state.betting_round.to_act
    if action.action_type is ActionType.FOLD:
        next_state = _apply_fold(state, player)
    elif action.action_type is ActionType.CHECK:
        next_state = _advance_without_bet_change(state)
    elif action.action_type is ActionType.CALL:
        next_state = _apply_contribution(
            state,
            player,
            Chips(state.betting_round.amount_to_call(player)),
        )
    elif action.action_type is ActionType.BET:
        if action.amount is None:
            raise ValueError("bet action requires amount")
        next_state = _apply_contribution(state, player, Chips(action.amount))
    elif action.action_type is ActionType.RAISE:
        if action.amount is None:
            raise ValueError("raise action requires amount")
        current = _player_bet(state.betting_round, player).committed
        next_state = _apply_contribution(state, player, Chips(action.amount - current))
    else:
        raise ValueError("unsupported action type")

    return AppliedTransition(previous_state=state, action=action, next_state=next_state)


def undo_transition(transition: AppliedTransition) -> GameState:
    return transition.previous_state


def _apply_fold(state: GameState, player: PlayerIndex) -> GameState:
    updated_players = tuple(
        PlayerState(
            player=entry.player,
            hole_cards=entry.hole_cards,
            folded=True if entry.player == player else entry.folded,
            all_in=entry.all_in,
        )
        if entry.player == player
        else entry
        for entry in state.players
    )
    updated_bets = tuple(
        PlayerBet(
            player=entry.player,
            committed=entry.committed,
            folded=True if entry.player == player else entry.folded,
            all_in=entry.all_in,
        )
        if entry.player == player
        else entry
        for entry in state.betting_round.bets
    )
    updated_round = BettingRoundState(
        pot=state.betting_round.pot,
        stacks=state.betting_round.stacks,
        bets=updated_bets,
        blinds=state.betting_round.blinds,
        to_act=player,
    )
    return _finalize_state(
        state,
        players=updated_players,
        betting_round=updated_round,
    )


def _advance_without_bet_change(state: GameState) -> GameState:
    return _finalize_state(
        state,
        players=state.players,
        betting_round=state.betting_round,
    )


def _apply_contribution(
    state: GameState,
    player: PlayerIndex,
    contribution: Chips,
) -> GameState:
    updated_stacks = []
    updated_bets = []
    is_all_in = False

    for stack in state.betting_round.stacks:
        if stack.player == player:
            remaining = Chips(stack.stack - contribution)
            is_all_in = remaining == 0
            updated_stacks.append(PlayerStack(player=stack.player, stack=remaining))
        else:
            updated_stacks.append(stack)

    for bet in state.betting_round.bets:
        if bet.player == player:
            updated_bets.append(
                PlayerBet(
                    player=bet.player,
                    committed=Chips(bet.committed + contribution),
                    folded=bet.folded,
                    all_in=is_all_in,
                )
            )
        else:
            updated_bets.append(bet)

    updated_players = tuple(
        PlayerState(
            player=entry.player,
            hole_cards=entry.hole_cards,
            folded=entry.folded,
            all_in=is_all_in if entry.player == player else entry.all_in,
        )
        if entry.player == player
        else entry
        for entry in state.players
    )
    updated_round = BettingRoundState(
        pot=state.betting_round.pot,
        stacks=tuple(updated_stacks),
        bets=tuple(updated_bets),
        blinds=state.betting_round.blinds,
        to_act=player,
    )
    return _finalize_state(
        state,
        players=updated_players,
        betting_round=updated_round,
    )


def _finalize_state(
    state: GameState,
    *,
    players: tuple[PlayerState, ...],
    betting_round: BettingRoundState,
) -> GameState:
    active_players = tuple(player for player in players if not player.folded)
    if len(active_players) <= 1:
        phase = HandPhase.TERMINAL
        to_act = betting_round.to_act
    else:
        eligible_players = tuple(
            player.player
            for player in players
            if not player.folded and not player.all_in
        )
        if len(eligible_players) <= 1:
            phase = HandPhase.SHOWDOWN
            to_act = betting_round.to_act
        else:
            phase = HandPhase.IN_PROGRESS
            to_act = _next_player_to_act(betting_round, eligible_players)

    updated_round = BettingRoundState(
        pot=betting_round.pot,
        stacks=betting_round.stacks,
        bets=betting_round.bets,
        blinds=betting_round.blinds,
        to_act=to_act,
    )
    return GameState(
        board=state.board,
        players=players,
        betting_round=updated_round,
        phase=phase,
        dealer=state.dealer,
    )


def _next_player_to_act(
    betting_round: BettingRoundState,
    eligible_players: tuple[PlayerIndex, ...],
) -> PlayerIndex:
    order = tuple(stack.player for stack in betting_round.stacks)
    current_index = order.index(betting_round.to_act)
    for offset in range(1, len(order) + 1):
        candidate = order[(current_index + offset) % len(order)]
        if candidate in eligible_players:
            return candidate
    return betting_round.to_act


def _player_bet(betting_round: BettingRoundState, player: PlayerIndex) -> PlayerBet:
    match = next(
        (entry for entry in betting_round.bets if entry.player == player),
        None,
    )
    if match is None:
        raise ValueError("unknown player")
    return match

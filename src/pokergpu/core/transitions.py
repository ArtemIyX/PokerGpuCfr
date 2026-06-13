from __future__ import annotations

from dataclasses import dataclass

from .actions import Action, ActionType
from .betting import BettingRoundState, Chips, PlayerBet, PlayerIndex, PlayerStack
from .board import Board, Street
from .cards import Card, make_deck
from .legality import is_legal_action
from .state import GameState, HandPhase, PlayerState
from .terminal import non_all_in_active_players


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
        stack = _player_stack(state.betting_round, player)
        contribution = Chips(min(state.betting_round.amount_to_call(player), stack))
        next_state = _apply_contribution(
            state,
            player,
            contribution,
            action=action,
        )
    elif action.action_type is ActionType.BET:
        if action.amount is None:
            raise ValueError("bet action requires amount")
        next_state = _apply_contribution(
            state,
            player,
            Chips(action.amount),
            action=action,
        )
    elif action.action_type is ActionType.RAISE:
        if action.amount is None:
            raise ValueError("raise action requires amount")
        current = _player_bet(state.betting_round, player).committed
        next_state = _apply_contribution(
            state,
            player,
            Chips(action.amount - current),
            action=action,
        )
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
        action=Action(ActionType.FOLD),
    )


def _advance_without_bet_change(state: GameState) -> GameState:
    return _finalize_state(
        state,
        players=state.players,
        betting_round=state.betting_round,
        action=Action(ActionType.CHECK),
    )


def _apply_contribution(
    state: GameState,
    player: PlayerIndex,
    contribution: Chips,
    *,
    action: Action,
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
        action=action,
    )


def _finalize_state(
    state: GameState,
    *,
    players: tuple[PlayerState, ...],
    betting_round: BettingRoundState,
    action: Action,
) -> GameState:
    active_player_states = tuple(player for player in players if not player.folded)
    if len(active_player_states) <= 1:
        phase = HandPhase.TERMINAL
        board = state.board
        round_state = betting_round
        to_act = betting_round.to_act
    else:
        candidate_state = GameState(
            board=state.board,
            players=players,
            betting_round=betting_round,
            phase=state.phase,
            dealer=state.dealer,
        )
        eligible_players = tuple(
            player.player for player in non_all_in_active_players(candidate_state)
        )
        if len(eligible_players) <= 1:
            phase = HandPhase.SHOWDOWN
            board = state.board
            round_state = betting_round
            to_act = betting_round.to_act
        elif _is_betting_round_complete(
            betting_round,
            eligible_players=eligible_players,
            action=action,
        ):
            round_state = _collect_bets_into_pot(
                betting_round,
                to_act=_first_player_to_act_for_new_round(
                    betting_round,
                    dealer=state.dealer,
                    eligible_players=eligible_players,
                ),
            )
            if state.current_street is Street.RIVER:
                phase = HandPhase.SHOWDOWN
                board = state.board
            else:
                phase = HandPhase.IN_PROGRESS
                board = _advance_board(state)
            to_act = round_state.to_act
        else:
            phase = HandPhase.IN_PROGRESS
            board = state.board
            round_state = betting_round
            to_act = _next_player_to_act(betting_round, eligible_players)

    updated_round = BettingRoundState(
        pot=round_state.pot,
        stacks=round_state.stacks,
        bets=round_state.bets,
        blinds=round_state.blinds,
        to_act=to_act,
    )
    return GameState(
        board=board,
        players=players,
        betting_round=updated_round,
        phase=phase,
        dealer=state.dealer,
    )


def _is_betting_round_complete(
    betting_round: BettingRoundState,
    *,
    eligible_players: tuple[PlayerIndex, ...],
    action: Action,
) -> bool:
    commitments = tuple(
        _player_bet(betting_round, player).committed for player in eligible_players
    )
    if len(set(commitments)) != 1:
        return False
    if action.action_type is not ActionType.CHECK:
        return False

    next_player = _next_player_to_act(betting_round, eligible_players)
    return next_player == eligible_players[0]


def _collect_bets_into_pot(
    betting_round: BettingRoundState,
    *,
    to_act: PlayerIndex,
) -> BettingRoundState:
    total_committed = sum(bet.committed for bet in betting_round.bets)
    reset_bets = tuple(
        PlayerBet(
            player=bet.player,
            committed=Chips(0),
            folded=bet.folded,
            all_in=bet.all_in,
        )
        for bet in betting_round.bets
    )
    return BettingRoundState(
        pot=betting_round.pot.add(Chips(total_committed)),
        stacks=betting_round.stacks,
        bets=reset_bets,
        blinds=betting_round.blinds,
        to_act=to_act,
    )


def _first_player_to_act_for_new_round(
    betting_round: BettingRoundState,
    *,
    dealer: PlayerIndex,
    eligible_players: tuple[PlayerIndex, ...],
) -> PlayerIndex:
    order = tuple(stack.player for stack in betting_round.stacks)
    dealer_index = order.index(dealer)
    for offset in range(1, len(order) + 1):
        candidate = order[(dealer_index + offset) % len(order)]
        if candidate in eligible_players:
            return candidate
    return betting_round.to_act


def _advance_board(state: GameState) -> Board:
    cards_to_add = {
        Street.PREFLOP: 3,
        Street.FLOP: 1,
        Street.TURN: 1,
    }.get(state.current_street, 0)
    if cards_to_add == 0:
        return state.board

    seen_cards: set[Card] = set(state.board.cards)
    for player in state.players:
        if player.hole_cards is not None:
            seen_cards.update(player.hole_cards)

    revealed_cards = list(state.board.cards)
    for card in make_deck():
        if card in seen_cards:
            continue
        revealed_cards.append(card)
        seen_cards.add(card)
        if len(revealed_cards) == len(state.board.cards) + cards_to_add:
            return Board(cards=tuple(revealed_cards))
    raise ValueError("not enough remaining cards to advance the board")


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


def _player_stack(betting_round: BettingRoundState, player: PlayerIndex) -> Chips:
    match = next(
        (entry.stack for entry in betting_round.stacks if entry.player == player),
        None,
    )
    if match is None:
        raise ValueError("unknown player")
    return match

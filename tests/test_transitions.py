import pytest

from pokergpu.core.actions import Action, ActionType
from pokergpu.core.betting import (
    BettingRoundState,
    BlindStructure,
    PlayerBet,
    PlayerIndex,
    PlayerStack,
    Pot,
    chips,
)
from pokergpu.core.board import Board
from pokergpu.core.cards import Card
from pokergpu.core.state import GameState, HandPhase, PlayerState
from pokergpu.core.transitions import (
    advance_board_to_next_street,
    advance_hand_to_next_street,
    apply_action,
    apply_action_with_record,
    undo_transition,
)


def _make_two_player_state(
    *,
    to_act: int,
    committed0: int,
    committed1: int,
) -> GameState:
    return GameState(
        board=Board(cards=()),
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(150)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(900)),
                PlayerStack(player=PlayerIndex(1), stack=chips(800)),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(committed0)),
                PlayerBet(player=PlayerIndex(1), committed=chips(committed1)),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(to_act),
        ),
        dealer=PlayerIndex(0),
    )


def test_apply_fold_transitions_to_terminal_when_one_player_remains() -> None:
    state = _make_two_player_state(to_act=1, committed0=100, committed1=0)

    next_state = apply_action(state, Action(ActionType.FOLD))

    assert next_state.phase is HandPhase.TERMINAL
    assert next_state.player_state(PlayerIndex(1)).folded


def test_apply_check_advances_turn() -> None:
    state = _make_two_player_state(to_act=0, committed0=0, committed1=0)

    next_state = apply_action(state, Action(ActionType.CHECK))

    assert next_state.phase is HandPhase.IN_PROGRESS
    assert next_state.betting_round.to_act == 1


def test_apply_call_updates_stack_and_committed_amount() -> None:
    state = _make_two_player_state(to_act=1, committed0=300, committed1=100)

    next_state = apply_action(state, Action(ActionType.CALL))

    assert next_state.betting_round.bets[1].committed == 300
    assert next_state.betting_round.stacks[1].stack == 600
    assert next_state.betting_round.to_act == 0


def test_apply_raise_updates_total_commitment() -> None:
    state = _make_two_player_state(to_act=1, committed0=300, committed1=100)

    next_state = apply_action(state, Action(ActionType.RAISE, amount=chips(500)))

    assert next_state.betting_round.bets[1].committed == 500
    assert next_state.betting_round.stacks[1].stack == 400
    assert next_state.betting_round.to_act == 0


def test_apply_action_with_record_can_undo() -> None:
    state = _make_two_player_state(to_act=1, committed0=300, committed1=100)

    transition = apply_action_with_record(state, Action(ActionType.CALL))

    assert undo_transition(transition) == state


def test_illegal_action_is_rejected() -> None:
    state = _make_two_player_state(to_act=0, committed0=0, committed1=0)

    with pytest.raises(ValueError):
        apply_action(state, Action(ActionType.CALL))


def test_legal_actions_produce_distinct_next_states() -> None:
    state = _make_two_player_state(to_act=1, committed0=300, committed1=100)

    fold_state = apply_action(state, Action(ActionType.FOLD))
    call_state = apply_action(state, Action(ActionType.CALL))
    raise_state = apply_action(state, Action(ActionType.RAISE, amount=chips(500)))

    assert fold_state != call_state
    assert call_state != raise_state
    assert fold_state != raise_state
    assert fold_state.phase is HandPhase.TERMINAL
    assert call_state.phase is HandPhase.IN_PROGRESS
    assert raise_state.phase is HandPhase.IN_PROGRESS
    assert fold_state.player_state(PlayerIndex(1)).folded
    assert call_state.betting_round.bets[1].committed == chips(300)
    assert raise_state.betting_round.bets[1].committed == chips(500)


def test_advance_board_to_next_street_updates_board_without_changing_round() -> None:
    state = _make_two_player_state(to_act=0, committed0=0, committed1=0)

    next_state = advance_board_to_next_street(state, next_board_cards=(Card.from_str("Ah"), Card.from_str("Kd"), Card.from_str("Tc")))

    assert next_state.board.street.value == "flop"
    assert next_state.betting_round == state.betting_round
    assert next_state.players == state.players


def test_advance_hand_to_next_street_resets_bets_and_passes_action() -> None:
    state = _make_two_player_state(to_act=0, committed0=100, committed1=100)

    next_state = advance_hand_to_next_street(
        state,
        next_board_cards=(Card.from_str("Ah"), Card.from_str("Kd"), Card.from_str("Tc")),
    )

    assert next_state.board.street.value == "flop"
    assert next_state.betting_round.bets[0].committed == chips(0)
    assert next_state.betting_round.bets[1].committed == chips(0)
    assert next_state.betting_round.to_act == PlayerIndex(1)

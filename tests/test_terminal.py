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
from pokergpu.core.state import GameState, HandPhase, PlayerState
from pokergpu.core.terminal import (
    active_players,
    is_hand_complete,
    is_showdown_state,
    is_terminal_state,
    non_all_in_active_players,
)


def _make_state(
    *,
    phase: HandPhase = HandPhase.IN_PROGRESS,
    folded1: bool = False,
    all_in1: bool = False,
) -> GameState:
    return GameState(
        board=Board(cards=()),
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(
                player=PlayerIndex(1),
                folded=folded1,
                all_in=all_in1,
            ),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(150)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(900)),
                PlayerStack(player=PlayerIndex(1), stack=chips(800)),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(100)),
                PlayerBet(
                    player=PlayerIndex(1),
                    committed=chips(100),
                    folded=folded1,
                    all_in=all_in1,
                ),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        ),
        phase=phase,
        dealer=PlayerIndex(0),
    )


def test_terminal_detection_when_only_one_player_is_active() -> None:
    state = _make_state(folded1=True)

    assert len(active_players(state)) == 1
    assert is_terminal_state(state)
    assert is_hand_complete(state)
    assert not is_showdown_state(state)


def test_showdown_detection_when_remaining_player_is_all_in() -> None:
    state = _make_state(all_in1=True)

    assert len(active_players(state)) == 2
    assert len(non_all_in_active_players(state)) == 1
    assert not is_terminal_state(state)
    assert is_showdown_state(state)
    assert is_hand_complete(state)


def test_phase_flags_are_respected() -> None:
    terminal_state = _make_state(phase=HandPhase.TERMINAL)
    showdown_state = _make_state(phase=HandPhase.SHOWDOWN)

    assert is_terminal_state(terminal_state)
    assert is_showdown_state(showdown_state)

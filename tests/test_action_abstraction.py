from pokergpu.abstraction.actions import BaselineActionAbstraction
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
from pokergpu.core.state import GameState, PlayerState


def test_baseline_abstraction_returns_check_and_bet_when_unopened() -> None:
    state = GameState(
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
                PlayerBet(player=PlayerIndex(0), committed=chips(0)),
                PlayerBet(player=PlayerIndex(1), committed=chips(0)),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        ),
        dealer=PlayerIndex(0),
    )

    actions = BaselineActionAbstraction().legal_actions(state)

    assert [action.action_type.value for action in actions] == ["check", "bet"]


def test_baseline_abstraction_returns_fold_call_raise_when_facing_bet() -> None:
    state = GameState(
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
                PlayerBet(player=PlayerIndex(0), committed=chips(300)),
                PlayerBet(player=PlayerIndex(1), committed=chips(100)),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(1),
        ),
        dealer=PlayerIndex(0),
    )

    actions = BaselineActionAbstraction().legal_actions(state)

    assert [action.action_type.value for action in actions] == [
        "fold",
        "call",
        "raise",
        "raise",
    ]

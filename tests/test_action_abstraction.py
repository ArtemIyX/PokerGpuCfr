from pokergpu.abstraction.actions import (
    BaselineActionAbstraction,
    make_compact_profile,
    make_default_profile,
)
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

    actions = BaselineActionAbstraction(
        profile=make_compact_profile()
    ).legal_actions(state)

    assert [action.action_type.value for action in actions] == ["check", "bet", "bet"]
    assert actions[1].amount == chips(150)
    assert actions[2].amount == chips(900)


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

    actions = BaselineActionAbstraction(
        profile=make_compact_profile()
    ).legal_actions(state)

    assert [action.action_type.value for action in actions] == [
        "fold",
        "call",
        "raise",
        "raise",
    ]
    assert actions[-1].amount == chips(900)


def test_profiles_can_change_generated_action_count() -> None:
    state = GameState(
        board=Board.from_str("AhKdTc"),
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

    default_actions = BaselineActionAbstraction(
        profile=make_default_profile()
    ).legal_actions(state)
    compact_actions = BaselineActionAbstraction(
        profile=make_compact_profile()
    ).legal_actions(state)

    assert len(default_actions) >= len(compact_actions)


def test_street_templates_change_betting_layout() -> None:
    preflop_state = GameState(
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
    flop_state = GameState(
        board=Board.from_str("AhKdTc"),
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(300)),
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

    preflop_actions = BaselineActionAbstraction(profile=make_default_profile()).legal_actions(
        preflop_state
    )
    flop_actions = BaselineActionAbstraction(profile=make_default_profile()).legal_actions(
        flop_state
    )

    assert [action.action_type.value for action in preflop_actions] == [
        "check",
        "bet",
        "bet",
        "bet",
    ]
    assert preflop_actions[-1].amount == chips(900)
    assert len(flop_actions) > len(preflop_actions)
    assert any(action.amount == chips(900) for action in flop_actions if action.action_type.value == "bet")

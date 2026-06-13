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
from pokergpu.core.board import Street
from pokergpu.core.state import GameState, PlayerState


def _make_preflop_state(
    *,
    to_act: int,
    dealer: int = 0,
    stacks: tuple[int, ...] = (1000, 1000, 1000, 1000, 1000, 1000),
    bets: tuple[int, ...] = (0, 0, 0, 0, 0, 0),
) -> GameState:
    players = tuple(PlayerState(player=PlayerIndex(index)) for index in range(6))
    return GameState(
        board=Board(cards=()),
        players=players,
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(sum(bets) + 150)),
            stacks=tuple(
                PlayerStack(player=PlayerIndex(index), stack=chips(stack))
                for index, stack in enumerate(stacks)
            ),
            bets=tuple(
                PlayerBet(player=PlayerIndex(index), committed=chips(committed))
                for index, committed in enumerate(bets)
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(to_act),
        ),
        dealer=PlayerIndex(dealer),
    )


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

    actions = BaselineActionAbstraction(
        profile=make_compact_profile()
    ).legal_actions(state)

    assert [action.action_type.value for action in actions] == [
        "fold",
        "call",
        "raise",
    ]


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


def test_legal_actions_are_stable_for_same_state() -> None:
    state = _make_preflop_state(to_act=1)
    abstraction = BaselineActionAbstraction(profile=make_default_profile())

    first = abstraction.legal_actions(state)
    second = abstraction.legal_actions(state)

    assert first == second


def test_position_groups_change_preflop_action_sets() -> None:
    abstraction = BaselineActionAbstraction(profile=make_default_profile())

    early = abstraction.legal_actions(_make_preflop_state(to_act=1))
    middle = abstraction.legal_actions(_make_preflop_state(to_act=3))
    late = abstraction.legal_actions(_make_preflop_state(to_act=5))
    blinds = abstraction.legal_actions(_make_preflop_state(to_act=0))

    assert early != middle
    assert middle != late
    assert late != blinds


def test_min_raise_is_respected() -> None:
    state = GameState(
        board=Board(cards=()),
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
                PlayerBet(player=PlayerIndex(0), committed=chips(300)),
                PlayerBet(player=PlayerIndex(1), committed=chips(100)),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(1),
        ),
        dealer=PlayerIndex(0),
    )

    actions = BaselineActionAbstraction(profile=make_default_profile()).legal_actions(
        state
    )

    raise_amounts = [
        int(action.amount)
        for action in actions
        if action.amount is not None
    ]
    assert min(raise_amounts) >= 500


def test_all_in_bet_is_retained_when_legal() -> None:
    state = _make_preflop_state(to_act=1, stacks=(250, 90, 1000, 1000, 1000, 1000))

    actions = BaselineActionAbstraction(profile=make_default_profile()).legal_actions(
        state
    )

    assert any(
        action.action_type.value == "bet" and int(action.amount or 0) == 90
        for action in actions
    )


def test_street_specific_action_sets_differ() -> None:
    from pokergpu.abstraction.actions import make_postflop_mvp_profile

    abstraction = BaselineActionAbstraction(profile=make_postflop_mvp_profile())

    preflop = abstraction.legal_actions(_make_preflop_state(to_act=1))
    flop = abstraction.legal_actions(
        GameState(
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
    )

    assert preflop != flop
    assert [action.action_type.value for action in flop] == ["check", "bet", "bet", "bet"]


def test_legal_actions_are_deterministically_ordered() -> None:
    state = GameState(
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
                PlayerBet(player=PlayerIndex(0), committed=chips(300)),
                PlayerBet(player=PlayerIndex(1), committed=chips(100)),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(1),
        ),
        dealer=PlayerIndex(0),
    )

    actions = BaselineActionAbstraction(profile=make_default_profile()).legal_actions(
        state
    )

    assert [action.action_type.value for action in actions] == [
        "fold",
        "call",
        "raise",
        "raise",
        "raise",
    ]
    assert actions == tuple(actions)
    assert actions == tuple(
        sorted(
            actions,
            key=lambda action: (
                {"fold": 0, "check": 1, "call": 2, "bet": 3, "raise": 4}[
                    action.action_type.value
                ],
                int(action.amount or 0),
            ),
        )
    )


def test_illegal_actions_are_filtered_out() -> None:
    state = GameState(
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
                PlayerBet(player=PlayerIndex(0), committed=chips(300)),
                PlayerBet(player=PlayerIndex(1), committed=chips(100)),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(1),
        ),
        dealer=PlayerIndex(0),
    )

    actions = BaselineActionAbstraction(profile=make_default_profile()).legal_actions(
        state
    )

    assert all(action.action_type.value != "check" for action in actions)
    assert all(action.action_type.value != "bet" for action in actions)


def test_all_in_is_not_forced_when_stack_is_not_binding() -> None:
    state = _make_preflop_state(to_act=1, stacks=(1000, 1000, 1000, 1000, 1000, 1000))
    actions = BaselineActionAbstraction(profile=make_default_profile()).legal_actions(
        state
    )

    assert not any(
        action.action_type.value == "bet" and int(action.amount or 0) == 1000
        for action in actions
    )


def test_postflop_mvp_profile_is_compact() -> None:
    from pokergpu.abstraction.actions import make_postflop_mvp_profile

    state = GameState(
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
    actions = BaselineActionAbstraction(
        profile=make_postflop_mvp_profile()
    ).legal_actions(state)

    assert [action.action_type.value for action in actions] == [
        "check",
        "bet",
        "bet",
        "bet",
    ]


def test_postflop_threeway_profile_is_compact() -> None:
    from pokergpu.abstraction.actions import make_postflop_threeway_profile

    profile = make_postflop_threeway_profile()

    assert profile.template_for_street(Street.FLOP).bet_sizes == (0.33, 0.5, 0.75)
    assert profile.template_for_street(Street.TURN).raise_to_multipliers == (1.0, 1.5)

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
from pokergpu.core.legality import (
    can_bet,
    can_call,
    can_check,
    can_fold,
    can_raise,
    is_legal_action,
)


def test_unopened_state_allows_check_and_bet() -> None:
    state = BettingRoundState(
        pot=Pot(amount=chips(150)),
        stacks=(
            PlayerStack(player=PlayerIndex(0), stack=chips(950)),
            PlayerStack(player=PlayerIndex(1), stack=chips(900)),
        ),
        bets=(
            PlayerBet(player=PlayerIndex(0), committed=chips(0)),
            PlayerBet(player=PlayerIndex(1), committed=chips(0)),
        ),
        blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
        to_act=PlayerIndex(0),
    )

    assert can_check(state)
    assert can_bet(state)
    assert not can_call(state)
    assert not can_fold(state)
    assert not can_raise(state)
    assert is_legal_action(state, Action(ActionType.CHECK))
    assert is_legal_action(state, Action(ActionType.BET, amount=chips(100)))
    assert not is_legal_action(state, Action(ActionType.BET, amount=chips(50)))


def test_facing_bet_allows_fold_call_raise() -> None:
    state = BettingRoundState(
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
    )

    assert not can_check(state)
    assert not can_bet(state)
    assert can_call(state)
    assert can_fold(state)
    assert can_raise(state)
    assert is_legal_action(state, Action(ActionType.FOLD))
    assert is_legal_action(state, Action(ActionType.CALL))
    assert is_legal_action(state, Action(ActionType.RAISE, amount=chips(500)))
    assert not is_legal_action(state, Action(ActionType.RAISE, amount=chips(400)))


def test_all_in_cap_is_legal_raise_bound() -> None:
    state = BettingRoundState(
        pot=Pot(amount=chips(300)),
        stacks=(
            PlayerStack(player=PlayerIndex(0), stack=chips(900)),
            PlayerStack(player=PlayerIndex(1), stack=chips(250)),
        ),
        bets=(
            PlayerBet(player=PlayerIndex(0), committed=chips(300)),
            PlayerBet(player=PlayerIndex(1), committed=chips(100)),
        ),
        blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
        to_act=PlayerIndex(1),
    )

    assert can_raise(state)
    assert is_legal_action(state, Action(ActionType.RAISE, amount=chips(350)))
    assert not is_legal_action(state, Action(ActionType.RAISE, amount=chips(351)))

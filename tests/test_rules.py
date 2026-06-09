import pytest

from pokergpu.core.betting import (
    BettingRoundState,
    BlindStructure,
    PlayerBet,
    PlayerIndex,
    PlayerStack,
    Pot,
    chips,
)
from pokergpu.core.rules import (
    max_raise_to,
    min_raise_increment,
    min_raise_to,
    raise_bounds,
    stack_after_call,
)


def test_min_raise_increment_uses_big_blind_when_unopened() -> None:
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

    assert min_raise_increment(state) == 100


def test_min_raise_to_uses_last_raise_size() -> None:
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

    assert min_raise_increment(state) == 200
    assert min_raise_to(state, PlayerIndex(1)) == 500


def test_raise_bounds_cap_at_all_in() -> None:
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

    assert max_raise_to(state, PlayerIndex(1)) == 350
    assert min_raise_to(state, PlayerIndex(1)) == 350
    assert raise_bounds(state, PlayerIndex(1)).is_forced_all_in_only


def test_stack_after_call_uses_amount_to_call() -> None:
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

    assert stack_after_call(state, PlayerIndex(1)) == 50


def test_unknown_player_is_rejected() -> None:
    state = BettingRoundState(
        pot=Pot(amount=chips(0)),
        stacks=(PlayerStack(player=PlayerIndex(0), stack=chips(1000)),),
        bets=(PlayerBet(player=PlayerIndex(0), committed=chips(0)),),
        blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
        to_act=PlayerIndex(0),
    )

    with pytest.raises(ValueError):
        min_raise_to(state, PlayerIndex(1))

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


def test_chips_rejects_negative_values() -> None:
    with pytest.raises(ValueError):
        chips(-1)


def test_blind_structure_validates_values() -> None:
    blinds = BlindStructure(
        small_blind=chips(50),
        big_blind=chips(100),
        ante=chips(10),
    )

    assert blinds.small_blind == 50
    assert blinds.big_blind == 100
    assert blinds.ante == 10


def test_pot_add_returns_new_pot() -> None:
    pot = Pot(amount=chips(100))

    new_pot = pot.add(chips(25))

    assert pot.amount == 100
    assert new_pot.amount == 125


def test_betting_round_state_computes_amount_to_call() -> None:
    state = BettingRoundState(
        pot=Pot(amount=chips(150)),
        stacks=(
            PlayerStack(player=PlayerIndex(0), stack=chips(900)),
            PlayerStack(player=PlayerIndex(1), stack=chips(800)),
        ),
        bets=(
            PlayerBet(player=PlayerIndex(0), committed=chips(100)),
            PlayerBet(player=PlayerIndex(1), committed=chips(40)),
        ),
        blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
        to_act=PlayerIndex(1),
    )

    assert state.player_count == 2
    assert state.highest_bet == 100
    assert state.amount_to_call(PlayerIndex(1)) == 60
    assert state.amount_to_call(PlayerIndex(0)) == 0


def test_betting_round_rejects_mismatched_players() -> None:
    with pytest.raises(ValueError):
        BettingRoundState(
            pot=Pot(amount=chips(0)),
            stacks=(PlayerStack(player=PlayerIndex(0), stack=chips(1000)),),
            bets=(PlayerBet(player=PlayerIndex(1), committed=chips(0)),),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        )

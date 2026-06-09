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
from pokergpu.core.board import Board
from pokergpu.core.cards import Card
from pokergpu.core.payouts import compute_payouts, total_pot
from pokergpu.core.state import GameState, HandPhase, PlayerState


def test_total_pot_includes_committed_bets() -> None:
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
                PlayerBet(player=PlayerIndex(0), committed=chips(100)),
                PlayerBet(player=PlayerIndex(1), committed=chips(100)),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        ),
        dealer=PlayerIndex(0),
    )

    assert total_pot(state) == 350


def test_compute_payouts_for_uncontested_terminal_pot() -> None:
    state = GameState(
        board=Board(cards=()),
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1), folded=True),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(150)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(900)),
                PlayerStack(player=PlayerIndex(1), stack=chips(800)),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(100)),
                PlayerBet(player=PlayerIndex(1), committed=chips(100), folded=True),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        ),
        phase=HandPhase.TERMINAL,
        dealer=PlayerIndex(0),
    )

    payouts = compute_payouts(state)

    assert payouts[0].amount == 350
    assert payouts[1].amount == 0


def test_compute_payouts_for_showdown_winner() -> None:
    state = GameState(
        board=Board.from_str("KhQhJh2c3d"),
        players=(
            PlayerState(
                player=PlayerIndex(0),
                hole_cards=(Card.from_str("Ah"), Card.from_str("Th")),
            ),
            PlayerState(
                player=PlayerIndex(1),
                hole_cards=(Card.from_str("Ad"), Card.from_str("As")),
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
                PlayerBet(player=PlayerIndex(1), committed=chips(100)),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        ),
        phase=HandPhase.SHOWDOWN,
        dealer=PlayerIndex(0),
    )

    payouts = compute_payouts(state)

    assert payouts[0].amount == 350
    assert payouts[1].amount == 0


def test_compute_payouts_rejects_side_pots_for_now() -> None:
    state = GameState(
        board=Board.from_str("KhQhJh2c3d"),
        players=(
            PlayerState(
                player=PlayerIndex(0),
                hole_cards=(Card.from_str("Ah"), Card.from_str("Th")),
            ),
            PlayerState(
                player=PlayerIndex(1),
                hole_cards=(Card.from_str("Ad"), Card.from_str("As")),
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
                PlayerBet(player=PlayerIndex(1), committed=chips(50), all_in=True),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        ),
        phase=HandPhase.SHOWDOWN,
        dealer=PlayerIndex(0),
    )

    with pytest.raises(NotImplementedError):
        compute_payouts(state)

from __future__ import annotations

import pytest

from pokergpu.cfr.solver import GameVariant
from pokergpu.cfr.solver import make_game_public_tree
from pokergpu.core.betting import BettingRoundState
from pokergpu.core.betting import BlindStructure
from pokergpu.core.betting import PlayerBet
from pokergpu.core.betting import PlayerIndex
from pokergpu.core.betting import PlayerStack
from pokergpu.core.betting import Pot
from pokergpu.core.betting import chips
from pokergpu.core.board import Board
from pokergpu.core.cards import Card
from pokergpu.core.state import GameState
from pokergpu.core.state import PlayerState


def _make_minimal_holdem_state() -> GameState:
    return GameState(
        board=Board(cards=()),
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(3)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(99)),
                PlayerStack(player=PlayerIndex(1), stack=chips(98)),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(1)),
                PlayerBet(player=PlayerIndex(1), committed=chips(2)),
            ),
            blinds=BlindStructure(small_blind=chips(1), big_blind=chips(2)),
            to_act=PlayerIndex(0),
        ),
        dealer=PlayerIndex(0),
    )


def test_minimal_holdem_public_state_is_valid() -> None:
    state = _make_minimal_holdem_state()

    assert state.player_count == 2
    assert state.board.is_preflop
    assert state.dealer == PlayerIndex(0)
    assert state.betting_round.to_act == PlayerIndex(0)


def test_invalid_holdem_public_state_rejects_duplicate_board_and_hole_cards() -> None:
    with pytest.raises(ValueError, match="duplicate card across board and hole cards"):
        GameState(
            board=Board.from_str("AhKdTc"),
            players=(
                PlayerState(
                    player=PlayerIndex(0),
                    hole_cards=(Card.from_str("Ah"), Card.from_str("Qs")),
                ),
                PlayerState(player=PlayerIndex(1)),
            ),
            betting_round=BettingRoundState(
                pot=Pot(amount=chips(300)),
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


def test_board_street_is_derived_from_board_cards() -> None:
    state = GameState(
        board=Board.from_str("AhKdTc"),
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(3)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(99)),
                PlayerStack(player=PlayerIndex(1), stack=chips(98)),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(1)),
                PlayerBet(player=PlayerIndex(1), committed=chips(2)),
            ),
            blinds=BlindStructure(small_blind=chips(1), big_blind=chips(2)),
            to_act=PlayerIndex(0),
        ),
        dealer=PlayerIndex(0),
    )

    assert state.board.street.value == "flop"
    assert state.current_street == state.board.street


def test_dealer_must_reference_a_valid_player() -> None:
    with pytest.raises(ValueError, match="dealer must reference a valid player"):
        GameState(
            board=Board(cards=()),
            players=(
                PlayerState(player=PlayerIndex(0)),
                PlayerState(player=PlayerIndex(1)),
            ),
            betting_round=BettingRoundState(
                pot=Pot(amount=chips(3)),
                stacks=(
                    PlayerStack(player=PlayerIndex(0), stack=chips(99)),
                    PlayerStack(player=PlayerIndex(1), stack=chips(98)),
                ),
                bets=(
                    PlayerBet(player=PlayerIndex(0), committed=chips(1)),
                    PlayerBet(player=PlayerIndex(1), committed=chips(2)),
                ),
                blinds=BlindStructure(small_blind=chips(1), big_blind=chips(2)),
                to_act=PlayerIndex(0),
            ),
            dealer=PlayerIndex(2),
        )


def test_player_stack_and_bet_alignment_is_enforced() -> None:
    with pytest.raises(ValueError, match="player states must match betting round players"):
        GameState(
            board=Board(cards=()),
            players=(
                PlayerState(player=PlayerIndex(0)),
                PlayerState(player=PlayerIndex(1)),
            ),
            betting_round=BettingRoundState(
                pot=Pot(amount=chips(3)),
                stacks=(
                    PlayerStack(player=PlayerIndex(0), stack=chips(99)),
                    PlayerStack(player=PlayerIndex(2), stack=chips(98)),
                ),
                bets=(
                    PlayerBet(player=PlayerIndex(0), committed=chips(1)),
                    PlayerBet(player=PlayerIndex(2), committed=chips(2)),
                ),
                blinds=BlindStructure(small_blind=chips(1), big_blind=chips(2)),
                to_act=PlayerIndex(0),
            ),
            dealer=PlayerIndex(0),
        )


def test_holdem_tree_factory_still_supports_toy_variants() -> None:
    assert make_game_public_tree(GameVariant.KUHN).node_count > 0
    assert make_game_public_tree(GameVariant.LEDUC).node_count > 0

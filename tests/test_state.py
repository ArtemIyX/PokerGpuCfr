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
from pokergpu.core.state import GameState, HandPhase, PlayerState


def test_game_state_exposes_active_and_folded_players() -> None:
    state = GameState(
        board=Board.from_str("AhKdTc"),
        players=(
            PlayerState(
                player=PlayerIndex(0),
                hole_cards=(Card.from_str("Qs"), Card.from_str("Jh")),
            ),
            PlayerState(player=PlayerIndex(1), folded=True),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(300)),
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
        phase=HandPhase.IN_PROGRESS,
        dealer=PlayerIndex(1),
    )

    assert state.current_street == state.board.street
    assert state.player_count == 2
    assert len(state.active_players) == 1
    assert len(state.folded_players) == 1
    assert len(state.showdown_eligible_players) == 1


def test_game_state_rejects_duplicate_cards_between_board_and_hole_cards() -> None:
    with pytest.raises(ValueError):
        GameState(
            board=Board.from_str("AhKdTc"),
            players=(
                PlayerState(
                    player=PlayerIndex(0),
                    hole_cards=(Card.from_str("Ah"), Card.from_str("Qs")),
                ),
            ),
            betting_round=BettingRoundState(
                pot=Pot(amount=chips(150)),
                stacks=(PlayerStack(player=PlayerIndex(0), stack=chips(1000)),),
                bets=(PlayerBet(player=PlayerIndex(0), committed=chips(0)),),
                blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
                to_act=PlayerIndex(0),
            ),
            dealer=PlayerIndex(0),
        )


def test_game_state_rejects_player_mismatch_with_betting_round() -> None:
    with pytest.raises(ValueError):
        GameState(
            board=Board(cards=()),
            players=(PlayerState(player=PlayerIndex(0)),),
            betting_round=BettingRoundState(
                pot=Pot(amount=chips(0)),
                stacks=(PlayerStack(player=PlayerIndex(1), stack=chips(1000)),),
                bets=(PlayerBet(player=PlayerIndex(1), committed=chips(0)),),
                blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
                to_act=PlayerIndex(1),
            ),
            dealer=PlayerIndex(0),
        )


def test_player_state_rejects_duplicate_hole_cards() -> None:
    card = Card.from_str("Ah")

    with pytest.raises(ValueError):
        PlayerState(
            player=PlayerIndex(0),
            hole_cards=(card, card),
        )

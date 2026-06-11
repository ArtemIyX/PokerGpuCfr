import numpy as np

from pokergpu.abstraction.hands import PlayerRangeVectors, RangeVector, private_hand_index
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
from pokergpu.core.cards import Card, Rank, Suit
from pokergpu.core.state import GameState, PlayerState
from pokergpu.value_network.equity import build_postflop_equity_label


def test_build_postflop_equity_label_returns_exact_zero_sum_river_value() -> None:
    board = Board.from_str("7c9hJsQdKh")
    hero = (Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS))
    villain = (Card(Rank.TWO, Suit.DIAMONDS), Card(Rank.THREE, Suit.DIAMONDS))
    state = GameState(
        board=board,
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(300)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(900)),
                PlayerStack(player=PlayerIndex(1), stack=chips(900)),
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
    p0 = np.zeros(1326, dtype=np.float32)
    p1 = np.zeros(1326, dtype=np.float32)
    p0[private_hand_index(*hero)] = 1.0
    p1[private_hand_index(*villain)] = 1.0
    ranges = PlayerRangeVectors.from_values(
        (RangeVector.from_values(p0), RangeVector.from_values(p1))
    )

    label = build_postflop_equity_label(state, ranges)

    assert label.values.shape == (1, 2)
    assert np.isclose(float(label.values[0, 0]), 300.0)
    assert np.isclose(float(label.values[0, 1]), -300.0)
    assert np.isclose(float(label.values[0, 0] + label.values[0, 1]), 0.0)

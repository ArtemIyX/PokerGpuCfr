from __future__ import annotations

import numpy as np

from pokergpu.core.betting import BettingRoundState, BlindStructure, PlayerBet, PlayerIndex, PlayerStack, Pot, chips
from pokergpu.core.board import Board
from pokergpu.core.state import GameState, PlayerState
from pokergpu.eval.cpu_stub import CpuStubLeafEvaluator
from pokergpu.eval.types import LeafFeatureBatch


def test_cpu_stub_leaf_evaluator_returns_nonzero_for_nonterminal_leaf() -> None:
    state = GameState(
        board=Board.from_str("AhKdTc"),
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(300)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(1700)),
                PlayerStack(player=PlayerIndex(1), stack=chips(1700)),
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
    batch = LeafFeatureBatch(
        node_indices=(0,),
        node_states=(state,),
        terminal_payoff=np.asarray([np.nan], dtype=np.float32),
        player_to_act=np.asarray([0], dtype=np.int32),
        street=np.asarray([3], dtype=np.int32),
        pot=np.asarray([300.0], dtype=np.float32),
        stack_p0=np.asarray([1700.0], dtype=np.float32),
        stack_p1=np.asarray([1700.0], dtype=np.float32),
        board_size=np.asarray([3], dtype=np.int32),
        reach_p0=np.asarray([1.0], dtype=np.float32),
        reach_p1=np.asarray([1.0], dtype=np.float32),
        reach_p2=np.asarray([0.0], dtype=np.float32),
        is_terminal=np.asarray([False], dtype=np.bool_),
        is_frontier=np.asarray([True], dtype=np.bool_),
        infoset_id=np.asarray([0], dtype=np.int32),
    )

    result = CpuStubLeafEvaluator().evaluate(batch)

    assert result.ev_player0.shape == (1,)
    assert result.ev_player1.shape == (1,)
    assert result.ev_player0[0] != 0.0
    assert result.ev_player1[0] != 0.0

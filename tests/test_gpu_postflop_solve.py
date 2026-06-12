from __future__ import annotations

import numpy as np
import pytest
import torch

from pokergpu.abstraction.hands import RangeVector
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
from pokergpu.runtime import PostflopResolveSpec
from pokergpu.runtime.gpu_postflop import resolve_postflop_gpu


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)


def _make_state() -> GameState:
    return GameState(
        board=Board.from_str("KhQs5hJs"),
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(200)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(2000)),
                PlayerStack(player=PlayerIndex(1), stack=chips(2000)),
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


def test_gpu_postflop_solver_returns_finite_root_ev() -> None:
    result = resolve_postflop_gpu(
        PostflopResolveSpec(
            state=_make_state(),
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            time_budget_sec=0.0,
            max_depth=6,
            max_nodes=2048,
        )
    )

    assert result.iterations == 1
    assert result.node_count > 0
    assert result.leaf_count > 0
    assert len(result.root_actions) >= 2
    assert np.isfinite(result.root_ev_player0)
    assert np.isfinite(result.root_ev_player1)
    assert np.isclose(float(result.root_strategy.sum()), 1.0)


def test_gpu_postflop_solver_is_deterministic_for_fixed_seed_state() -> None:
    spec = PostflopResolveSpec(
        state=_make_state(),
        range_p0=RangeVector.uniform(),
        range_p1=RangeVector.uniform(),
        time_budget_sec=0.0,
        max_depth=6,
        max_nodes=2048,
    )
    result_a = resolve_postflop_gpu(spec)
    result_b = resolve_postflop_gpu(spec)

    assert np.isclose(result_a.root_ev_player0, result_b.root_ev_player0)
    assert np.isclose(result_a.root_ev_player1, result_b.root_ev_player1)
    assert np.allclose(result_a.root_strategy, result_b.root_strategy)

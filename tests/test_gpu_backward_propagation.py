from __future__ import annotations

import numpy as np
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
from pokergpu.runtime.gpu_postflop import _prepare_gpu_solve
from pokergpu.runtime.gpu_passes import regret_matching_table_inplace
from pokergpu.runtime.gpu_passes import _COMPACT_ITERATION_CORE
from pokergpu.runtime.postflop import PostflopResolveSpec


def test_compact_iteration_propagates_leaf_value_to_root() -> None:
    packed = _prepare_gpu_solve(
        PostflopResolveSpec(
            state=_make_state(),
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            time_budget_sec=0.0,
            iterations=1,
            solver_version="test",
            max_depth=1,
            max_nodes=64,
        )
    )
    state = packed.gpu_state
    assert state is not None
    regret_matching_table_inplace(
        state.strategy_table,
        state.regrets,
        state.action_infoset_index,
        state.action_slot_index,
        state.action_counts,
    )

    out_p0 = state.backward_p0
    out_p1 = state.backward_p1
    out_p0.zero_()
    out_p1.zero_()
    leaf_index = int(packed.plan.root_child_nodes[0].item())
    out_p0[leaf_index] = 12.5
    out_p1[leaf_index] = -12.5

    _COMPACT_ITERATION_CORE(
        state,
        state.strategy_table,
        node_range_p0=state.node_range_p0,
        node_range_p1=state.node_range_p1,
        out_p0=out_p0,
        out_p1=out_p1,
        regrets=state.regrets,
        strategy_sums=state.strategy_sums,
        node_values_p0=out_p0,
        node_values_p1=out_p1,
    )

    assert float(out_p0[0].item()) != 0.0


def test_root_child_appears_in_backward_schedule() -> None:
    packed = _prepare_gpu_solve(
        PostflopResolveSpec(
            state=_make_state(),
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            time_budget_sec=0.0,
            iterations=1,
            solver_version="test",
            max_depth=1,
            max_nodes=64,
        )
    )
    state = packed.gpu_state
    assert state is not None
    root_child = int(packed.plan.root_child_nodes[0].item())

    hits = []
    for edge_dst in state.compact_backward_edge_dst:
        hits.append(bool(torch.any(edge_dst == root_child).item()))

    assert any(hits), "root child must appear in at least one backward edge block"


def _make_state() -> GameState:
    return GameState(
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

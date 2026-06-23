from __future__ import annotations

import numpy as np
import pytest

from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.cfr.stage2 import aggregate_prob_sum
from pokergpu.cfr.stage3 import compute_opponent_reach
from pokergpu.cfr.stage4 import ShowdownEquityBatchInput, ShowdownEquityResult
from pokergpu.cfr.stage6 import BackwardCFVInput, backward_cfv
from pokergpu.cfr.stage7 import apply_dense_backward_cfv_update
from pokergpu.cfr.stage7 import (
    DenseCfrState,
    regret_matching,
    update_average_strategy,
    update_regret,
)
from pokergpu.cfr.solver import build_dense_infoset_table
from pokergpu.cfr.solver import CfrVariant
from pokergpu.cfr.solver import GameVariant
from pokergpu.cfr.solver import SolverStageRequest
from pokergpu.cfr.solver import make_game_public_tree
from pokergpu.cfr.solver import run_solver_stage
from pokergpu.cfr.solver import TimingSpec
from pokergpu.core.betting import Chips
from pokergpu.core.board import Board
from pokergpu.tree.public_tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


def test_regret_matching_uses_positive_regrets() -> None:
    assert regret_matching((-1.0, 3.0, 1.0)) == (0.0, 0.75, 0.25)


def test_regret_matching_falls_back_to_uniform() -> None:
    assert regret_matching((0.0, -2.0)) == (0.5, 0.5)


def test_update_regret_adds_counterfactual_differences() -> None:
    assert update_regret((1.0, -2.0), (4.0, 1.0), 2.0) == (3.0, -3.0)


def test_update_average_strategy_accumulates_reach_weight() -> None:
    assert update_average_strategy((1.0, 2.0), (0.25, 0.75), 4.0) == (2.0, 5.0)


def test_dense_cfr_state_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        DenseCfrState(regret_sums=((0.0, 1.0),), strategy_sums=((0.0,),))


def test_apply_dense_backward_cfv_update_uses_stage6_action_values() -> None:
    tree = PublicTree(
        node_types=(
            NodeType.PLAYER0,
            NodeType.TERMINAL,
            NodeType.TERMINAL,
        ),
        first_child=(0, 2, 2),
        child_count=(2, 0, 0),
        children=(
            ChildLink(child=NodeId(1)),
            ChildLink(child=NodeId(2)),
        ),
        infoset_ids=(InfosetId(0), None, None),
        terminal_payoffs=(None, Chips(1), Chips(3)),
    )
    forward = ForwardProfileResult(
        node_reach=(1.0, 0.5, 0.5),
        infoset_reach=(1.0,),
        action_reach=((0.5, 0.5), (), ()),
    )
    aggregate = aggregate_prob_sum(tree, forward)
    opponent = compute_opponent_reach(tree, aggregate)
    backward = backward_cfv(
        BackwardCFVInput(
            tree=tree,
            forward=forward,
            aggregate=aggregate,
            opponent_reach=opponent,
            showdown=ShowdownEquityResult(
                node_showdown_equity=(0.0, 1.0, 3.0),
                node_showdown_equity_bb=(0.0, 1.0, 3.0),
                input_rows=ShowdownEquityBatchInput(rows=()),
                output_rows=(),
            ),
            leaf_values=np.asarray((), dtype=np.float64),
        )
    )
    table = build_dense_infoset_table(tree)
    state = DenseCfrState(regret_sums=((0.0, 0.0),), strategy_sums=((0.0, 0.0),))

    result = apply_dense_backward_cfv_update(state, backward, infoset_table=table)

    assert result.regret_sums[0] == (-1.0, 1.0)
    assert result.strategy_sums[0] == (0.5, 0.5)


def test_apply_dense_backward_cfv_update_threaded_matches_serial() -> None:
    tree = PublicTree(
        node_types=(
            NodeType.PLAYER0,
            NodeType.TERMINAL,
            NodeType.TERMINAL,
        ),
        first_child=(0, 2, 2),
        child_count=(2, 0, 0),
        children=(
            ChildLink(child=NodeId(1)),
            ChildLink(child=NodeId(2)),
        ),
        infoset_ids=(InfosetId(0), None, None),
        terminal_payoffs=(None, Chips(1), Chips(3)),
    )
    forward = ForwardProfileResult(
        node_reach=(1.0, 0.5, 0.5),
        infoset_reach=(1.0,),
        action_reach=((0.5, 0.5), (), ()),
    )
    aggregate = aggregate_prob_sum(tree, forward)
    opponent = compute_opponent_reach(tree, aggregate)
    backward = backward_cfv(
        BackwardCFVInput(
            tree=tree,
            forward=forward,
            aggregate=aggregate,
            opponent_reach=opponent,
            showdown=ShowdownEquityResult(
                node_showdown_equity=(0.0, 1.0, 3.0),
                node_showdown_equity_bb=(0.0, 1.0, 3.0),
                input_rows=ShowdownEquityBatchInput(rows=()),
                output_rows=(),
            ),
            leaf_values=np.asarray((), dtype=np.float64),
        )
    )
    table = build_dense_infoset_table(tree)
    state = DenseCfrState(regret_sums=((0.0, 0.0),), strategy_sums=((0.0, 0.0),))

    serial = apply_dense_backward_cfv_update(state, backward, infoset_table=table, max_workers=1)
    threaded = apply_dense_backward_cfv_update(state, backward, infoset_table=table, max_workers=4)

    assert threaded == serial


def test_apply_dense_backward_cfv_update_accumulates_over_iterations() -> None:
    tree = PublicTree(
        node_types=(
            NodeType.PLAYER0,
            NodeType.TERMINAL,
            NodeType.TERMINAL,
        ),
        first_child=(0, 2, 2),
        child_count=(2, 0, 0),
        children=(
            ChildLink(child=NodeId(1)),
            ChildLink(child=NodeId(2)),
        ),
        infoset_ids=(InfosetId(0), None, None),
        terminal_payoffs=(None, Chips(1), Chips(3)),
    )
    forward = ForwardProfileResult(
        node_reach=(1.0, 0.5, 0.5),
        infoset_reach=(1.0,),
        action_reach=((0.5, 0.5), (), ()),
    )
    aggregate = aggregate_prob_sum(tree, forward)
    opponent = compute_opponent_reach(tree, aggregate)
    backward = backward_cfv(
        BackwardCFVInput(
            tree=tree,
            forward=forward,
            aggregate=aggregate,
            opponent_reach=opponent,
            showdown=ShowdownEquityResult(
                node_showdown_equity=(0.0, 1.0, 3.0),
                node_showdown_equity_bb=(0.0, 1.0, 3.0),
                input_rows=ShowdownEquityBatchInput(rows=()),
                output_rows=(),
            ),
            leaf_values=np.asarray((), dtype=np.float64),
        )
    )
    table = build_dense_infoset_table(tree)
    state = DenseCfrState(regret_sums=((0.0, 0.0),), strategy_sums=((0.0, 0.0),))

    first = apply_dense_backward_cfv_update(state, backward, infoset_table=table)
    second = apply_dense_backward_cfv_update(first, backward, infoset_table=table)

    assert first.regret_sums[0] == (-1.0, 1.0)
    assert second.regret_sums[0] == (-2.0, 2.0)
    assert first.strategy_sums[0] == (0.5, 0.5)
    assert second.strategy_sums[0] == (0.5, 1.5)


@pytest.mark.parametrize(
    "variant",
    [CfrVariant.CFR, CfrVariant.CFR_PLUS, CfrVariant.DCFR, CfrVariant.PREDICTIVE_CFR_PLUS],
)
def test_holdem_tree_accepts_supported_cfr_variants(variant: CfrVariant) -> None:
    tree = make_game_public_tree(GameVariant.HOLDEM_HU, depth_limit=1)
    table = build_dense_infoset_table(tree)
    state = DenseCfrState(
        regret_sums=tuple(tuple(0.0 for _ in range(table.action_counts[index])) for index in range(table.infoset_count)),
        strategy_sums=tuple(tuple(0.0 for _ in range(table.action_counts[index])) for index in range(table.infoset_count)),
    )
    request = SolverStageRequest(
        game=GameVariant.HOLDEM_HU,
        cfr_variant=variant,
        depth_limit=1,
        timing=TimingSpec(measure=False),
    )

    result = run_solver_stage(
        request,
        tree=tree,
        dense_state=state,
        board=Board(cards=()),
    )

    assert result.final_state is not None

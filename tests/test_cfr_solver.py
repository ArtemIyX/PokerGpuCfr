from __future__ import annotations

import pytest
import numpy as np

from pokergpu.core.betting import Chips
from pokergpu.cfr.solver import (
    DenseCfrState,
    evaluate_backward_cfv,
    evaluate_frontier_leaf_values,
    evaluate_leaf_node_values,
    SolverState,
    aggregate_root_action_values,
    run_dense_backward_cfv_iteration,
    run_dense_backward_cfv_iterations,
    make_toy_pipeline_tree,
    make_toy_public_tree,
    propagate_reach,
    propagate_opponent_reach,
    run_tree_backward_cfv_iteration,
    run_tree_root_iteration,
    evaluate_showdown_node_values,
)
from pokergpu.cfr.gpu_leaf_backend import GpuLeafBackend
from pokergpu.cfr.leaf_backend_factory import create_heuristic_leaf_backend
from pokergpu.cfr.leaf_eval import LEAF_EVAL_OUTPUT_WIDTH
from pokergpu.cfr.leaf_eval import LeafEvalBatchInput, LeafEvalBatchOutput
from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.cfr.stage2 import aggregate_prob_sum
from pokergpu.core.board import Board
from pokergpu.tree.public_tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


def test_make_toy_public_tree_is_small_and_solved_like() -> None:
    tree = make_toy_public_tree()

    assert tree.node_count == 3
    assert tree.node_types[0] is NodeType.PLAYER0
    assert tree.child_count[0] == 2


def test_make_toy_pipeline_tree_includes_a_leaf() -> None:
    tree = make_toy_pipeline_tree()

    assert tree.node_count == 4
    assert NodeType.LEAF in tree.node_types
    assert tree.child_count[2] == 0


def test_toy_solver_iteration_runs_on_tiny_tree_shape() -> None:
    tree = make_toy_public_tree()

    state = SolverState(regret_sums=(0.0, 0.0), strategy_sums=(0.0, 0.0))
    result = run_tree_root_iteration(tree, state)

    assert result.strategy == (0.5, 0.5)
    assert result.state.strategy_sums == (0.5, 0.5)


def test_toy_solver_iteration_uses_descendant_terminal_values() -> None:
    tree = PublicTree(
        node_types=(
            NodeType.PLAYER0,
            NodeType.PLAYER1,
            NodeType.TERMINAL,
            NodeType.TERMINAL,
            NodeType.TERMINAL,
        ),
        first_child=(0, 2, 4, 4, 4),
        child_count=(2, 2, 0, 0, 0),
        children=(
            ChildLink(child=NodeId(1)),
            ChildLink(child=NodeId(2)),
            ChildLink(child=NodeId(3)),
            ChildLink(child=NodeId(4)),
        ),
        infoset_ids=(InfosetId(0), InfosetId(1), None, None, None),
        terminal_payoffs=(None, None, Chips(1), Chips(3), Chips(-1)),
    )

    state = SolverState(regret_sums=(0.0, 0.0), strategy_sums=(0.0, 0.0))
    result = run_tree_root_iteration(tree, state)

    assert result.action_values == (1.0, 1.0)


def test_propagate_reach_tracks_node_and_infoset_reach() -> None:
    tree = make_toy_public_tree()

    result = propagate_reach(tree)

    assert result.node_reach == (1.0, 0.5, 0.5)
    assert result.infoset_reach == (1.0,)
    assert result.action_reach[0] == (0.5, 0.5)


def test_aggregate_root_action_values_reads_terminal_payoffs() -> None:
    tree = make_toy_public_tree()

    assert aggregate_root_action_values(tree) == (1.0, -1.0)


def test_aggregate_root_action_values_threaded_matches_serial() -> None:
    tree = make_toy_public_tree()

    assert aggregate_root_action_values(tree, max_workers=2) == aggregate_root_action_values(tree)


def test_run_tree_root_iteration_threaded_matches_serial() -> None:
    tree = make_toy_public_tree()
    state = SolverState(regret_sums=(0.0, 0.0), strategy_sums=(0.0, 0.0))

    serial = run_tree_root_iteration(tree, state)
    threaded = run_tree_root_iteration(tree, state, max_workers=2)

    assert threaded == serial


def test_propagate_opponent_reach_matches_stage3_result() -> None:
    tree = PublicTree(
        node_types=(
            NodeType.PLAYER0,
            NodeType.PLAYER1,
            NodeType.TERMINAL,
            NodeType.TERMINAL,
        ),
        first_child=(0, 1, 0, 0),
        child_count=(1, 2, 0, 0),
        children=(
            ChildLink(child=NodeId(1)),
            ChildLink(child=NodeId(2)),
            ChildLink(child=NodeId(3)),
        ),
        infoset_ids=(InfosetId(0), InfosetId(0), None, None),
        terminal_payoffs=(None, None, Chips(0), Chips(0)),
    )
    forward = ForwardProfileResult(
        node_reach=(1.0, 0.4, 0.0, 0.0),
        infoset_reach=(1.4,),
        action_reach=((0.4,), (0.25, 0.75), (), ()),
    )
    aggregate = aggregate_prob_sum(tree, forward)

    result = propagate_opponent_reach(tree, aggregate)

    assert result.infoset_opponent_reach == (1.4,)
    assert result.node_opponent_share == pytest.approx((5 / 7, 2 / 7, 0.0, 0.0))


def test_evaluate_showdown_node_values_runs_stage3_then_stage4() -> None:
    tree = PublicTree(
        node_types=(NodeType.LEAF,),
        first_child=(0,),
        child_count=(0,),
        children=(),
        infoset_ids=(None,),
        terminal_payoffs=(None,),
    )
    forward = ForwardProfileResult(node_reach=(1.0,), infoset_reach=(), action_reach=((),))

    result = evaluate_showdown_node_values(tree, forward, board=Board.from_str("AhKdTc9s2c"))

    assert len(result.node_showdown_equity) == 1
    assert len(result.output_rows) == 1
    assert result.output_rows[0].node_id == 0


def test_evaluate_leaf_node_values_uses_leaf_batch_contract() -> None:
    class _Kernel:
        def __call__(self, batch: LeafEvalBatchInput) -> LeafEvalBatchOutput:
            values = np.full((len(batch.node_ids), LEAF_EVAL_OUTPUT_WIDTH), 0.25, dtype=np.float32)
            return LeafEvalBatchOutput(node_ids=batch.node_ids, values=values)

    tree = PublicTree(
        node_types=(NodeType.LEAF,),
        first_child=(0,),
        child_count=(0,),
        children=(),
        infoset_ids=(None,),
        terminal_payoffs=(None,),
    )
    forward = ForwardProfileResult(node_reach=(1.0,), infoset_reach=(), action_reach=((),))

    result = evaluate_leaf_node_values(tree, forward, backend=GpuLeafBackend(kernel=_Kernel()))

    assert result.node_ids == (0,)
    assert result.node_values == (0.25,)


def test_evaluate_frontier_leaf_values_defaults_to_heuristic_backend() -> None:
    tree = PublicTree(
        node_types=(NodeType.LEAF,),
        first_child=(0,),
        child_count=(0,),
        children=(),
        infoset_ids=(None,),
        terminal_payoffs=(None,),
    )
    forward = ForwardProfileResult(node_reach=(1.0,), infoset_reach=(), action_reach=((),))

    default_result = evaluate_frontier_leaf_values(tree, forward)
    heuristic_result = evaluate_frontier_leaf_values(
        tree,
        forward,
        backend=create_heuristic_leaf_backend(),
    )

    assert default_result == heuristic_result


def test_evaluate_backward_cfv_runs_on_toy_pipeline_tree() -> None:
    class _Kernel:
        def __call__(self, batch: LeafEvalBatchInput) -> LeafEvalBatchOutput:
            values = np.full((len(batch.node_ids), LEAF_EVAL_OUTPUT_WIDTH), 0.5, dtype=np.float32)
            return LeafEvalBatchOutput(node_ids=batch.node_ids, values=values)

    tree = make_toy_pipeline_tree()
    forward = ForwardProfileResult(
        node_reach=(1.0, 0.5, 0.5, 0.5),
        infoset_reach=(1.0, 0.5),
        action_reach=((0.5, 0.5), (1.0,), (), ()),
    )

    result = evaluate_backward_cfv(tree, forward, backend=GpuLeafBackend(kernel=_Kernel()))

    assert result.node_values.shape == (tree.node_count,)
    assert result.infoset_values.shape == (2,)
    assert result.node_values[2] == pytest.approx(0.5)
    assert result.node_values[3] == pytest.approx(2.0)


def test_run_tree_backward_cfv_iteration_matches_solver_wrapper() -> None:
    class _Kernel:
        def __call__(self, batch: LeafEvalBatchInput) -> LeafEvalBatchOutput:
            values = np.full((len(batch.node_ids), LEAF_EVAL_OUTPUT_WIDTH), 0.5, dtype=np.float32)
            return LeafEvalBatchOutput(node_ids=batch.node_ids, values=values)

    tree = make_toy_pipeline_tree()
    result = run_tree_backward_cfv_iteration(
        tree,
        backend=GpuLeafBackend(kernel=_Kernel()),
        infoset_strategies={
            InfosetId(0): (0.25, 0.75),
            InfosetId(1): (1.0,),
        },
    )

    assert result.node_values.shape == (tree.node_count,)
    assert result.infoset_values.shape == (2,)
    assert result.node_values[0] == pytest.approx(1.625)
    assert result.action_values[0] == (0.5, 2.0)


def test_run_dense_backward_cfv_iteration_updates_dense_state() -> None:
    class _Kernel:
        def __call__(self, batch: LeafEvalBatchInput) -> LeafEvalBatchOutput:
            values = np.full((len(batch.node_ids), LEAF_EVAL_OUTPUT_WIDTH), 0.5, dtype=np.float32)
            return LeafEvalBatchOutput(node_ids=batch.node_ids, values=values)

    tree = make_toy_pipeline_tree()
    state = DenseCfrState(
        regret_sums=((0.0, 0.0), (0.0,),),
        strategy_sums=((0.0, 0.0), (0.0,),),
    )

    result = run_dense_backward_cfv_iteration(
        tree,
        state,
        backend=GpuLeafBackend(kernel=_Kernel()),
        infoset_strategies={
            InfosetId(0): (0.25, 0.75),
            InfosetId(1): (1.0,),
        },
    )

    assert result.regret_sums[0] == (-1.125, 0.375)
    assert result.strategy_sums[0] == (0.5, 0.5)


def test_run_dense_backward_cfv_iterations_accumulates_changes() -> None:
    class _Kernel:
        def __call__(self, batch: LeafEvalBatchInput) -> LeafEvalBatchOutput:
            values = np.full((len(batch.node_ids), LEAF_EVAL_OUTPUT_WIDTH), 0.5, dtype=np.float32)
            return LeafEvalBatchOutput(node_ids=batch.node_ids, values=values)

    tree = make_toy_pipeline_tree()
    state = DenseCfrState(
        regret_sums=((0.0, 0.0), (0.0,),),
        strategy_sums=((0.0, 0.0), (0.0,),),
    )

    result = run_dense_backward_cfv_iterations(
        tree,
        state,
        3,
        backend=GpuLeafBackend(kernel=_Kernel()),
        infoset_strategies={
            InfosetId(0): (0.25, 0.75),
            InfosetId(1): (1.0,),
        },
    )

    assert result != state
    assert result.regret_sums[0] == (-3.375, 1.125)

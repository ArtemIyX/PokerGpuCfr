from __future__ import annotations

import pytest

from pokergpu.core.betting import Chips
from pokergpu.cfr.solver import (
    SolverState,
    aggregate_root_action_values,
    make_toy_public_tree,
    propagate_reach,
    propagate_opponent_reach,
    run_tree_root_iteration,
)
from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.cfr.stage2 import aggregate_prob_sum
from pokergpu.tree.public_tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


def test_make_toy_public_tree_is_small_and_solved_like() -> None:
    tree = make_toy_public_tree()

    assert tree.node_count == 3
    assert tree.node_types[0] is NodeType.PLAYER0
    assert tree.child_count[0] == 2


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

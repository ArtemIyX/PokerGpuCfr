from __future__ import annotations

from typing import cast

import pytest

from pokergpu.cfr.leaf_backend_factory import create_heuristic_leaf_backend
from pokergpu.cfr.leaf_backend_factory import create_leaf_backend
from pokergpu.cfr.stage2 import build_leaf_eval_batch
from pokergpu.cfr.stage2 import aggregate_prob_sum
from pokergpu.cfr.stage1 import propagate_forward
from pokergpu.cfr.solver import CfrVariant
from pokergpu.cfr.solver import DenseCfrState
from pokergpu.cfr.solver import GameVariant
from pokergpu.cfr.solver import SolverStageRequest
from pokergpu.cfr.solver import TimingSpec
from pokergpu.cfr.solver import build_dense_infoset_table
from pokergpu.cfr.solver import make_game_public_tree
from pokergpu.cfr.solver import run_solver_stage
from pokergpu.core.board import Board
from pokergpu.core.board import Street
from pokergpu.abstraction.actions import action_labels_for_street
from pokergpu.abstraction.actions import make_holdem_hu_profile
from pokergpu.solver_holdem_hu_cli import _format_root_strategy


def test_holdem_hu_tree_builds_dense_infosets() -> None:
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)
    table = build_dense_infoset_table(tree)

    assert table.infoset_count == 4
    assert table.infoset_order == (0, 1, 2, 3)
    assert table.action_counts == (6, 6, 6, 6)
    assert table.action_labels[0][0] == "check"
    assert table.action_labels[1] != table.action_labels[0]
    assert table.action_labels[2] != table.action_labels[1]
    assert table.action_labels[3] != table.action_labels[2]


def test_holdem_hu_root_strategy_labels_match_actions() -> None:
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)
    table = build_dense_infoset_table(tree)
    state = DenseCfrState(
        regret_sums=tuple((0.0,) * 6 for _ in range(table.infoset_count)),
        strategy_sums=tuple((1.0, 2.0, 3.0, 4.0, 5.0, 6.0) if index == 0 else (0.0,) * 6 for index in range(table.infoset_count)),
    )

    root_strategy = _format_root_strategy(state, tree)

    assert root_strategy is not None
    assert "check:" in root_strategy
    assert "bet:25pct:" in root_strategy
    assert "bet:150pct:" in root_strategy
    assert len(table.action_labels[0]) == 6


def test_holdem_hu_action_labels_vary_by_street() -> None:
    profile = make_holdem_hu_profile()

    preflop = action_labels_for_street(profile, Street.PREFLOP)
    flop = action_labels_for_street(profile, Street.FLOP)
    turn = action_labels_for_street(profile, Street.TURN)
    river = action_labels_for_street(profile, Street.RIVER)

    assert preflop != flop
    assert flop != turn
    assert turn != river
    assert preflop[0] == "check"
    assert flop[0] == "check"
    assert "bet:" in flop[1]
    assert preflop != river


def test_holdem_hu_solver_smoke_test_runs_end_to_end() -> None:
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)
    table = build_dense_infoset_table(tree)
    request = SolverStageRequest(
        game=GameVariant.HOLDEM_HU,
        cfr_variant=CfrVariant.CFR,
        depth_limit=1,
        timing=TimingSpec(measure=False),
    )
    dense_state = DenseCfrState(
        regret_sums=tuple((0.0, 0.0) for _ in range(table.infoset_count)),
        strategy_sums=tuple((0.0, 0.0) for _ in range(table.infoset_count)),
    )

    result = run_solver_stage(
        request,
        tree=tree,
        dense_state=dense_state,
        board=Board(cards=()),
        backend=create_heuristic_leaf_backend(),
    )

    assert result.final_state is not None
    assert len(result.final_state.regret_sums) == table.infoset_count
    assert len(result.final_state.strategy_sums) == table.infoset_count
    root_strategy = cast(tuple[float, ...], result.final_state.strategy_sums[0])
    total = sum(root_strategy)
    assert total > 0.0
    assert len(root_strategy) == 6
    normalized = tuple(value / total for value in root_strategy)
    assert abs(sum(normalized) - 1.0) < 1e-6


def test_holdem_hu_leaf_features_include_board_context() -> None:
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)
    forward = propagate_forward(tree)
    aggregate = aggregate_prob_sum(tree, forward, Board.from_str("AhKdTc"))

    assert aggregate.leaf_batch.features.shape[1] == 162
    assert aggregate.leaf_batch.features.shape[0] == len(aggregate.leaf_node_ids)
    assert all(value == 1.0 for value in aggregate.leaf_batch.features[:, 2])
    assert all(value == 3.0 for value in aggregate.leaf_batch.features[:, 3])
    assert all(value != 0.0 for value in aggregate.leaf_batch.features[:, 4])
    assert all(value >= 0.0 for value in aggregate.leaf_batch.features[:, 6:58].ravel())


def test_holdem_hu_leaf_batch_works_with_gpu_backend() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Hold'em GPU leaf parity")

    tree = make_game_public_tree(GameVariant.HOLDEM_HU)
    forward = propagate_forward(tree)
    aggregate = aggregate_prob_sum(tree, forward, Board.from_str("AhKdTc"))
    leaf_eval_batch = build_leaf_eval_batch(aggregate.leaf_batch)
    heuristic_backend = create_heuristic_leaf_backend()
    gpu_backend = create_leaf_backend()

    heuristic_result = heuristic_backend.evaluate(leaf_eval_batch)
    gpu_result = gpu_backend.evaluate(leaf_eval_batch)

    assert heuristic_result.node_ids == gpu_result.node_ids
    assert heuristic_result.values.shape == gpu_result.values.shape
    assert gpu_result.values.shape[1] == 1

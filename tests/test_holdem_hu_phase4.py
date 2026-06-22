from __future__ import annotations

from pokergpu.cfr.leaf_backend_factory import create_heuristic_leaf_backend
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


def test_holdem_hu_tree_builds_dense_infosets() -> None:
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)
    table = build_dense_infoset_table(tree)

    assert table.infoset_count == 2
    assert table.infoset_order == (0, 1)
    assert table.action_counts == (2, 2)


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
    root_strategy = result.final_state.strategy_sums[0]
    total = sum(root_strategy)
    assert total > 0.0
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

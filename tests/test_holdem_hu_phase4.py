from __future__ import annotations

from types import SimpleNamespace
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
from pokergpu.tree.public_tree import NodeId
from pokergpu import solver_holdem_hu_cli
from pokergpu.solver_holdem_hu_cli import _format_root_strategy


def test_holdem_hu_tree_builds_dense_infosets() -> None:
    build_dense_infoset_table.cache_clear()
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)
    table = build_dense_infoset_table(tree)

    assert table.infoset_count == 1
    assert table.infoset_order == (0,)
    assert table.action_counts == (6,)
    assert table.action_labels[0][0] == "check"


def test_holdem_hu_tree_shape_is_deterministic() -> None:
    first = make_game_public_tree(GameVariant.HOLDEM_HU)
    second = make_game_public_tree(GameVariant.HOLDEM_HU)

    assert first == second
    assert first.node_count == 7
    assert first.node_types[0] is first.node_types[0]


def test_holdem_hu_tree_root_and_street_node_labels() -> None:
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)

    labels0 = tree.action_labels[0]

    assert labels0 is not None
    assert labels0 == ("check", "bet:25pct", "bet:50pct", "bet:75pct", "bet:100pct", "bet:150pct")


def test_holdem_hu_tree_transitions_flow_street_to_street() -> None:
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)

    assert tuple(int(link.child) for link in tree.child_links(NodeId(0))) == (1, 2, 3, 4, 5, 6)


def test_holdem_hu_tree_terminal_nodes_have_no_actions() -> None:
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)

    for node_index in range(1, tree.node_count):
        assert tree.child_count[node_index] == 0
        assert tree.child_links(NodeId(node_index)) == ()


def test_holdem_hu_tree_action_label_width_matches_child_count() -> None:
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)

    for node_index in range(1):
        assert len(tree.action_labels[node_index] or ()) == tree.child_count[node_index]


def test_holdem_hu_root_strategy_labels_match_actions() -> None:
    build_dense_infoset_table.cache_clear()
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
    assert flop[0] == "flop_check"
    assert "bet:" in flop[1]
    assert preflop != river


def test_holdem_hu_tree_rejects_bad_manual_action_labels() -> None:
    from pokergpu.tree.public_tree import ChildLink
    from pokergpu.tree.public_tree import InfosetId
    from pokergpu.tree.public_tree import NodeType
    from pokergpu.tree.public_tree import PublicTree

    with pytest.raises(ValueError, match="action_labels must match node count"):
        PublicTree(
            node_types=(NodeType.PLAYER0, NodeType.TERMINAL),
            first_child=(0, 1),
            child_count=(1, 0),
            children=(ChildLink(child=NodeId(1)),),
            infoset_ids=(InfosetId(0), None),
            terminal_payoffs=(None, None),
            action_labels=(("check",),),
        )


def test_holdem_hu_solver_smoke_test_runs_end_to_end() -> None:
    build_dense_infoset_table.cache_clear()
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)
    table = build_dense_infoset_table(tree)
    request = SolverStageRequest(
        game=GameVariant.HOLDEM_HU,
        cfr_variant=CfrVariant.CFR,
        depth_limit=1,
        timing=TimingSpec(measure=False),
    )
    dense_state = DenseCfrState(
        regret_sums=tuple(tuple(0.0 for _ in range(table.action_counts[index])) for index in range(table.infoset_count)),
        strategy_sums=tuple(tuple(0.0 for _ in range(table.action_counts[index])) for index in range(table.infoset_count)),
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
    assert len(root_strategy) == table.action_counts[0]
    normalized = tuple(value / total for value in root_strategy)
    assert abs(sum(normalized) - 1.0) < 1e-6


def test_holdem_hu_root_action_values_are_not_all_zero() -> None:
    build_dense_infoset_table.cache_clear()
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)
    table = build_dense_infoset_table(tree)
    request = SolverStageRequest(
        game=GameVariant.HOLDEM_HU,
        cfr_variant=CfrVariant.CFR,
        depth_limit=1,
        timing=TimingSpec(measure=False),
    )
    dense_state = DenseCfrState(
        regret_sums=tuple(tuple(0.0 for _ in range(table.action_counts[index])) for index in range(table.infoset_count)),
        strategy_sums=tuple(tuple(0.0 for _ in range(table.action_counts[index])) for index in range(table.infoset_count)),
    )

    result = run_solver_stage(
        request,
        tree=tree,
        dense_state=dense_state,
        board=Board(cards=()),
        backend=create_heuristic_leaf_backend(),
    )

    assert result.diagnostics is not None
    assert result.diagnostics["root_action_values"] != (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_holdem_hu_cli_debug_prints_root_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run_solver_stage(*args: object, **kwargs: object) -> SimpleNamespace:
        request = args[0]
        return SimpleNamespace(
            request=request,
            final_state=kwargs.get("dense_state"),
            timing_seconds=None,
            profiler_output=None,
            diagnostics={
                "game": "holdem_hu",
                "cfr_variant": "cfr",
                "tree_nodes": 10,
                "root_infoset": 0,
                "root_regrets": (0.5, -0.5, 0.0, 0.0, 0.0, 0.0),
                "root_strategy_sums": (2.0, 1.0, 0.0, 0.0, 0.0, 0.0),
                "root_action_values": (1.0, -1.0, 0.0, 0.0, 0.0, 0.0),
                "root_node_value": 1.25,
            },
        )

    class _FakeDebugSink:
        def add_scalar(self, *args: object, **kwargs: object) -> None:
            return None

        def add_histogram(self, *args: object, **kwargs: object) -> None:
            return None

        def add_text(self, *args: object, **kwargs: object) -> None:
            return None

        def add_sample(self, *args: object, **kwargs: object) -> None:
            return None

        def flush(self) -> None:
            return None

    class _FakeDebugSession:
        def __init__(self) -> None:
            self.sink = _FakeDebugSink()

        def close(self) -> None:
            return None

    monkeypatch.setattr(solver_holdem_hu_cli, "run_solver_stage", fake_run_solver_stage)
    monkeypatch.setattr(solver_holdem_hu_cli, "create_debug_session", lambda spec, run_name: _FakeDebugSession())

    exit_code = solver_holdem_hu_cli.main(
        [
            "--variant",
            "cfr",
            "--depth",
            "1",
            "--debug",
        ]
    )

    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "diagnostic.root_infoset=0" in captured
    assert "diagnostic.root_regrets=(0.5, -0.5, 0.0, 0.0, 0.0, 0.0)" in captured
    assert "diagnostic.root_strategy_sums=(2.0, 1.0, 0.0, 0.0, 0.0, 0.0)" in captured
    assert "diagnostic.root_action_values=(1.0, -1.0, 0.0, 0.0, 0.0, 0.0)" in captured
    assert "diagnostic.root_node_value=1.25" in captured
    assert "debug.root_strategy=" in captured


def test_holdem_hu_leaf_features_include_board_context() -> None:
    build_dense_infoset_table.cache_clear()
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
    build_dense_infoset_table.cache_clear()
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

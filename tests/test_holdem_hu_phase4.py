from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
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

    assert table.infoset_count == 19
    assert table.infoset_order == tuple(range(19))
    assert table.action_counts[0] == 6
    assert table.action_counts[1:] == (1,) * 18
    assert table.action_labels[0][0] == "check"


def test_holdem_hu_tree_shape_is_deterministic() -> None:
    first = make_game_public_tree(GameVariant.HOLDEM_HU)
    second = make_game_public_tree(GameVariant.HOLDEM_HU)

    assert first == second
    assert first.node_count == 25
    assert first.node_types[0] is first.node_types[0]
    assert first.node_types[-1].value == "leaf"


def test_holdem_hu_tree_root_and_street_node_labels() -> None:
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)

    labels0 = tree.action_labels[0]

    assert labels0 is not None
    assert labels0 == ("check", "bet:25pct", "bet:50pct", "bet:75pct", "bet:100pct", "bet:150pct")


def test_holdem_hu_tree_transitions_flow_street_to_street() -> None:
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)

    assert tuple(int(link.child) for link in tree.child_links(NodeId(0))) == (1, 2, 3, 4, 5, 6)
    assert all(tree.child_count[index] == 1 for index in range(1, 19))
    assert all(tree.child_count[index] == 0 for index in range(19, 25))


def test_holdem_hu_tree_terminal_nodes_have_no_actions() -> None:
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)

    for node_index in range(19, tree.node_count):
        assert tree.child_count[node_index] == 0
        assert tree.child_links(NodeId(node_index)) == ()


def test_holdem_hu_tree_action_label_width_matches_child_count() -> None:
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)

    assert len(tree.action_labels[0] or ()) == tree.child_count[0]


def test_holdem_hu_root_strategy_labels_match_actions() -> None:
    build_dense_infoset_table.cache_clear()
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)
    table = build_dense_infoset_table(tree)
    state = DenseCfrState(
        regret_sums=tuple((0.0,) * table.action_counts[index] for index in range(table.infoset_count)),
        strategy_sums=tuple(
            (1.0, 2.0, 3.0, 4.0, 5.0, 6.0) if index == 0 else (0.0,) * table.action_counts[index]
            for index in range(table.infoset_count)
        ),
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


def test_holdem_hu_leaf_branch_changes_with_board_context() -> None:
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

    no_board = run_solver_stage(
        request,
        tree=tree,
        dense_state=dense_state,
        board=Board(cards=()),
        backend=create_heuristic_leaf_backend(),
    )
    flop_board = run_solver_stage(
        request,
        tree=tree,
        dense_state=dense_state,
        board=Board.from_str("AhKdTc"),
        backend=create_heuristic_leaf_backend(),
    )

    assert no_board.diagnostics is not None
    assert flop_board.diagnostics is not None
    assert no_board.diagnostics["root_action_values"] != flop_board.diagnostics["root_action_values"]


def test_holdem_hu_root_strategy_accumulates_over_iterations() -> None:
    build_dense_infoset_table.cache_clear()
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)
    table = build_dense_infoset_table(tree)
    request = SolverStageRequest(
        game=GameVariant.HOLDEM_HU,
        cfr_variant=CfrVariant.CFR,
        depth_limit=1,
        iterations=5,
        timing=TimingSpec(measure=False),
    )
    dense_state = DenseCfrState(
        regret_sums=tuple(tuple(0.0 for _ in range(table.action_counts[index])) for index in range(table.infoset_count)),
        strategy_sums=tuple(tuple(0.0 for _ in range(table.action_counts[index])) for index in range(table.infoset_count)),
    )

    current: DenseCfrState | None = dense_state
    first = None
    last = None
    board = Board.from_str("AhKdTc")
    for _ in range(request.iterations):
        assert current is not None
        current_state = current
        result = run_solver_stage(
            request,
            tree=tree,
            dense_state=current_state,
            board=board,
            backend=create_heuristic_leaf_backend(),
        )
        assert result.final_state is not None
        current = result.final_state if isinstance(result.final_state, DenseCfrState) else None
        assert current is not None
        root_row = current.strategy_sums[0]
        total = sum(root_row)
        normalized = tuple(value / total for value in root_row) if total > 0.0 else tuple(0.0 for _ in root_row)
        if first is None:
            first = normalized
        last = normalized

    assert first is not None
    assert last is not None
    assert first != last
    assert max(last) > min(last)
    assert sum(abs(a - b) for a, b in zip(first, last, strict=True)) > 0.0


def test_holdem_hu_root_regrets_accumulate_over_iterations() -> None:
    build_dense_infoset_table.cache_clear()
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)
    table = build_dense_infoset_table(tree)
    request = SolverStageRequest(
        game=GameVariant.HOLDEM_HU,
        cfr_variant=CfrVariant.CFR,
        depth_limit=1,
        iterations=2,
        timing=TimingSpec(measure=False),
    )
    dense_state = DenseCfrState(
        regret_sums=tuple(tuple(0.0 for _ in range(table.action_counts[index])) for index in range(table.infoset_count)),
        strategy_sums=tuple(tuple(0.0 for _ in range(table.action_counts[index])) for index in range(table.infoset_count)),
    )

    board = Board.from_str("AhKdTc")
    first = run_solver_stage(
        request,
        tree=tree,
        dense_state=dense_state,
        board=board,
        backend=create_heuristic_leaf_backend(),
    )
    assert first.final_state is not None
    first_state = cast(DenseCfrState, first.final_state)
    second = run_solver_stage(
        request,
        tree=tree,
        dense_state=first_state,
        board=board,
        backend=create_heuristic_leaf_backend(),
    )

    first_diagnostics = first.diagnostics
    second_diagnostics = second.diagnostics
    assert first_diagnostics is not None
    assert second_diagnostics is not None
    first_regrets = cast(tuple[float, ...], first_diagnostics["root_regrets"])
    second_regrets = cast(tuple[float, ...], second_diagnostics["root_regrets"])
    assert first_regrets != second_regrets
    assert any(abs(second_value) >= abs(first_value) for first_value, second_value in zip(first_regrets, second_regrets, strict=True))


@pytest.mark.parametrize(
    "variant",
    [CfrVariant.CFR, CfrVariant.CFR_PLUS, CfrVariant.DCFR, CfrVariant.PREDICTIVE_CFR_PLUS],
)
def test_holdem_hu_supported_variants_route_through_solver(
    variant: CfrVariant,
) -> None:
    build_dense_infoset_table.cache_clear()
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)
    table = build_dense_infoset_table(tree)
    request = SolverStageRequest(
        game=GameVariant.HOLDEM_HU,
        cfr_variant=variant,
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
        board=Board.from_str("AhKdTc"),
        backend=create_heuristic_leaf_backend(),
    )

    assert result.request.cfr_variant is variant
    assert result.final_state is not None
    assert result.diagnostics is not None
    assert result.diagnostics["cfr_variant"] == variant.value


def test_holdem_hu_repeat_runs_are_deterministic_for_fixed_seed() -> None:
    build_dense_infoset_table.cache_clear()
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)
    table = build_dense_infoset_table(tree)
    request = SolverStageRequest(
        game=GameVariant.HOLDEM_HU,
        cfr_variant=CfrVariant.CFR,
        depth_limit=1,
        iterations=3,
        seed=17,
        timing=TimingSpec(measure=False),
    )
    dense_state = DenseCfrState(
        regret_sums=tuple(tuple(0.0 for _ in range(table.action_counts[index])) for index in range(table.infoset_count)),
        strategy_sums=tuple(tuple(0.0 for _ in range(table.action_counts[index])) for index in range(table.infoset_count)),
    )
    board = Board.from_str("AhKdTc")

    first = run_solver_stage(
        request,
        tree=tree,
        dense_state=dense_state,
        board=board,
        backend=create_heuristic_leaf_backend(),
    )
    second = run_solver_stage(
        request,
        tree=tree,
        dense_state=dense_state,
        board=board,
        backend=create_heuristic_leaf_backend(),
    )

    assert first.diagnostics == second.diagnostics
    assert first.final_state == second.final_state


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


def test_holdem_hu_random_state_mode_uses_seeded_board(
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
            diagnostics={},
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

    captured_runs: list[str] = []
    for seed in (17, 18, 19, 20):
        exit_code = solver_holdem_hu_cli.main(
            [
                "--variant",
                "cfr",
                "--depth",
                "1",
                "--state-mode",
                "random",
                "--seed",
                str(seed),
                "--debug",
            ]
        )
        captured = capsys.readouterr().out
        assert exit_code == 0
        assert "state_mode=random" in captured
        assert f"state_seed={seed}" in captured
        assert "board=" in captured
        assert "board_cards=" in captured
        captured_runs.append(captured)

    assert any("board_cards=[]" not in run for run in captured_runs)
    assert any("board_cards=[]" in run for run in captured_runs)


def test_holdem_hu_tree_depth_limit_hit_count_is_reported() -> None:
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
        board=Board.from_str("AhKdTc"),
        backend=create_heuristic_leaf_backend(),
    )

    assert result.diagnostics is not None
    assert cast(int, result.diagnostics["tree_depth_limit_hits"]) >= 1


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
    assert all(value >= 0.0 for value in np.asarray(aggregate.leaf_batch.features[:, 6:58].ravel(), dtype=np.float32))


def test_holdem_hu_heuristic_leaf_backend_is_board_sensitive() -> None:
    build_dense_infoset_table.cache_clear()
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)
    forward = propagate_forward(tree)

    empty_batch = build_leaf_eval_batch(aggregate_prob_sum(tree, forward, Board(cards=())).leaf_batch)
    flop_batch = build_leaf_eval_batch(aggregate_prob_sum(tree, forward, Board.from_str("AhKdTc")).leaf_batch)
    backend = create_heuristic_leaf_backend()

    empty_values = backend.evaluate(empty_batch)
    flop_values = backend.evaluate(flop_batch)

    assert isinstance(empty_values, type(flop_values))
    assert empty_values.values.shape == flop_values.values.shape
    assert tuple(empty_values.values[:, 0]) != tuple(flop_values.values[:, 0])


def test_holdem_hu_heuristic_leaf_backend_stays_in_sane_range() -> None:
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
        board=Board.from_str("AhKdTc"),
        backend=create_heuristic_leaf_backend(),
    )

    assert result.diagnostics is not None
    root_values = cast(tuple[float, ...], result.diagnostics["root_action_values"])
    assert max(abs(value) for value in root_values) <= 3.5
    assert max(root_values) - min(root_values) <= 6.0


def test_holdem_hu_cli_defaults_to_model_leaf_backend() -> None:
    parser = solver_holdem_hu_cli.build_parser()
    args = parser.parse_args(["--variant", "cfr", "--depth", "1"])

    assert args.leaf_evaluator == "model"


def test_holdem_hu_cli_exposes_compact_tree_flag() -> None:
    parser = solver_holdem_hu_cli.build_parser()
    args = parser.parse_args(["--variant", "cfr", "--depth", "1", "--compact-tree"])

    assert args.compact_tree is True


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

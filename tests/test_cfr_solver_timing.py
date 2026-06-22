from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any
from types import ModuleType

import pytest

from pokergpu.cfr.solver import CfrVariant
from pokergpu.cfr.solver import GameVariant
from pokergpu.cfr.solver import ProfilingKind
from pokergpu.cfr.solver import ProfilerSpec
from pokergpu.cfr.solver import SolverStageRequest
from pokergpu.cfr.solver import TimingSpec
from pokergpu.cfr.solver import run_solver_stage
from pokergpu.cfr.solver.kuhn import make_kuhn_public_tree
from pokergpu.cfr.solver.state import DenseCfrState
from pokergpu.core.board import Board


def test_run_solver_stage_returns_timing_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    import pokergpu.cfr.solver.service as service

    request = SolverStageRequest(
        game=GameVariant.KUHN,
        cfr_variant=CfrVariant.CFR,
        depth_limit=2,
        measure_time=True,
        timing=TimingSpec(measure=True, include_stage_breakdown=True, include_branch_breakdown=True),
    )
    tree = make_kuhn_public_tree()
    state = DenseCfrState(
        regret_sums=tuple((0.0, 0.0) for _ in range(6)),
        strategy_sums=tuple((0.0, 0.0) for _ in range(6)),
    )
    board = Board(cards=())
    counter = itertools.count()
    monkeypatch.setattr(_service_time(service), "perf_counter", lambda: float(next(counter)))
    _patch_service_stages(monkeypatch, service, tree.node_count, state)

    result = run_solver_stage(request, tree=tree, dense_state=state, board=board)

    assert result.timing_seconds is not None
    assert "total" in result.timing_seconds
    assert "stage1_forward" in result.timing_seconds
    assert "stage2_aggregate" in result.timing_seconds
    assert "branch_total" in result.timing_seconds
    assert "branch_overlap" in result.timing_seconds
    assert result.profiler_output is None


def test_run_solver_stage_writes_cprofile_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pokergpu.cfr.solver.service as service

    output_path = tmp_path / "solver.prof"
    request = SolverStageRequest(
        game=GameVariant.KUHN,
        cfr_variant=CfrVariant.CFR,
        depth_limit=2,
        profiler=ProfilerSpec(kind=ProfilingKind.CPROFILE, output_path=str(output_path)),
    )
    tree = make_kuhn_public_tree()
    state = DenseCfrState(
        regret_sums=tuple((0.0, 0.0) for _ in range(6)),
        strategy_sums=tuple((0.0, 0.0) for _ in range(6)),
    )
    board = Board(cards=())
    counter = itertools.count()
    monkeypatch.setattr(_service_time(service), "perf_counter", lambda: float(next(counter)))
    _patch_service_stages(monkeypatch, service, tree.node_count, state)

    result = run_solver_stage(request, tree=tree, dense_state=state, board=board)

    assert result.profiler_output == str(output_path)
    assert output_path.exists()


def test_run_solver_stage_creates_default_cprofile_output_when_path_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pokergpu.cfr.solver.service as service

    request = SolverStageRequest(
        game=GameVariant.KUHN,
        cfr_variant=CfrVariant.CFR,
        depth_limit=2,
        profiler=ProfilerSpec(kind=ProfilingKind.CPROFILE),
    )
    tree = make_kuhn_public_tree()
    state = DenseCfrState(
        regret_sums=tuple((0.0, 0.0) for _ in range(6)),
        strategy_sums=tuple((0.0, 0.0) for _ in range(6)),
    )
    board = Board(cards=())
    counter = itertools.count()
    monkeypatch.setattr(_service_time(service), "perf_counter", lambda: float(next(counter)))
    _patch_service_stages(monkeypatch, service, tree.node_count, state)

    result = run_solver_stage(request, tree=tree, dense_state=state, board=board)

    assert result.profiler_output is not None
    assert result.profiler_output.endswith(".prof")


def test_run_solver_stage_accepts_torch_profiler_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    import pokergpu.cfr.solver.service as service

    request = SolverStageRequest(
        game=GameVariant.KUHN,
        cfr_variant=CfrVariant.CFR,
        depth_limit=2,
        profiler=ProfilerSpec(kind=ProfilingKind.TORCH),
    )
    tree = make_kuhn_public_tree()
    state = DenseCfrState(
        regret_sums=tuple((0.0, 0.0) for _ in range(6)),
        strategy_sums=tuple((0.0, 0.0) for _ in range(6)),
    )
    board = Board(cards=())
    counter = itertools.count()
    monkeypatch.setattr(_service_time(service), "perf_counter", lambda: float(next(counter)))
    _patch_service_stages(monkeypatch, service, tree.node_count, state)

    result = run_solver_stage(request, tree=tree, dense_state=state, board=board)

    assert result.request == request
    assert result.profiler_output is None


def _patch_service_stages(
    monkeypatch: pytest.MonkeyPatch,
    service: ModuleType,
    node_count: int,
    dense_state: DenseCfrState,
) -> None:
    monkeypatch.setattr(service, "propagate_forward", _make_forward_stub(node_count))
    monkeypatch.setattr(service, "aggregate_prob_sum", _make_aggregate_stub(node_count))
    monkeypatch.setattr(service, "compute_opponent_reach", _make_opponent_stub(node_count))
    monkeypatch.setattr(service, "compute_showdown_equity", _make_showdown_stub(node_count))
    monkeypatch.setattr(service, "evaluate_leaf_node_values", _make_leaf_stub(node_count))
    monkeypatch.setattr(service, "backward_cfv", _make_backward_stub())
    monkeypatch.setattr(service, "apply_dense_backward_cfv_update", _make_update_stub(dense_state))


def _service_time(service: ModuleType) -> Any:
    return service.time


def _forward(node_count: int) -> object:
    return type(
        "Forward",
        (),
        {
            "node_reach": tuple(1.0 for _ in range(node_count)),
            "infoset_reach": tuple(1.0 for _ in range(6)),
            "action_reach": tuple(() for _ in range(node_count)),
        },
    )()


def _aggregate(node_count: int) -> object:
    return type(
        "Aggregate",
        (),
        {
            "node_aggregate": type("AggNode", (), {"reach": tuple(1.0 for _ in range(node_count))})(),
            "leaf_node_ids": tuple(),
        },
    )()


def _opponent(node_count: int) -> object:
    return type(
        "Opponent",
        (),
        {
            "node_opponent_reach": tuple(1.0 for _ in range(node_count)),
            "node_opponent_share": tuple(1.0 for _ in range(node_count)),
            "node_hand_opponent_reach": tuple(() for _ in range(node_count)),
        },
    )()


def _showdown(node_count: int) -> object:
    return type(
        "Showdown",
        (),
        {
            "node_showdown_equity": (0.0,) * node_count,
            "node_showdown_equity_bb": (0.0,) * node_count,
            "input_rows": type("Rows", (), {"rows": ()})(),
            "output_rows": (),
        },
    )()


def _leaf_values(node_count: int) -> object:
    return type("Leaf", (), {"node_values": (1.0,) * node_count})()


def _backward() -> object:
    return type(
        "Backward",
        (),
        {
            "node_values": (0.0,) * 19,
            "infoset_values": (0.0,) * 6,
            "action_values": ((),) * 19,
        },
    )()


def _make_forward_stub(node_count: int) -> Any:
    def _stub(*args: object, **kwargs: object) -> object:
        return _forward(node_count)

    return _stub


def _make_aggregate_stub(node_count: int) -> Any:
    def _stub(*args: object, **kwargs: object) -> object:
        return _aggregate(node_count)

    return _stub


def _make_opponent_stub(node_count: int) -> Any:
    def _stub(*args: object, **kwargs: object) -> object:
        return _opponent(node_count)

    return _stub


def _make_showdown_stub(node_count: int) -> Any:
    def _stub(*args: object, **kwargs: object) -> object:
        return _showdown(node_count)

    return _stub


def _make_leaf_stub(node_count: int) -> Any:
    def _stub(*args: object, **kwargs: object) -> object:
        return _leaf_values(node_count)

    return _stub


def _make_backward_stub() -> Any:
    def _stub(*args: object, **kwargs: object) -> object:
        return _backward()

    return _stub


def _make_update_stub(dense_state: DenseCfrState) -> Any:
    def _stub(*args: object, **kwargs: object) -> DenseCfrState:
        return dense_state

    return _stub

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import cast

import pytest

import pokergpu.solver_cli as solver_cli
from pokergpu.cfr.solver import CfrVariant
from pokergpu.cfr.solver import GameVariant
from pokergpu.cfr.solver import ProfilerSpec
from pokergpu.cfr.solver import ProfilingKind
from pokergpu.cfr.solver import SolverStageRequest
from pokergpu.cfr.solver import TimingSpec


def test_build_parser_exposes_solver_arguments() -> None:
    parser = solver_cli.build_parser()
    args = parser.parse_args(
        [
            "--game",
            "kuhn",
            "--variant",
            "cfr_plus",
            "--depth",
            "2",
            "--iterations",
            "3",
            "--cpu-workers",
            "4",
            "--measure-time",
            "--progress",
        ]
    )

    assert args.game == "kuhn"
    assert args.variant == "cfr_plus"
    assert args.depth == 2
    assert args.iterations == 3
    assert args.cpu_workers == 4
    assert args.measure_time is True
    assert args.progress is True


def test_main_runs_and_prints_summary(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[SolverStageRequest] = []
    progress_inputs: list[list[int]] = []

    def fake_run_solver_stage(*args: object, **kwargs: Any) -> SimpleNamespace:
        request = cast(SolverStageRequest, args[0])
        calls.append(request)
        return SimpleNamespace(
            request=request,
            final_state=kwargs.get("dense_state"),
            timing_seconds={"total": 1.25},
            profiler_output=None,
            diagnostics={"game": request.game.value},
        )

    def fake_tqdm(iterable: Iterable[int], total: int | None = None, desc: str | None = None) -> list[int]:
        _ = total, desc
        data = list(iterable)
        progress_inputs.append(data)
        return data

    monkeypatch.setattr(solver_cli, "run_solver_stage", fake_run_solver_stage)
    monkeypatch.setattr(solver_cli, "tqdm_module", SimpleNamespace(tqdm=fake_tqdm))

    exit_code = solver_cli.main(
        [
            "--game",
            "kuhn",
            "--variant",
            "cfr",
            "--depth",
            "2",
            "--iterations",
            "2",
            "--measure-time",
            "--progress",
        ]
    )

    captured = capsys.readouterr().out

    assert exit_code == 0
    assert len(calls) == 2
    assert progress_inputs == [[0, 1]]
    assert "game=kuhn" in captured
    assert "variant=cfr" in captured
    assert "depth=2" in captured
    assert "total_seconds=1.250000" in captured


def test_main_accepts_profiler_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[ProfilerSpec | None] = []

    def fake_run_solver_stage(*args: object, **kwargs: Any) -> SimpleNamespace:
        request = cast(SolverStageRequest, args[0])
        seen.append(request.profiler)
        return SimpleNamespace(
            request=request,
            final_state=kwargs.get("dense_state"),
            timing_seconds=None,
            profiler_output="solver.prof",
            diagnostics={},
        )

    monkeypatch.setattr(solver_cli, "run_solver_stage", fake_run_solver_stage)

    exit_code = solver_cli.main(
        [
            "--game",
            "leduc",
            "--variant",
            "dcfr",
            "--depth",
            "2",
            "--profile",
            "cprofile",
            "--profile-output",
            "solver.prof",
        ]
    )

    assert exit_code == 0
    assert seen == [ProfilerSpec(kind=ProfilingKind.CPROFILE, output_path="solver.prof")]


def test_main_writes_summary_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_solver_stage(*args: object, **kwargs: Any) -> SimpleNamespace:
        request = cast(SolverStageRequest, args[0])
        return SimpleNamespace(
            request=request,
            final_state=kwargs.get("dense_state"),
            timing_seconds={"total": 2.5},
            profiler_output=None,
            diagnostics={"tree_nodes": 19},
        )

    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(solver_cli, "run_solver_stage", fake_run_solver_stage)

    exit_code = solver_cli.main(
        [
            "--game",
            "kuhn",
            "--variant",
            "cfr",
            "--depth",
            "2",
            "--summary-output",
            str(summary_path),
        ]
    )

    content = summary_path.read_text(encoding="utf-8")

    assert exit_code == 0
    assert '"game": "kuhn"' in content
    assert '"variant": "cfr"' in content
    assert '"tree_nodes": 19' in content

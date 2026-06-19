from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import cast

import pytest

import pokergpu.solver_cli as solver_cli
from pokergpu.cfr.solver import CfrVariant
from pokergpu.cfr.solver import DenseCfrState
from pokergpu.cfr.solver import GameStateMode
from pokergpu.cfr.solver import GameStateSpec
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
            "--seed",
            "17",
            "--measure-time",
            "--debug",
            "--progress",
        ]
    )

    assert args.game == "kuhn"
    assert args.variant == "cfr_plus"
    assert args.depth == 2
    assert args.iterations == 3
    assert args.cpu_workers == 4
    assert args.seed == 17
    assert args.measure_time is True
    assert args.debug is True
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
            "--seed",
            "11",
            "--board",
            "AhKdQs",
            "--measure-time",
            "--debug",
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
    assert "seed=11" in captured
    assert "board=AhKdQs" in captured
    assert "board_cards=[Ah, Kd, Qs]" in captured
    assert "state_mode=random" in captured
    assert "chips=unavailable" in captured
    assert "player_cards=unavailable" in captured
    assert "iterations=2" in captured
    assert "root_strategy=" in captured
    assert "debug.root_strategy=" in captured
    assert "total_seconds=1.250000" in captured


def test_main_debug_prints_seed_and_state_metadata(
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
            diagnostics={"tree_nodes": 19},
        )

    monkeypatch.setattr(solver_cli, "run_solver_stage", fake_run_solver_stage)

    exit_code = solver_cli.main(
        [
            "--game",
            "kuhn",
            "--variant",
            "cfr",
            "--depth",
            "2",
            "--seed",
            "7",
            "--debug",
        ]
    )

    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "seed=7" in captured
    assert "state_mode=random" in captured
    assert "board_cards=[]" in captured
    assert "chips=unavailable" in captured
    assert "player_cards=unavailable" in captured


def test_main_root_strategy_comes_from_final_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    states = [
        DenseCfrState(
            regret_sums=((0.0, 0.0), (0.0, 0.0)),
            strategy_sums=((1.0, 3.0), (0.0, 0.0)),
        ),
        DenseCfrState(
            regret_sums=((0.0, 0.0), (0.0, 0.0)),
            strategy_sums=((4.0, 0.0), (0.0, 0.0)),
        ),
    ]

    def fake_run_solver_stage(*args: object, **kwargs: object) -> SimpleNamespace:
        request = args[0]
        return SimpleNamespace(
            request=request,
            final_state=states.pop(0),
            timing_seconds=None,
            profiler_output=None,
            diagnostics={},
        )

    monkeypatch.setattr(solver_cli, "run_solver_stage", fake_run_solver_stage)

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
        ]
    )

    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "root_strategy=(1.000, 0.000)" in captured


def test_main_debug_exact_state_should_show_decoded_cards_and_chips(
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

    monkeypatch.setattr(solver_cli, "run_solver_stage", fake_run_solver_stage)

    exit_code = solver_cli.main(
        [
            "--game",
            "kuhn",
            "--variant",
            "cfr",
            "--depth",
            "2",
            "--state-mode",
            "exact",
            "--encoded-state",
            json.dumps(
                {
                    "board": "",
                    "players": [
                        {"player": 0, "hole_cards": ["Ah", "Kd"]},
                        {"player": 1, "hole_cards": ["Qs", "Jh"]},
                    ],
                    "stacks": [
                        {"player": 0, "stack": 900},
                        {"player": 1, "stack": 800},
                    ],
                    "bets": [
                        {"player": 0, "committed": 100},
                        {"player": 1, "committed": 100},
                    ],
                    "blinds": {"small_blind": 50, "big_blind": 100, "ante": 0},
                    "pot": 200,
                    "dealer": 0,
                    "to_act": 0,
                    "phase": "in_progress",
                }
            ),
            "--seed",
            "9",
            "--debug",
        ]
    )

    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "state_mode=exact" in captured
    assert "encoded_state=" in captured
    assert "chips=[p0=900, p1=800]" in captured
    assert "player_cards=[p0=AhKd, p1=QsJh]" in captured


def test_main_debug_exact_state_falls_back_for_non_json_payload(
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

    monkeypatch.setattr(solver_cli, "run_solver_stage", fake_run_solver_stage)

    exit_code = solver_cli.main(
        [
            "--game",
            "kuhn",
            "--variant",
            "cfr",
            "--depth",
            "2",
            "--state-mode",
            "exact",
            "--encoded-state",
            "raw-bytes",
            "--debug",
        ]
    )

    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "chips=unavailable" in captured
    assert "player_cards=unavailable" in captured


def test_root_strategy_should_change_when_state_changes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    states = [
        DenseCfrState(
            regret_sums=((0.0, 0.0), (0.0, 0.0)),
            strategy_sums=((1.0, 1.0), (0.0, 0.0)),
        ),
        DenseCfrState(
            regret_sums=((0.0, 0.0), (0.0, 0.0)),
            strategy_sums=((9.0, 1.0), (0.0, 0.0)),
        ),
    ]

    def fake_run_solver_stage(*args: object, **kwargs: object) -> SimpleNamespace:
        request = args[0]
        return SimpleNamespace(
            request=request,
            final_state=states.pop(0),
            timing_seconds=None,
            profiler_output=None,
            diagnostics={},
        )

    monkeypatch.setattr(solver_cli, "run_solver_stage", fake_run_solver_stage)

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
            "--seed",
            "13",
        ]
    )

    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "root_strategy=(0.500, 0.500)" not in captured
    assert "root_strategy=(0.900, 0.100)" in captured


def test_root_strategy_should_not_be_identical_across_different_exact_states(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    states = [
        DenseCfrState(
            regret_sums=((0.0, 0.0), (0.0, 0.0)),
            strategy_sums=((1.0, 3.0), (0.0, 0.0)),
        ),
        DenseCfrState(
            regret_sums=((0.0, 0.0), (0.0, 0.0)),
            strategy_sums=((3.0, 1.0), (0.0, 0.0)),
        ),
    ]

    payloads = [
        {
            "board": "",
            "players": [
                {"player": 0, "hole_cards": ["Ah", "Kd"]},
                {"player": 1, "hole_cards": ["Qs", "Jh"]},
            ],
            "stacks": [
                {"player": 0, "stack": 900},
                {"player": 1, "stack": 800},
            ],
            "bets": [
                {"player": 0, "committed": 100},
                {"player": 1, "committed": 100},
            ],
            "blinds": {"small_blind": 50, "big_blind": 100, "ante": 0},
            "pot": 200,
            "dealer": 0,
            "to_act": 0,
            "phase": "in_progress",
        },
        {
            "board": "",
            "players": [
                {"player": 0, "hole_cards": ["2c", "3d"]},
                {"player": 1, "hole_cards": ["As", "Kh"]},
            ],
            "stacks": [
                {"player": 0, "stack": 600},
                {"player": 1, "stack": 1400},
            ],
            "bets": [
                {"player": 0, "committed": 50},
                {"player": 1, "committed": 150},
            ],
            "blinds": {"small_blind": 50, "big_blind": 100, "ante": 0},
            "pot": 300,
            "dealer": 1,
            "to_act": 1,
            "phase": "in_progress",
        },
    ]

    def fake_run_solver_stage(*args: object, **kwargs: object) -> SimpleNamespace:
        request = cast(SolverStageRequest, args[0])
        index = 0 if request.state is None or request.state.encoded_state is None else len(request.state.encoded_state) % 2
        return SimpleNamespace(
            request=request,
            final_state=states[index],
            timing_seconds=None,
            profiler_output=None,
            diagnostics={},
        )

    monkeypatch.setattr(solver_cli, "run_solver_stage", fake_run_solver_stage)

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
            "--state-mode",
            "exact",
            "--encoded-state",
            json.dumps(payloads[0]),
            "--debug",
        ]
    )

    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "public_state=" in captured
    assert "root_strategy=(0.995, 0.005)" not in captured


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


def test_main_reports_seed_in_summary_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
            "--seed",
            "31",
            "--summary-output",
            str(summary_path),
        ]
    )

    content = summary_path.read_text(encoding="utf-8")

    assert exit_code == 0
    assert '"seed": 31' in content


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
            "--seed",
            "19",
            "--board",
            "AhKdQs",
            "--summary-output",
            str(summary_path),
        ]
    )

    content = summary_path.read_text(encoding="utf-8")

    assert exit_code == 0
    assert '"game": "kuhn"' in content
    assert '"variant": "cfr"' in content
    assert '"seed": 19' in content
    assert '"board": "AhKdQs"' in content
    assert '"tree_nodes": 19' in content
    assert '"root_strategy": "(0.500, 0.500)"' in content

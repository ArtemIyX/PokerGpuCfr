from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from typing import cast

import pytest

import pokergpu.solver_holdem_hu_cli as solver_holdem_hu_cli
from pokergpu.cfr.solver import DenseCfrState
from pokergpu.cfr.solver import GameStateMode
from pokergpu.cfr.solver import GameStateSpec
from pokergpu.cfr.solver import GameVariant
from pokergpu.cfr.solver import SolverStageRequest
from pokergpu.cfr.solver import make_game_public_tree
from pokergpu.cfr.solver.infosets import build_dense_infoset_table


def test_main_reports_exact_solved_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tree = make_game_public_tree(GameVariant.HOLDEM_HU, depth_limit=1)
    table = build_dense_infoset_table(tree)
    encoded = b"state"

    def fake_run_solver_stage(*args: object, **kwargs: object) -> SimpleNamespace:
        request = cast(SolverStageRequest, args[0])
        return SimpleNamespace(
            request=request,
            final_state=kwargs.get("dense_state"),
            timing_seconds=None,
            profiler_output=None,
            diagnostics={"board": "AhKdTc"},
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
            self.log_dir = None

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
            "--state-mode",
            "exact",
            "--encoded-state",
            encoded.decode(),
            "--debug",
            "--compact-tree",
        ]
    )

    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "state_mode=exact" in captured
    assert "encoded_state=" in captured
    assert "debug_log_dir=" not in captured
    assert table.infoset_count > 0

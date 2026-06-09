import sys
from pathlib import Path

import pytest

from pokergpu import cli


def test_kuhn_cli_prints_strategy_summary(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["pokergpu", "kuhn", "50", "--variant", "cfr_plus"],
    )

    exit_code = cli.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "variant=cfr_plus" in captured.out
    assert "iterations=50" in captured.out
    assert "avg_value_p0=" in captured.out
    assert "root_bet_J=" in captured.out
    assert "root_bet_Q=" in captured.out
    assert "root_bet_K=" in captured.out


def test_leduc_cli_prints_strategy_summary(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["pokergpu", "leduc", "20", "--variant", "dcfr"],
    )

    exit_code = cli.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "variant=dcfr" in captured.out
    assert "iterations=20" in captured.out
    assert "avg_value_p0=" in captured.out
    assert "root_bet_J=" in captured.out
    assert "root_bet_Q=" in captured.out
    assert "root_bet_K=" in captured.out


def test_compare_toy_cli_writes_csv_with_progress(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "toy.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pokergpu",
            "compare-toy",
            "--output",
            str(output_path),
            "--kuhn",
            "1",
            "--leduc",
            "1",
            "--variants",
            "vanilla",
        ],
    )

    exit_code = cli.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "progress [" in captured.out
    assert f"output={output_path}" in captured.out
    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3

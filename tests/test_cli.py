import sys

import pytest

from pokergpu import cli


def test_kuhn_cli_prints_strategy_summary(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["pokergpu", "kuhn", "50"])

    exit_code = cli.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "iterations=50" in captured.out
    assert "avg_value_p0=" in captured.out
    assert "root_bet_J=" in captured.out
    assert "root_bet_Q=" in captured.out
    assert "root_bet_K=" in captured.out


def test_leduc_cli_prints_strategy_summary(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["pokergpu", "leduc", "20"])

    exit_code = cli.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "iterations=20" in captured.out
    assert "avg_value_p0=" in captured.out
    assert "root_bet_J=" in captured.out
    assert "root_bet_Q=" in captured.out
    assert "root_bet_K=" in captured.out

from __future__ import annotations

from pokergpu.cfr.toy import ToyCfrState, run_toy_cfr_iteration


def test_toy_cfr_iteration_prefers_higher_value_action() -> None:
    state = ToyCfrState(regret_sums=(0.0, 0.0), strategy_sums=(0.0, 0.0))

    first = run_toy_cfr_iteration(state, (1.0, -1.0))
    second = run_toy_cfr_iteration(first.state, (1.0, -1.0))
    third = run_toy_cfr_iteration(second.state, (1.0, -1.0))

    assert first.strategy == (0.5, 0.5)
    assert second.strategy[0] >= second.strategy[1]
    assert third.strategy[0] > third.strategy[1]


def test_toy_cfr_iteration_accumulates_average_strategy() -> None:
    state = ToyCfrState(regret_sums=(0.0, 0.0), strategy_sums=(0.0, 0.0))

    result = run_toy_cfr_iteration(state, (2.0, 0.0), reach_weight=3.0)

    assert result.state.strategy_sums == (1.5, 1.5)

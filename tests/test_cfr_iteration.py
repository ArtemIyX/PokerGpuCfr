import math

import numpy as np
import pytest

from pokergpu.cfr import InfosetLayout, InfosetStore, run_cfr_iteration


def test_cfr_iteration_updates_regrets_and_strategy_sums() -> None:
    store = InfosetStore.zeros(InfosetLayout.from_action_counts([2]))

    result = run_cfr_iteration(store, [np.array([1.0, -1.0], dtype=np.float32)])

    assert math.isclose(
        float(result.infoset_values[0]),
        0.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert np.allclose(store.regrets, np.array([1.0, -1.0], dtype=np.float32))
    assert np.allclose(store.strategy_sums, np.array([0.5, 0.5], dtype=np.float32))
    assert np.allclose(
        result.strategies[0],
        np.array([0.5, 0.5], dtype=np.float32),
        atol=1e-6,
    )


def test_cfr_iteration_uses_current_regrets_for_strategy() -> None:
    store = InfosetStore.zeros(InfosetLayout.from_action_counts([2]))
    store.regrets[:] = np.array([3.0, 1.0], dtype=np.float32)

    result = run_cfr_iteration(store, [np.array([2.0, 0.0], dtype=np.float32)])

    assert np.allclose(
        result.strategies[0],
        np.array([0.75, 0.25], dtype=np.float32),
        atol=1e-6,
    )
    assert np.allclose(store.strategy_sums, np.array([0.75, 0.25], dtype=np.float32))
    assert np.allclose(store.regrets, np.array([3.5, -0.5], dtype=np.float32))


def test_cfr_iteration_supports_multiple_infosets() -> None:
    store = InfosetStore.zeros(InfosetLayout.from_action_counts([2, 3]))

    run_cfr_iteration(
        store,
        [
            np.array([1.0, -1.0], dtype=np.float32),
            np.array([0.0, 4.0, -2.0], dtype=np.float32),
        ],
        strategy_weight=2.0,
    )

    assert np.allclose(store.strategy_sums[:2], np.array([1.0, 1.0], dtype=np.float32))
    assert np.allclose(
        store.strategy_sums[2:],
        np.array([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0], dtype=np.float32),
        atol=1e-6,
    )


def test_cfr_iteration_rejects_wrong_infoset_count() -> None:
    store = InfosetStore.zeros(InfosetLayout.from_action_counts([2]))

    with pytest.raises(ValueError):
        run_cfr_iteration(store, [])


def test_cfr_iteration_rejects_wrong_action_count() -> None:
    store = InfosetStore.zeros(InfosetLayout.from_action_counts([2]))

    with pytest.raises(ValueError):
        run_cfr_iteration(store, [np.array([1.0], dtype=np.float32)])


def test_cfr_iteration_can_limit_updates_to_active_infosets() -> None:
    store = InfosetStore.zeros(InfosetLayout.from_action_counts([2, 2]))

    run_cfr_iteration(
        store,
        [
            np.array([1.0, -1.0], dtype=np.float32),
            np.array([4.0, -4.0], dtype=np.float32),
        ],
        active_infosets=[1],
    )

    assert np.allclose(store.regrets[:2], np.zeros(2, dtype=np.float32))
    assert np.allclose(store.strategy_sums[:2], np.zeros(2, dtype=np.float32))
    assert np.allclose(store.regrets[2:], np.array([4.0, -4.0], dtype=np.float32))

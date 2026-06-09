import math

import numpy as np
import pytest

from pokergpu.cfr import InfosetLayout, InfosetStore, regret_matching


def test_infoset_layout_builds_contiguous_offsets() -> None:
    layout = InfosetLayout.from_action_counts([2, 3, 1])

    assert layout.action_counts == (2, 3, 1)
    assert layout.offsets == (0, 2, 5)
    assert layout.total_actions == 6


def test_infoset_layout_rejects_non_positive_action_counts() -> None:
    with pytest.raises(ValueError):
        InfosetLayout.from_action_counts([2, 0])


def test_infoset_store_exposes_infoset_slices() -> None:
    layout = InfosetLayout.from_action_counts([2, 3])
    store = InfosetStore.zeros(layout)
    store.regrets[:] = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    assert np.array_equal(
        store.regrets_for_infoset(1),
        np.array([3.0, 4.0, 5.0], dtype=np.float32),
    )


def test_regret_matching_uses_positive_regrets_only() -> None:
    strategy = regret_matching(np.array([-2.0, 3.0, 1.0], dtype=np.float32))

    assert math.isclose(float(strategy[0]), 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(strategy[1]), 0.75, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(strategy[2]), 0.25, rel_tol=0.0, abs_tol=1e-6)


def test_regret_matching_falls_back_to_uniform() -> None:
    strategy = regret_matching(np.array([-2.0, 0.0, -1.0], dtype=np.float32))

    assert np.allclose(
        strategy,
        np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32),
        atol=1e-6,
    )


def test_infoset_store_average_strategy_falls_back_to_uniform() -> None:
    store = InfosetStore.zeros(InfosetLayout.from_action_counts([3]))

    strategy = store.average_strategy(0)

    assert np.allclose(
        strategy,
        np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32),
        atol=1e-6,
    )


def test_infoset_store_average_strategy_normalizes_sum() -> None:
    store = InfosetStore.zeros(InfosetLayout.from_action_counts([2]))
    store.strategy_sums[:] = np.array([2.0, 6.0], dtype=np.float32)

    strategy = store.average_strategy(0)

    assert math.isclose(float(strategy[0]), 0.25, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(strategy[1]), 0.75, rel_tol=0.0, abs_tol=1e-6)

import math

import numpy as np
import pytest

from pokergpu.cfr import (
    InfosetLayout,
    InfosetStore,
    build_infoset_layout,
    regret_matching,
)


def test_infoset_layout_builds_contiguous_offsets() -> None:
    layout = build_infoset_layout([2, 3, 1])

    assert layout.infoset_count == 3
    assert layout.offsets == (0, 2, 5)
    assert layout.total_actions == 6
    assert layout.action_range(1) == slice(2, 5)


def test_infoset_store_uses_flat_contiguous_arrays() -> None:
    layout = InfosetLayout.from_action_counts((2, 1, 3))
    store = InfosetStore.zeros(layout)

    assert store.regrets.shape == (6,)
    assert store.strategy_sums.shape == (6,)
    assert store.regrets_for_infoset(0).shape == (2,)
    assert store.strategy_sums_for_infoset(2).shape == (3,)


def test_infoset_store_slices_are_views() -> None:
    layout = InfosetLayout.from_action_counts((2, 2))
    store = InfosetStore.zeros(layout)

    regrets_view = store.regrets_for_infoset(1)
    regrets_view[:] = np.array([1.0, 2.0], dtype=np.float32)

    assert math.isclose(float(store.regrets[2]), 1.0)
    assert math.isclose(float(store.regrets[3]), 2.0)


def test_average_strategy_falls_back_to_uniform() -> None:
    layout = InfosetLayout.from_action_counts((3,))
    store = InfosetStore.zeros(layout)

    average = store.average_strategy(0)

    assert average.shape == (3,)
    assert np.allclose(average, np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32))


def test_regret_matching_is_uniform_on_zero_regrets() -> None:
    result = regret_matching(np.zeros(4, dtype=np.float32))

    assert np.allclose(result, np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32))


def test_invalid_layout_rejects_non_contiguous_offsets() -> None:
    with pytest.raises(ValueError):
        InfosetLayout(action_counts=(2, 2), offsets=(0, 3), total_actions=4)

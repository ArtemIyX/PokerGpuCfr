from __future__ import annotations

import math

import numpy as np

from pokergpu.cfr import toy_expected_value, train_toy_mccfr


def test_toy_mccfr_expected_value_is_zero() -> None:
    result = train_toy_mccfr(iterations=256, seed=7)

    assert math.isclose(toy_expected_value(), 0.0, abs_tol=1e-6)
    assert math.isfinite(result.expected_value_p0)
    assert result.store.regrets.shape == (4,)
    assert result.store.strategy_sums.shape == (4,)
    assert np.any(result.store.regrets != 0.0)

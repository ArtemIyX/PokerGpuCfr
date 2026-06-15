from __future__ import annotations

import pytest

from pokergpu.cfr.stage7 import (
    DenseCfrState,
    regret_matching,
    update_average_strategy,
    update_regret,
)


def test_regret_matching_uses_positive_regrets() -> None:
    assert regret_matching((-1.0, 3.0, 1.0)) == (0.0, 0.75, 0.25)


def test_regret_matching_falls_back_to_uniform() -> None:
    assert regret_matching((0.0, -2.0)) == (0.5, 0.5)


def test_update_regret_adds_counterfactual_differences() -> None:
    assert update_regret((1.0, -2.0), (4.0, 1.0), 2.0) == (3.0, -3.0)


def test_update_average_strategy_accumulates_reach_weight() -> None:
    assert update_average_strategy((1.0, 2.0), (0.25, 0.75), 4.0) == (2.0, 5.0)


def test_dense_cfr_state_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        DenseCfrState(regret_sums=((0.0, 1.0),), strategy_sums=((0.0,),))

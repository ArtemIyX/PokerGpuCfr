import math

import numpy as np
import pytest

from pokergpu.abstraction.hands import (
    PlayerRangeVectors,
    RangeVector,
    private_hand_count,
)
from pokergpu.core.betting import (
    BettingRoundState,
    BlindStructure,
    PlayerBet,
    PlayerIndex,
    PlayerStack,
    Pot,
    chips,
)
from pokergpu.core.board import Board
from pokergpu.core.state import GameState, HandPhase, PlayerState
from pokergpu.eval.device import EvalDeviceConfig
from pokergpu.value_network import (
    PokerValueTarget,
    ValueFeatureSpec,
    ValueTargetKind,
    build_value_feature_batch,
    build_value_label,
    feature_dimension,
    scalar_ev_target,
)


def make_state() -> GameState:
    return GameState(
        board=Board.from_str("AhKdQc"),
        players=(
            PlayerState(player=PlayerIndex(0), hole_cards=None),
            PlayerState(player=PlayerIndex(1), hole_cards=None, folded=True),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(300)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(900)),
                PlayerStack(player=PlayerIndex(1), stack=chips(800)),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(100)),
                PlayerBet(player=PlayerIndex(1), committed=chips(100), folded=True),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        ),
        phase=HandPhase.IN_PROGRESS,
        dealer=PlayerIndex(1),
    )


def make_ranges() -> PlayerRangeVectors:
    values = [0.0] * private_hand_count()
    values[0] = 1.0
    range_vector = RangeVector.from_values(values)
    return PlayerRangeVectors.from_values((range_vector, range_vector))


def test_scalar_ev_target_defaults_to_one_value_per_player() -> None:
    target = scalar_ev_target(2)

    assert target.kind is ValueTargetKind.SCALAR_EV
    assert target.bucket_count == 1


def test_scalar_ev_target_rejects_invalid_player_count() -> None:
    with pytest.raises(ValueError):
        scalar_ev_target(0)


def test_bucketed_target_requires_multiple_buckets() -> None:
    with pytest.raises(ValueError):
        PokerValueTarget(kind=ValueTargetKind.BUCKETED_VALUE, 
                         player_count=2, 
                         bucket_count=1)


def test_feature_dimension_scales_with_ranges_and_player_mask() -> None:
    spec = ValueFeatureSpec(player_count=2, max_history_length=8)

    assert feature_dimension(spec) == 1 + 1 + 5 + 2 + (2 * 1326) + 2 + 8


def test_feature_batch_builds_expected_shape() -> None:
    state = make_state()
    ranges = make_ranges()
    spec = ValueFeatureSpec(player_count=2, max_history_length=8)

    batch = build_value_feature_batch((state,), (ranges,), spec)

    assert batch.values.shape == (1, feature_dimension(spec))
    assert batch.values.dtype == np.float32
    assert math.isclose(float(batch.values[0, 0]), 1.0, abs_tol=1e-6)


def test_feature_batch_rejects_mismatched_inputs() -> None:
    state = make_state()
    ranges = make_ranges()
    spec = ValueFeatureSpec(player_count=2)

    with pytest.raises(ValueError):
        build_value_feature_batch((), (ranges,), spec)
    with pytest.raises(ValueError):
        build_value_feature_batch((state,), (), spec)


def test_feature_batch_accepts_cuda_request_when_available() -> None:
    state = make_state()
    ranges = make_ranges()
    spec = ValueFeatureSpec(player_count=2)

    batch = build_value_feature_batch(
        (state,),
        (ranges,),
        spec,
        device_config=EvalDeviceConfig(mode="auto"),
    )

    assert batch.batch_size == 1


def test_scalar_ev_labels_use_one_row_per_sample() -> None:
    target = scalar_ev_target(2)

    label = build_value_label([1.5, -1.5], target)

    assert label.kind is ValueTargetKind.SCALAR_EV
    assert label.values.shape == (1, 2)
    assert math.isclose(float(label.values[0, 0]), 1.5, abs_tol=1e-6)


def test_bucketed_labels_preserve_bucket_axis() -> None:
    target = PokerValueTarget(kind=ValueTargetKind.BUCKETED_VALUE, 
                              player_count=2, 
                              bucket_count=3)

    label = build_value_label(
        np.asarray([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]], 
                   dtype=np.float32), 
        target)

    assert label.values.shape == (2, 3)
    assert math.isclose(float(label.values[1, 2]), 0.4, abs_tol=1e-6)


def test_bucketed_labels_reject_wrong_shape() -> None:
    target = PokerValueTarget(kind=ValueTargetKind.BUCKETED_VALUE, 
                              player_count=2, 
                              bucket_count=3)

    with pytest.raises(ValueError):
        build_value_label([0.1, 0.2], target)

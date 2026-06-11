import math
from pathlib import Path

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
from pokergpu.value_network import (
    PokerValueLabel,
    ValueFeatureBatch,
    ValueFeatureSpec,
    build_value_label,
    scalar_ev_target,
)
from pokergpu.value_network.dataset import (
    DatasetManifestEntry,
    DatasetSplitRule,
    FeatureNormalizer,
    ValueDatasetSample,
    build_dataset_manifest_entry,
    export_dataset_sample,
    export_value_sample,
    fit_feature_normalizer,
    load_dataset_manifest,
    load_feature_normalizer,
    load_value_sample,
    normalize_feature_batch,
    save_dataset_manifest,
    save_feature_normalizer,
    save_value_sample,
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


def make_label() -> PokerValueLabel:
    return build_value_label([1.25, -1.25], scalar_ev_target(2))


def test_export_value_sample_builds_sample() -> None:
    sample = export_value_sample(
        "spot-1",
        make_state(),
        make_ranges(),
        make_label(),
        ValueFeatureSpec(player_count=2),
        metadata={"street": "flop"},
    )

    assert sample.sample_id == "spot-1"
    assert sample.features.ndim == 1
    assert sample.label.shape == (1, 2)
    assert sample.metadata["street"] == "flop"


def test_value_dataset_sample_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError):
        ValueDatasetSample(
            sample_id="",
            features=np.zeros((2, 2), dtype=np.float32),
            label=np.zeros((1, 2), dtype=np.float32),
            metadata={},
        )


def test_split_rule_is_deterministic() -> None:
    rule = DatasetSplitRule(validation_modulo=5, validation_remainder=2)

    assert rule.split("spot-1") in {"train", "val"}
    assert rule.split("spot-1") == rule.split("spot-1")


def test_fit_feature_normalizer_produces_positive_std() -> None:
    sample_a = ValueDatasetSample(
        sample_id="a",
        features=np.asarray([1.0, 3.0], dtype=np.float32),
        label=np.asarray([[0.0]], dtype=np.float32),
        metadata={},
    )
    sample_b = ValueDatasetSample(
        sample_id="b",
        features=np.asarray([3.0, 7.0], dtype=np.float32),
        label=np.asarray([[0.0]], dtype=np.float32),
        metadata={},
    )

    normalizer = fit_feature_normalizer([sample_a, sample_b])

    assert normalizer.feature_count == 2
    assert math.isclose(float(normalizer.mean[0]), 2.0, abs_tol=1e-6)
    assert float(normalizer.std[0]) > 0.0


def test_fit_label_normalizer_clamps_scale() -> None:
    from pokergpu.value_network.dataset import fit_label_normalizer

    sample_a = ValueDatasetSample(
        sample_id="a",
        features=np.asarray([1.0, 3.0], dtype=np.float32),
        label=np.asarray([[100.0, -100.0]], dtype=np.float32),
        metadata={},
    )
    sample_b = ValueDatasetSample(
        sample_id="b",
        features=np.asarray([3.0, 7.0], dtype=np.float32),
        label=np.asarray([[200.0, -200.0]], dtype=np.float32),
        metadata={},
    )

    normalizer = fit_label_normalizer([sample_a, sample_b])

    assert float(normalizer.std[0]) >= 1000.0


def test_normalize_feature_batch_round_trips_shape() -> None:
    sample = ValueDatasetSample(
        sample_id="a",
        features=np.asarray([1.0, 3.0], dtype=np.float32),
        label=np.asarray([[0.0]], dtype=np.float32),
        metadata={},
    )
    normalizer = FeatureNormalizer(
        mean=np.asarray([1.0, 2.0], dtype=np.float32),
        std=np.asarray([1.0, 2.0], dtype=np.float32),
    )

    batch = normalize_feature_batch(
        ValueFeatureBatch(np.asarray([sample.features], dtype=np.float32)),
        normalizer,
    )

    assert batch.values.shape == (1, 2)
    assert math.isclose(float(batch.values[0, 1]), 0.5, abs_tol=1e-6)


def test_save_and_load_value_sample(tmp_path: Path) -> None:
    sample = ValueDatasetSample(
        sample_id="spot-1",
        features=np.asarray([1.0, 2.0], dtype=np.float32),
        label=np.asarray([[3.0, 4.0]], dtype=np.float32),
        metadata={"a": 1},
    )
    path = tmp_path / "spot-1.npz"

    save_value_sample(sample, path)
    loaded = load_value_sample(path)

    assert loaded.sample_id == "spot-1"
    assert math.isclose(float(loaded.features[1]), 2.0, abs_tol=1e-6)
    assert math.isclose(float(loaded.label[0, 1]), 4.0, abs_tol=1e-6)


def test_save_and_load_normalizer(tmp_path: Path) -> None:
    normalizer = FeatureNormalizer(
        mean=np.asarray([1.0, 2.0], dtype=np.float32),
        std=np.asarray([3.0, 4.0], dtype=np.float32),
    )
    path = tmp_path / "stats.json"

    save_feature_normalizer(normalizer, path)
    loaded = load_feature_normalizer(path)

    assert math.isclose(float(loaded.mean[0]), 1.0, abs_tol=1e-6)
    assert math.isclose(float(loaded.std[1]), 4.0, abs_tol=1e-6)


def test_save_and_load_manifest(tmp_path: Path) -> None:
    entries = [
        DatasetManifestEntry(
            sample_id="spot-1",
            split="train",
            path="train/spot-1.npz",
            feature_count=10,
            label_shape=(1, 2),
        )
    ]
    path = tmp_path / "manifest.json"

    save_dataset_manifest(entries, path)
    loaded = load_dataset_manifest(path)

    assert loaded[0].sample_id == "spot-1"
    assert loaded[0].path == "train/spot-1.npz"


def test_export_dataset_sample_writes_expected_split_path(tmp_path: Path) -> None:
    rule = DatasetSplitRule(validation_modulo=2, validation_remainder=0)
    entry = export_dataset_sample(
        "spot-1",
        make_state(),
        make_ranges(),
        make_label(),
        ValueFeatureSpec(player_count=2),
        tmp_path,
        rule,
        metadata={"street": "flop"},
    )

    assert entry.split in {"train", "val"}
    assert (tmp_path / entry.path).exists()


def test_build_dataset_manifest_entry_uses_sample_shapes() -> None:
    sample = ValueDatasetSample(
        sample_id="spot-1",
        features=np.asarray([1.0, 2.0], dtype=np.float32),
        label=np.asarray([[3.0, 4.0]], dtype=np.float32),
        metadata={},
    )
    entry = build_dataset_manifest_entry(sample, DatasetSplitRule(), "train/spot-1.npz")

    assert entry.feature_count == 2
    assert entry.label_shape == (1, 2)


def test_curated_solver_spots_cover_multiple_textures() -> None:
    from pokergpu.value_network.dataset import curated_solver_spots

    spots = curated_solver_spots()
    textures = {spot.board_texture for spot in spots}
    streets = {spot.state.current_street.value for spot in spots}
    actions = {spot.action_line for spot in spots}

    assert len(spots) >= 12
    assert {"dry", "wet", "paired"}.issubset(textures)
    assert {"flop", "turn", "river"}.issubset(streets)
    assert {"check-check", "c-bet", "bet-call", "bet-fold"}.issubset(actions)

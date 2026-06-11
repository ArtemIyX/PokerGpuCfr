from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from pokergpu.abstraction.hands import PlayerRangeVectors
from pokergpu.core.betting import BettingRoundState, BlindStructure, PlayerBet, PlayerIndex, PlayerStack, Pot, chips
from pokergpu.core.board import Board
from pokergpu.core.state import GameState, PlayerState

from .target import (
    PokerValueLabel,
    ValueFeatureBatch,
    ValueFeatureSpec,
    build_value_feature_batch,
)

__all__ = [
    "DATASET_VERSION",
    "DatasetManifestEntry",
    "DatasetManifestEntryPayload",
    "DatasetSplitRule",
    "FeatureNormalizer",
    "FeatureNormalizerPayload",
    "LabelNormalizer",
    "LabelNormalizerPayload",
    "ValueDatasetSample",
    "ValueFeatureBatch",
    "curated_solver_spots",
    "CuratedSpot",
    "build_dataset_manifest_entry",
    "export_dataset_sample",
    "export_value_sample",
    "fit_feature_normalizer",
    "load_dataset_manifest",
    "load_feature_normalizer",
    "load_value_sample",
    "normalize_feature_batch",
    "save_dataset_manifest",
    "save_feature_normalizer",
    "save_value_sample",
]

DATASET_VERSION = 1


class FeatureNormalizerPayload(TypedDict):
    version: int
    mean: list[float]
    std: list[float]


class LabelNormalizerPayload(TypedDict):
    version: int
    mean: list[float]
    std: list[float]


class DatasetManifestEntryPayload(TypedDict):
    sample_id: str
    split: str
    path: str
    feature_count: int
    label_shape: list[int]


@dataclass(frozen=True, slots=True)
class CuratedSpot:
    state: GameState
    board_texture: str
    pot_bucket: str
    stack_bucket: str
    action_line: str


@dataclass(frozen=True, slots=True)
class ValueDatasetSample:
    sample_id: str
    features: NDArray[np.float32]
    label: NDArray[np.float32]
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample id must not be empty")
        if self.features.ndim != 1:
            raise ValueError("sample features must be one-dimensional")
        if self.label.ndim != 2:
            raise ValueError("sample label must be two-dimensional")
        if self.features.dtype != np.float32:
            raise ValueError("sample features must use float32")
        if self.label.dtype != np.float32:
            raise ValueError("sample label must use float32")


@dataclass(frozen=True, slots=True)
class DatasetSplitRule:
    validation_modulo: int = 10
    validation_remainder: int = 0

    def __post_init__(self) -> None:
        if self.validation_modulo <= 1:
            raise ValueError("validation modulo must be greater than one")
        if not 0 <= self.validation_remainder < self.validation_modulo:
            raise ValueError("validation remainder out of range")

    def is_validation(self, sample_id: str) -> bool:
        return (_sample_hash(sample_id) % self.validation_modulo 
                == self.validation_remainder)

    def split(self, sample_id: str) -> str:
        return "val" if self.is_validation(sample_id) else "train"


@dataclass(frozen=True, slots=True)
class FeatureNormalizer:
    mean: NDArray[np.float32]
    std: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.std.ndim != 1:
            raise ValueError("normalizer vectors must be one-dimensional")
        if self.mean.shape != self.std.shape:
            raise ValueError("normalizer mean and std must match")
        if np.any(self.std <= 0):
            raise ValueError("normalizer std must be positive")
        if self.mean.dtype != np.float32 or self.std.dtype != np.float32:
            raise ValueError("normalizer arrays must use float32")

    @property
    def feature_count(self) -> int:
        return int(self.mean.shape[0])

    def normalize(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        if features.ndim != 1:
            raise ValueError("features must be one-dimensional")
        if features.shape[0] != self.feature_count:
            raise ValueError("feature length mismatch")
        return (features - self.mean) / self.std

    def denormalize(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        if features.ndim != 1:
            raise ValueError("features must be one-dimensional")
        if features.shape[0] != self.feature_count:
            raise ValueError("feature length mismatch")
        return features * self.std + self.mean

    def to_json(self) -> FeatureNormalizerPayload:
        return {
            "version": DATASET_VERSION,
            "mean": [float(value) for value in self.mean.tolist()],
            "std": [float(value) for value in self.std.tolist()],
        }

    @classmethod
    def from_json(cls, payload: FeatureNormalizerPayload) -> FeatureNormalizer:
        if payload["version"] != DATASET_VERSION:
            raise ValueError("unsupported normalizer version")
        mean = np.asarray(payload["mean"], dtype=np.float32)
        std = np.asarray(payload["std"], dtype=np.float32)
        return cls(mean=mean, std=std)


@dataclass(frozen=True, slots=True)
class LabelNormalizer:
    mean: NDArray[np.float32]
    std: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.std.ndim != 1:
            raise ValueError("label normalizer vectors must be one-dimensional")
        if self.mean.shape != self.std.shape:
            raise ValueError("label normalizer mean and std must match")
        if np.any(self.std <= 0):
            raise ValueError("label normalizer std must be positive")
        if self.mean.dtype != np.float32 or self.std.dtype != np.float32:
            raise ValueError("label normalizer arrays must use float32")

    @property
    def label_count(self) -> int:
        return int(self.mean.shape[0])

    def normalize(self, labels: NDArray[np.float32]) -> NDArray[np.float32]:
        if labels.ndim != 2:
            raise ValueError("labels must be two-dimensional")
        if labels.shape[1] != self.label_count:
            raise ValueError("label length mismatch")
        return (labels - self.mean) / self.std

    def denormalize(self, labels: NDArray[np.float32]) -> NDArray[np.float32]:
        if labels.ndim != 2:
            raise ValueError("labels must be two-dimensional")
        if labels.shape[1] != self.label_count:
            raise ValueError("label length mismatch")
        return labels * self.std + self.mean

    def to_json(self) -> LabelNormalizerPayload:
        return {
            "version": DATASET_VERSION,
            "mean": [float(value) for value in self.mean.tolist()],
            "std": [float(value) for value in self.std.tolist()],
        }

    @classmethod
    def from_json(cls, payload: LabelNormalizerPayload) -> LabelNormalizer:
        if payload["version"] != DATASET_VERSION:
            raise ValueError("unsupported normalizer version")
        mean = np.asarray(payload["mean"], dtype=np.float32)
        std = np.asarray(payload["std"], dtype=np.float32)
        return cls(mean=mean, std=std)


@dataclass(frozen=True, slots=True)
class DatasetManifestEntry:
    sample_id: str
    split: str
    path: str
    feature_count: int
    label_shape: tuple[int, int]

    def to_dict(self) -> DatasetManifestEntryPayload:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "path": self.path,
            "feature_count": self.feature_count,
            "label_shape": list(self.label_shape),
        }

    @classmethod
    def from_dict(cls, payload: DatasetManifestEntryPayload) -> DatasetManifestEntry:
        shape = payload["label_shape"]
        if len(shape) != 2:
            raise ValueError("label shape must contain two dimensions")
        return cls(
            sample_id=payload["sample_id"],
            split=payload["split"],
            path=payload["path"],
            feature_count=payload["feature_count"],
            label_shape=(shape[0], shape[1]),
        )


def _sample_hash(sample_id: str) -> int:
    digest = blake2b(sample_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def normalize_feature_batch(
    batch: ValueFeatureBatch,
    normalizer: FeatureNormalizer,
) -> ValueFeatureBatch:
    normalized_rows = np.asarray(
        [normalizer.normalize(row) for row in batch.values],
        dtype=np.float32,
    )
    return ValueFeatureBatch(normalized_rows)


def fit_feature_normalizer(samples: list[ValueDatasetSample]) -> FeatureNormalizer:
    if not samples:
        raise ValueError("samples must not be empty")
    matrix = np.vstack([sample.features for sample in samples]).astype(np.float32)
    mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    variance = matrix.var(axis=0, dtype=np.float64).astype(np.float32)
    std = np.sqrt(np.maximum(variance, np.float32(1e-8))).astype(np.float32)
    return FeatureNormalizer(mean=mean, std=std)


def fit_label_normalizer(samples: list[ValueDatasetSample]) -> LabelNormalizer:
    if not samples:
        raise ValueError("samples must not be empty")
    matrix = np.concatenate([sample.label for sample in samples], axis=0).astype(
        np.float32
    )
    mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    variance = matrix.var(axis=0, dtype=np.float64).astype(np.float32)
    std = np.sqrt(np.maximum(variance, np.float32(1e-8))).astype(np.float32)
    return LabelNormalizer(mean=mean, std=std)


def export_value_sample(
    sample_id: str,
    state: GameState,
    ranges: PlayerRangeVectors,
    label: PokerValueLabel,
    feature_spec: ValueFeatureSpec,
    metadata: dict[str, object] | None = None,
) -> ValueDatasetSample:
    features = build_value_feature_batch((state,), (ranges,), feature_spec).values[0]
    return ValueDatasetSample(
        sample_id=sample_id,
        features=features.copy(),
        label=label.values.copy(),
        metadata=dict(metadata or {}),
    )


def save_value_sample(sample: ValueDatasetSample, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_id=np.array(sample.sample_id),
        features=sample.features,
        label=sample.label,
        metadata=np.array(json.dumps(sample.metadata, sort_keys=True)),
    )


def load_value_sample(path: Path) -> ValueDatasetSample:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        return ValueDatasetSample(
            sample_id=str(data["sample_id"].item()),
            features=np.asarray(data["features"], dtype=np.float32),
            label=np.asarray(data["label"], dtype=np.float32),
            metadata=metadata,
        )


def save_dataset_manifest(entries: list[DatasetManifestEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [entry.to_dict() for entry in entries]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_dataset_manifest(path: Path) -> list[DatasetManifestEntry]:
    payload = cast(list[DatasetManifestEntryPayload], 
                   json.loads(path.read_text(encoding="utf-8")))
    return [DatasetManifestEntry.from_dict(item) for item in payload]


def save_feature_normalizer(normalizer: FeatureNormalizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalizer.to_json(), 
                               indent=2, 
                               sort_keys=True), 
                    encoding="utf-8")


def load_feature_normalizer(path: Path) -> FeatureNormalizer:
    payload = cast(FeatureNormalizerPayload, 
                   json.loads(path.read_text(encoding="utf-8")))
    return FeatureNormalizer.from_json(payload)


def save_label_normalizer(normalizer: LabelNormalizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalizer.to_json(), indent=2, sort_keys=True), encoding="utf-8")


def load_label_normalizer(path: Path) -> LabelNormalizer:
    payload = cast(LabelNormalizerPayload, json.loads(path.read_text(encoding="utf-8")))
    return LabelNormalizer.from_json(payload)


def build_dataset_manifest_entry(
    sample: ValueDatasetSample,
    split_rule: DatasetSplitRule,
    relative_path: str,
) -> DatasetManifestEntry:
    return DatasetManifestEntry(
        sample_id=sample.sample_id,
        split=split_rule.split(sample.sample_id),
        path=relative_path,
        feature_count=int(sample.features.shape[0]),
        label_shape=(int(sample.label.shape[0]), int(sample.label.shape[1])),
    )


def export_dataset_sample(
    sample_id: str,
    state: GameState,
    ranges: PlayerRangeVectors,
    label: PokerValueLabel,
    feature_spec: ValueFeatureSpec,
    output_dir: Path,
    split_rule: DatasetSplitRule,
    metadata: dict[str, object] | None = None,
) -> DatasetManifestEntry:
    sample = export_value_sample(sample_id, 
                                 state, 
                                 ranges, 
                                 label, 
                                 feature_spec, 
                                 metadata)
    split = split_rule.split(sample.sample_id)
    file_name = f"{sample.sample_id}.npz"
    relative_path = f"{split}/{file_name}"
    save_value_sample(sample, output_dir / relative_path)
    return build_dataset_manifest_entry(sample, split_rule, relative_path)


def curated_solver_spots() -> tuple[CuratedSpot, ...]:
    return (
        _make_spot("AhKdQc", "flop", "small", "shallow", "check-check"),
        _make_spot("9hThJh", "flop", "medium", "deep", "c-bet"),
        _make_spot("8s8d8h", "flop", "large", "deep", "check-raise"),
        _make_spot("2c7d9s", "flop", "small", "medium", "bet-call"),
        _make_spot("AhQh2h", "flop", "medium", "deep", "bet-fold"),
        _make_spot("KcKd2s", "flop", "large", "shallow", "check-bet"),
        _make_spot("AhKdQcJd", "turn", "small", "shallow", "c-bet"),
        _make_spot("9hThJh2c", "turn", "medium", "deep", "check"),
        _make_spot("8s8d8h2d", "turn", "large", "deep", "c-bet"),
        _make_spot("2c7d9sTd", "turn", "small", "medium", "check-raise"),
        _make_spot("AhQh2hJs", "turn", "medium", "shallow", "bet-call"),
        _make_spot("KcKd2s7h", "turn", "large", "deep", "bet-fold"),
        _make_spot("AhKdQcJd9s", "river", "small", "shallow", "check"),
        _make_spot("9hThJh2c3d", "river", "medium", "deep", "c-bet"),
        _make_spot("8s8d8h2d3c", "river", "large", "deep", "check"),
        _make_spot("2c7d9sTdKh", "river", "small", "medium", "bet-call"),
        _make_spot("AhQh2hJs9d", "river", "medium", "shallow", "check-raise"),
        _make_spot("KcKd2s7h4c", "river", "large", "deep", "bet-fold"),
    )


def _make_spot(
    board_text: str,
    street: str,
    pot_bucket: str,
    stack_bucket: str,
    action_line: str,
) -> CuratedSpot:
    board = Board.from_str(board_text)
    if board.street.value != street:
        raise ValueError("board street mismatch")
    pot_amount = {
        "small": chips(100),
        "medium": chips(300),
        "large": chips(800),
    }[pot_bucket]
    stack_amount = {
        "shallow": chips(3000),
        "medium": chips(6000),
        "deep": chips(10000),
    }[stack_bucket]
    state = GameState(
        board=board,
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=pot_amount),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=stack_amount),
                PlayerStack(player=PlayerIndex(1), stack=stack_amount),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(0)),
                PlayerBet(player=PlayerIndex(1), committed=chips(0)),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        ),
        dealer=PlayerIndex(0),
    )
    return CuratedSpot(
        state=state,
        board_texture="paired" if _is_paired(board) else _board_texture(board),
        pot_bucket=pot_bucket,
        stack_bucket=stack_bucket,
        action_line=action_line,
    )


def _board_texture(board: Board) -> str:
    ranks = [card.rank for card in board.cards]
    unique = len(set(ranks))
    suits = [card.suit for card in board.cards]
    if len(set(suits)) == 1:
        return "monotone"
    if max((ranks.count(rank) for rank in set(ranks)), default=0) >= 2:
        return "paired"
    if any(abs(a.order_value - b.order_value) <= 2 for a in ranks for b in ranks if a != b):
        return "connected"
    if unique == len(ranks):
        return "dry"
    return "wet"


def _is_paired(board: Board) -> bool:
    ranks = [card.rank for card in board.cards]
    return len(set(ranks)) < len(ranks)

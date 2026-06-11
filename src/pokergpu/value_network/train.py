from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .checkpoint import ValueCheckpoint, save_checkpoint
from .dataset import (
    DatasetManifestEntry,
    DatasetSplitRule,
    FeatureNormalizer,
    ValueDatasetSample,
    ValueFeatureBatch,
    fit_feature_normalizer,
    load_dataset_manifest,
    load_value_sample,
    normalize_feature_batch,
)
from .model import (
    ValueMLP,
    build_value_model,
    build_value_network_config,
    infer_value,
    train_value_step,
)
from .target import ValueFeatureSpec, ValueTargetKind

if TYPE_CHECKING:
    pass


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 1e-3
    validation_modulo: int = 10
    validation_remainder: int = 0
    amp: bool = False

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning rate must be positive")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: ValueMLP
    checkpoint: ValueCheckpoint
    train_loss: float
    val_loss: float
    predictions: NDArray[np.float32]
    labels: NDArray[np.float32]


def _load_samples(
    entries: list[DatasetManifestEntry],
    dataset_dir: Path,
) -> list[ValueDatasetSample]:
    return [load_value_sample(dataset_dir / entry.path) for entry in entries]


def _split_entries(
    entries: list[DatasetManifestEntry],
    rule: DatasetSplitRule,
) -> tuple[list[DatasetManifestEntry], list[DatasetManifestEntry]]:
    train_entries: list[DatasetManifestEntry] = []
    val_entries: list[DatasetManifestEntry] = []
    for entry in entries:
        if rule.is_validation(entry.sample_id):
            val_entries.append(entry)
        else:
            train_entries.append(entry)
    return train_entries, val_entries


def _batch_slice(samples: list[ValueDatasetSample], 
                 start: int, 
                 stop: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    feature_rows = [sample.features for sample in samples[start:stop]]
    label_rows = [sample.label[0] for sample in samples[start:stop]]
    return (
        np.asarray(feature_rows, dtype=np.float32),
        np.asarray(label_rows, dtype=np.float32),
    )


def _evaluate_loss(
    model: ValueMLP,
    samples: list[ValueDatasetSample],
    normalizer: FeatureNormalizer,
    batch_size: int,
) -> float:
    if not samples:
        return 0.0
    losses: list[float] = []
    for start in range(0, len(samples), batch_size):
        stop = min(start + batch_size, len(samples))
        features, targets = _batch_slice(samples, start, stop)
        normalized = normalize_feature_batch(
            ValueFeatureBatch(features),
            normalizer,
        ).values
        prediction = infer_value(model, normalized)
        losses.append(float(np.mean((prediction - targets) ** 2)))
    return float(np.mean(losses, dtype=np.float64))


def train_baseline(
    manifest_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    feature_spec: ValueFeatureSpec,
    target_kind: ValueTargetKind,
    config: TrainingConfig | None = None,
    normalizer: FeatureNormalizer | None = None,
) -> TrainingResult:
    if config is None:
        config = TrainingConfig()

    entries = load_dataset_manifest(manifest_path)
    split_rule = DatasetSplitRule(
        validation_modulo=config.validation_modulo,
        validation_remainder=config.validation_remainder,
    )
    train_entries, val_entries = _split_entries(entries, split_rule)
    train_samples = _load_samples(train_entries, dataset_dir)
    val_samples = _load_samples(val_entries, dataset_dir)
    if not train_samples:
        raise ValueError("training split is empty")

    if normalizer is None:
        normalizer = fit_feature_normalizer(train_samples)

    model_config = build_value_network_config(feature_spec, target_kind)
    model = build_value_model(model_config, device="cpu")
    import torch
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    best_val = float("inf")
    best_checkpoint: ValueCheckpoint | None = None
    train_loss = 0.0
    val_loss = 0.0

    for epoch in range(config.epochs):
        epoch_losses: list[float] = []
        for start in range(0, len(train_samples), config.batch_size):
            stop = min(start + config.batch_size, len(train_samples))
            features, targets = _batch_slice(train_samples, start, stop)
            normalized = normalize_feature_batch(
                ValueFeatureBatch(features),
                normalizer,
            ).values
            epoch_losses.append(
                train_value_step(
                    model,
                    optimizer,
                    normalized,
                    targets,
                    amp=config.amp,
                )
            )
        train_loss = float(np.mean(epoch_losses, dtype=np.float64))
        val_loss = _evaluate_loss(model, val_samples, normalizer, config.batch_size)
        if val_loss <= best_val:
            best_val = val_loss
            best_checkpoint = ValueCheckpoint(
                step=epoch + 1,
                best_metric=val_loss,
                feature_spec=feature_spec,
                target_kind=target_kind,
                target_bucket_count=model_config.output_dim,
                model_config=model_config,
                normalizer=normalizer,
            )
            save_checkpoint(output_dir / "best_checkpoint.pt", 
                            model, 
                            optimizer, 
                            best_checkpoint)

    if best_checkpoint is None:
        best_checkpoint = ValueCheckpoint(
            step=config.epochs,
            best_metric=val_loss,
            feature_spec=feature_spec,
            target_kind=target_kind,
            target_bucket_count=model_config.output_dim,
            model_config=model_config,
            normalizer=normalizer,
        )

    preview_predictions = np.zeros((0, model_config.output_dim), dtype=np.float32)
    preview_labels = np.zeros((0, model_config.output_dim), dtype=np.float32)
    if val_samples:
        features, targets = _batch_slice(val_samples, 
                                         0, 
                                         min(len(val_samples), 
                                             config.batch_size))
        normalized = normalize_feature_batch(
            ValueFeatureBatch(features),
            normalizer,
        ).values
        preview_predictions = infer_value(model, normalized)
        preview_labels = targets

    return TrainingResult(
        model=model,
        checkpoint=best_checkpoint,
        train_loss=train_loss,
        val_loss=val_loss,
        predictions=preview_predictions,
        labels=preview_labels,
    )

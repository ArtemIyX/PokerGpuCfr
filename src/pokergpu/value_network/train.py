from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable
import time

import numpy as np
from numpy.typing import NDArray

from .checkpoint import ValueCheckpoint, save_checkpoint
from .dataset import (
    DatasetPackManifestEntry,
    DatasetSplitRule,
    FeatureNormalizer,
    LabelNormalizer,
    ValueDatasetSample,
    ValueFeatureBatch,
    fit_feature_normalizer,
    fit_label_normalizer,
    load_dataset_manifest,
    load_value_sample_pack,
    load_value_sample,
    normalize_feature_batch,
)
from .model import (
    ValueMLP,
    build_value_model,
    build_value_network_config,
    default_value_device,
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
    hidden_dim: int = 512
    hidden_layers: int = 6
    dropout: float = 0.0
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
        if self.hidden_dim <= 0:
            raise ValueError("hidden dim must be positive")
        if self.hidden_layers <= 0:
            raise ValueError("hidden layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: ValueMLP
    checkpoint: ValueCheckpoint
    train_loss: float
    val_loss: float
    predictions: NDArray[np.float32]
    labels: NDArray[np.float32]


def _load_samples(
    entries: list[DatasetPackManifestEntry],
    dataset_dir: Path,
    progress_callback: Callable[[int, int, str], None] | None = None,
    label: str = "load samples",
) -> list[ValueDatasetSample]:
    samples: list[ValueDatasetSample] = []
    seen: set[Path] = set()
    pack_paths: list[Path] = []
    sample_paths: list[Path] = []
    for entry in entries:
        candidate_pack = (
            dataset_dir / entry.pack_path
            if entry.pack_path
            else dataset_dir / f"{entry.split}.pack.npz"
        )
        candidate_sample = dataset_dir / entry.path if entry.path else None
        chosen = None
        if candidate_pack.exists():
            chosen = candidate_pack
        elif candidate_sample is not None and candidate_sample.exists():
            chosen = candidate_sample
        if chosen is not None and chosen not in seen:
            seen.add(chosen)
            if chosen.name.endswith(".pack.npz"):
                pack_paths.append(chosen)
            else:
                sample_paths.append(chosen)
    for index, pack_path in enumerate(pack_paths, start=1):
        samples.extend(load_value_sample_pack(pack_path))
        if progress_callback is not None:
            progress_callback(index, len(pack_paths), label)
    for index, sample_path in enumerate(sample_paths, start=len(pack_paths) + 1):
        samples.append(load_value_sample(sample_path))
        if progress_callback is not None:
            progress_callback(index, len(pack_paths) + len(sample_paths), label)
    return samples


def _split_entries(
    entries: list[DatasetPackManifestEntry],
    rule: DatasetSplitRule,
) -> tuple[list[DatasetPackManifestEntry], list[DatasetPackManifestEntry]]:
    train_entries: list[DatasetPackManifestEntry] = []
    val_entries: list[DatasetPackManifestEntry] = []
    for entry in entries:
        if entry.split == "val":
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
    label_normalizer: LabelNormalizer,
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
        targets = label_normalizer.normalize(targets)
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
    feature_normalizer: FeatureNormalizer | None = None,
    label_normalizer: LabelNormalizer | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> TrainingResult:
    if config is None:
        config = TrainingConfig()
    print("train: load manifest")

    entries = load_dataset_manifest(manifest_path)
    split_rule = DatasetSplitRule(
        validation_modulo=config.validation_modulo,
        validation_remainder=config.validation_remainder,
    )
    train_entries, val_entries = _split_entries(entries, split_rule)
    print(
        "train: dataset "
        f"pack_files={len({entry.pack_path for entry in entries if entry.pack_path})} "
        f"train_samples={sum(entry.sample_count for entry in train_entries)} "
        f"val_samples={sum(entry.sample_count for entry in val_entries)}"
    )
    print(f"train: load samples train={len(train_entries)} val={len(val_entries)}")
    train_samples = _load_samples(
        train_entries,
        dataset_dir,
        progress_callback=progress_callback,
        label="load train samples",
    )
    val_samples = _load_samples(
        val_entries,
        dataset_dir,
        progress_callback=progress_callback,
        label="load val samples",
    )
    print(
        "train: loaded "
        f"train_samples={len(train_samples)} "
        f"val_samples={len(val_samples)} "
        f"total_samples={len(train_samples) + len(val_samples)}"
    )
    if not train_samples:
        raise ValueError("training split is empty")

    if feature_normalizer is None:
        print("train: fit feature normalizer")
        feature_normalizer = fit_feature_normalizer(train_samples)
    if label_normalizer is None:
        print("train: fit label normalizer")
        label_normalizer = fit_label_normalizer(train_samples)
    if progress_callback is not None:
        progress_callback(1, 1, "normalizers ready")

    print("train: build model")
    model_config = build_value_network_config(
        feature_spec,
        target_kind,
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
        dropout=config.dropout,
    )
    device = default_value_device()
    print(f"train: device={device}")
    model = build_value_model(model_config, device=device)
    import torch
    print("train: build optimizer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    best_val = float("inf")
    best_checkpoint: ValueCheckpoint | None = None
    train_loss = 0.0
    val_loss = 0.0

    for epoch in range(config.epochs):
        epoch_started_at = time.monotonic()
        epoch_losses: list[float] = []
        batch_total = (len(train_samples) + config.batch_size - 1) // config.batch_size
        batch_index = 0
        for start in range(0, len(train_samples), config.batch_size):
            stop = min(start + config.batch_size, len(train_samples))
            features, targets = _batch_slice(train_samples, start, stop)
            normalized = normalize_feature_batch(
                ValueFeatureBatch(features),
                feature_normalizer,
            ).values
            targets = label_normalizer.normalize(targets)
            epoch_losses.append(
                train_value_step(
                    model,
                    optimizer,
                    normalized,
                    targets,
                    amp=config.amp,
                )
            )
            batch_index += 1
            if progress_callback is not None:
                progress_callback(batch_index, batch_total, f"train epoch {epoch + 1}")
        train_loss = float(np.mean(epoch_losses, dtype=np.float64))
        val_loss = _evaluate_loss(
            model,
            val_samples,
            feature_normalizer,
            label_normalizer,
            config.batch_size,
        )
        epoch_elapsed = time.monotonic() - epoch_started_at
        print(
            f"train: epoch={epoch + 1}/{config.epochs} "
            f"train_loss={train_loss:.9e} val_loss={val_loss:.9e} "
            f"elapsed_seconds={epoch_elapsed:.3f}"
        )
        if val_loss <= best_val:
            best_val = val_loss
            best_checkpoint = ValueCheckpoint(
                step=epoch + 1,
                best_metric=val_loss,
                feature_spec=feature_spec,
                target_kind=target_kind,
                target_bucket_count=model_config.output_dim,
                model_config=model_config,
                normalizer=feature_normalizer,
            )
            save_checkpoint(output_dir / "best_checkpoint.pt", 
                            model, 
                            optimizer, 
                            best_checkpoint)
            print("train: saved best checkpoint")

    if best_checkpoint is None:
        best_checkpoint = ValueCheckpoint(
            step=config.epochs,
            best_metric=val_loss,
            feature_spec=feature_spec,
            target_kind=target_kind,
            target_bucket_count=model_config.output_dim,
            model_config=model_config,
            normalizer=feature_normalizer,
        )

    preview_predictions = np.zeros((0, model_config.output_dim), dtype=np.float32)
    preview_labels = np.zeros((0, model_config.output_dim), dtype=np.float32)
    preview_mae = 0.0
    if val_samples:
        print("train: build preview")
        features, targets = _batch_slice(val_samples, 
                                         0, 
                                         min(len(val_samples), 
                                             config.batch_size))
        normalized = normalize_feature_batch(
            ValueFeatureBatch(features),
            feature_normalizer,
        ).values
        preview_predictions = label_normalizer.denormalize(
            infer_value(model, normalized)
        )
        preview_labels = targets
        preview_mae = float(np.mean(np.abs(preview_predictions - preview_labels), dtype=np.float64))
        print(f"train: preview_mae={preview_mae:.9e}")

    return TrainingResult(
        model=model,
        checkpoint=best_checkpoint,
        train_loss=train_loss,
        val_loss=val_loss,
        predictions=preview_predictions,
        labels=preview_labels,
    )

from pathlib import Path

import numpy as np
import pytest

from pokergpu.value_network import (
    DatasetManifestEntry,
    FeatureNormalizer,
    ValueDatasetSample,
    ValueFeatureSpec,
    ValueTargetKind,
    save_value_sample_pack,
    save_dataset_manifest,
)
from pokergpu.value_network.train import TrainingConfig, train_baseline


def test_train_baseline_runs_and_returns_losses(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "out"
    feature_spec = ValueFeatureSpec(player_count=2, max_history_length=8)
    entries = []

    normalizer = FeatureNormalizer(
        mean=np.zeros(1 + 1 + 5 + 2 + (2 * 1326) + 2 + 8, dtype=np.float32),
        std=np.ones(1 + 1 + 5 + 2 + (2 * 1326) + 2 + 8, dtype=np.float32),
    )
    for index in range(6):
        sample = ValueDatasetSample(
            sample_id=f"spot-{index}",
            features=np.full(normalizer.feature_count, float(index), dtype=np.float32),
            label=np.asarray([[float(index), float(-index)]], dtype=np.float32),
            metadata={},
        )
        split = "train" if index < 4 else "val"
        entries.append(
            DatasetManifestEntry(
                sample_id=sample.sample_id,
                split=split,
                path=f"{split}/{sample.sample_id}.npz",
                feature_count=sample.features.shape[0],
                label_shape=(1, 2),
            )
        )
        save_value_sample_pack([sample], dataset_dir / f"{split}.pack.npz")

    manifest_path = tmp_path / "manifest.json"
    save_dataset_manifest(entries, manifest_path)

    result = train_baseline(
        manifest_path=manifest_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        feature_spec=feature_spec,
        target_kind=ValueTargetKind.SCALAR_EV,
        config=TrainingConfig(
            epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            hidden_dim=768,
            hidden_layers=8,
        ),
        feature_normalizer=normalizer,
    )

    assert result.train_loss >= 0.0
    assert result.val_loss >= 0.0
    assert result.checkpoint.best_metric >= 0.0
    assert result.checkpoint.model_config.hidden_dim == 768
    assert result.checkpoint.model_config.hidden_layers == 8
    assert result.predictions.shape[1] == 2
    assert result.labels.shape[1] == 2
    assert (output_dir / "best_checkpoint.pt").exists()

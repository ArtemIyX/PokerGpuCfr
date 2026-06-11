import math
from pathlib import Path

import numpy as np
import pytest

from pokergpu.value_network import (
    FeatureNormalizer,
    ValueFeatureSpec,
    ValueTargetKind,
    build_value_network_config,
)
from pokergpu.value_network.checkpoint import (
    ValueCheckpoint,
    load_checkpoint,
    save_checkpoint,
)
from pokergpu.value_network.model import (
    build_value_model,
    infer_value,
    train_value_step,
)


def test_value_network_config_uses_feature_spec() -> None:
    spec = ValueFeatureSpec(player_count=2, max_history_length=8)
    config = build_value_network_config(spec, ValueTargetKind.SCALAR_EV)

    assert config.input_dim > 0
    assert config.output_dim == 2


def test_value_model_forward_shape() -> None:
    pytest.importorskip("torch")
    spec = ValueFeatureSpec(player_count=2, max_history_length=8)
    config = build_value_network_config(spec, ValueTargetKind.SCALAR_EV)
    model = build_value_model(config, device="cpu")
    features = np.zeros((3, config.input_dim), dtype=np.float32)

    output = infer_value(model, features)

    assert output.shape == (3, 2)


def test_training_step_updates_without_error() -> None:
    torch = pytest.importorskip("torch")
    spec = ValueFeatureSpec(player_count=2, max_history_length=8)
    config = build_value_network_config(spec, ValueTargetKind.SCALAR_EV)
    model = build_value_model(config, device="cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    features = np.zeros((4, config.input_dim), dtype=np.float32)
    targets = np.ones((4, 2), dtype=np.float32)

    loss = train_value_step(model, optimizer, features, targets)

    assert loss >= 0.0


def test_checkpoint_save_and_load_round_trips(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    spec = ValueFeatureSpec(player_count=2, max_history_length=8)
    config = build_value_network_config(spec, ValueTargetKind.SCALAR_EV)
    model = build_value_model(config, device="cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = ValueCheckpoint(
        step=7,
        best_metric=0.25,
        feature_spec=spec,
        target_kind=ValueTargetKind.SCALAR_EV,
        target_bucket_count=1,
        model_config=config,
        normalizer=FeatureNormalizer(
            mean=np.zeros(config.input_dim, dtype=np.float32),
            std=np.ones(config.input_dim, dtype=np.float32),
        ),
    )
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(path, model, optimizer, checkpoint)
    loaded_checkpoint, model_state, optimizer_state = load_checkpoint(path)

    assert loaded_checkpoint.step == 7
    assert math.isclose(loaded_checkpoint.best_metric, 0.25, abs_tol=1e-6)
    assert loaded_checkpoint.feature_spec.player_count == 2
    assert loaded_checkpoint.model_config.input_dim == config.input_dim
    assert isinstance(model_state, dict)
    assert isinstance(optimizer_state, dict)

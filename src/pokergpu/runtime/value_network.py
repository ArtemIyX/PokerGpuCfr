from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np

from pokergpu.eval import CpuStubLeafEvaluator, LeafEvaluator, LeafFeatureBatch, LeafValueBatch
from pokergpu.value_network.dataset import FeatureNormalizer, normalize_feature_batch
from pokergpu.value_network.checkpoint import load_checkpoint
from pokergpu.value_network.dataset import ValueFeatureBatch
from pokergpu.value_network.model import ValueMLP, build_value_model, infer_value
from pokergpu.value_network.target import ValueFeatureSpec, ValueTargetKind


@dataclass(frozen=True, slots=True)
class PostflopRuntimeValueNetworkConfig:
    checkpoint_path: Path | None = None
    fallback_to_cpu: bool = True


class PostflopRuntimeValueNetworkEvaluator(LeafEvaluator):
    def __init__(self, model: ValueMLP, feature_spec: ValueFeatureSpec, normalizer: FeatureNormalizer) -> None:
        self._model = model
        self._feature_spec = feature_spec
        self._normalizer = normalizer

    def evaluate(self, batch: LeafFeatureBatch) -> LeafValueBatch:
        features = _build_runtime_leaf_features(batch, self._feature_spec)
        normalized = normalize_feature_batch(ValueFeatureBatch(features), self._normalizer).values
        values = infer_value(self._model, normalized)
        if values.shape != (batch.size, 2):
            raise ValueError("runtime value network must return two outputs per leaf")
        return LeafValueBatch(
            ev_player0=np.asarray(values[:, 0], dtype=np.float32),
            ev_player1=np.asarray(values[:, 1], dtype=np.float32),
        )


def default_postflop_leaf_evaluator(
    config: PostflopRuntimeValueNetworkConfig | None = None,
) -> LeafEvaluator:
    cfg = config or PostflopRuntimeValueNetworkConfig()
    checkpoint_path = cfg.checkpoint_path or _default_checkpoint_path()
    if checkpoint_path is None or not checkpoint_path.exists():
        return CpuStubLeafEvaluator()
    try:
        return _load_postflop_value_network(checkpoint_path)
    except Exception:
        if cfg.fallback_to_cpu:
            return CpuStubLeafEvaluator()
        raise


def _default_checkpoint_path() -> Path | None:
    env_path = os.getenv("POKERGPU_POSTFLOP_VNET_CHECKPOINT", "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    return None


def _load_postflop_value_network(path: Path) -> PostflopRuntimeValueNetworkEvaluator:
    checkpoint, model_state, _optimizer_state = load_checkpoint(path)
    _validate_runtime_checkpoint(
        input_dim=checkpoint.model_config.input_dim,
        output_dim=checkpoint.model_config.output_dim,
        target_kind=checkpoint.target_kind,
        feature_spec=checkpoint.feature_spec,
    )
    model = build_value_model(checkpoint.model_config, device="cpu")
    model.load_state_dict(model_state, strict=True)
    return PostflopRuntimeValueNetworkEvaluator(model, checkpoint.feature_spec, checkpoint.normalizer)


def _validate_runtime_checkpoint(
    *,
    input_dim: int,
    output_dim: int,
    target_kind: ValueTargetKind,
    feature_spec: object,
) -> None:
    if output_dim != 2:
        raise ValueError("runtime value network output dimension mismatch")
    if target_kind is not ValueTargetKind.SCALAR_EV:
        raise ValueError("runtime value network requires scalar EV targets")
    if getattr(feature_spec, "player_count", None) != 2:
        raise ValueError("runtime value network requires heads-up checkpoints")


def _build_runtime_leaf_features(batch: LeafFeatureBatch, feature_spec: ValueFeatureSpec) -> np.ndarray:
    feature_count = (
        1
        + 1
        + 5
        + feature_spec.player_count
        + (feature_spec.player_count * 1326)
        + feature_spec.player_count
        + feature_spec.max_history_length
    )
    features = np.zeros((batch.size, feature_count), dtype=np.float32)
    offset = 0
    features[:, offset] = np.asarray(batch.street, dtype=np.float32)
    offset += 1
    features[:, offset] = np.asarray(batch.pot, dtype=np.float32)
    offset += 1
    offset += 5
    for row_index, player_index in enumerate(np.asarray(batch.player_to_act, dtype=np.int32)):
        if 0 <= player_index < feature_spec.player_count:
            features[row_index, offset + player_index] = 1.0
    offset += feature_spec.player_count
    offset += feature_spec.player_count * 1326
    if feature_spec.player_count >= 1:
        features[:, offset] = np.asarray(batch.stack_p0, dtype=np.float32)
    if feature_spec.player_count >= 2:
        features[:, offset + 1] = np.asarray(batch.stack_p1, dtype=np.float32)
    offset += feature_spec.player_count
    features[:, offset + 0] = np.asarray(batch.board_size, dtype=np.float32)
    features[:, offset + 1] = np.asarray(batch.player_to_act, dtype=np.float32)
    features[:, offset + 2] = np.asarray(batch.is_terminal.astype(np.float32), dtype=np.float32)
    features[:, offset + 3] = np.asarray(batch.is_frontier.astype(np.float32), dtype=np.float32)
    features[:, offset + 4] = np.asarray(np.clip(batch.infoset_id, -1, 1_000_000), dtype=np.float32)
    return features

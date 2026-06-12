from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np

from pokergpu.eval import CpuStubLeafEvaluator, LeafEvaluator, LeafFeatureBatch, LeafValueBatch
from pokergpu.value_network.checkpoint import load_checkpoint
from pokergpu.value_network.model import ValueMLP, build_value_model, infer_value
from pokergpu.value_network.target import ValueTargetKind


_RUNTIME_FEATURE_DIM = 10


@dataclass(frozen=True, slots=True)
class PostflopRuntimeValueNetworkConfig:
    checkpoint_path: Path | None = None
    fallback_to_cpu: bool = True


class PostflopRuntimeValueNetworkEvaluator(LeafEvaluator):
    def __init__(self, model: ValueMLP) -> None:
        self._model = model

    def evaluate(self, batch: LeafFeatureBatch) -> LeafValueBatch:
        features = _build_runtime_leaf_features(batch)
        values = infer_value(self._model, features)
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
    return PostflopRuntimeValueNetworkEvaluator(model)


def _validate_runtime_checkpoint(
    *,
    input_dim: int,
    output_dim: int,
    target_kind: ValueTargetKind,
    feature_spec: object,
) -> None:
    if input_dim != _RUNTIME_FEATURE_DIM:
        raise ValueError("runtime value network input dimension mismatch")
    if output_dim != 2:
        raise ValueError("runtime value network output dimension mismatch")
    if target_kind is not ValueTargetKind.SCALAR_EV:
        raise ValueError("runtime value network requires scalar EV targets")
    if getattr(feature_spec, "player_count", None) != 2:
        raise ValueError("runtime value network requires heads-up checkpoints")


def _build_runtime_leaf_features(batch: LeafFeatureBatch) -> np.ndarray:
    features = np.zeros((batch.size, _RUNTIME_FEATURE_DIM), dtype=np.float32)
    features[:, 0] = np.float32(1.0)
    features[:, 1] = np.asarray(batch.pot, dtype=np.float32)
    features[:, 2] = np.asarray(batch.stack_p0 - batch.stack_p1, dtype=np.float32)
    features[:, 3] = np.asarray(batch.board_size, dtype=np.float32)
    features[:, 4] = np.asarray(batch.street, dtype=np.float32)
    features[:, 5] = np.asarray(batch.reach_p0 - batch.reach_p1, dtype=np.float32)
    features[:, 6] = np.asarray(batch.player_to_act, dtype=np.float32)
    features[:, 7] = np.asarray(batch.is_terminal.astype(np.float32), dtype=np.float32)
    features[:, 8] = np.asarray(batch.is_frontier.astype(np.float32), dtype=np.float32)
    features[:, 9] = np.asarray(np.clip(batch.infoset_id, -1, 1_000_000), dtype=np.float32)
    return features

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pokergpu.eval import (
    CpuStubLeafEvaluator,
    LeafEvaluator,
    LeafFeatureBatch,
    LeafValueBatch,
)
from pokergpu.value_network.checkpoint import load_checkpoint
from pokergpu.value_network.dataset import (
    FeatureNormalizer,
    ValueFeatureBatch,
    normalize_feature_batch,
)
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
        self._input_dim = normalizer.feature_count

    def evaluate(self, batch: LeafFeatureBatch) -> LeafValueBatch:
        features = _build_runtime_leaf_features(batch, self._feature_spec, self._input_dim)
        normalized = normalize_feature_batch(ValueFeatureBatch(features), self._normalizer).values
        values = infer_value(self._model, normalized)
        if values.ndim != 2 or values.shape[0] != batch.size:
            raise ValueError("runtime value network must return one value row per leaf")
        if values.shape[1] != self._feature_spec.player_count:
            raise ValueError("runtime value network output dimension mismatch")
        return LeafValueBatch(
            values=np.asarray(values, dtype=np.float32),
            ev_player0=np.asarray(values[:, 0], dtype=np.float32),
            ev_player1=np.asarray(values[:, 1], dtype=np.float32),
            ev_player2=np.asarray(values[:, 2], dtype=np.float32)
            if values.shape[1] > 2
            else None,
        )

    def evaluate_tensors(self, tensors: dict[str, object]) -> LeafValueBatch:
        def to_np(name: str, dtype: np.dtype[Any] | type[np.generic]) -> np.ndarray:
            value = tensors[name]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=dtype)
        features = np.zeros((int(to_np("street", np.int32).shape[0]), self._input_dim), dtype=np.float32)
        offset = 0

        def put(column: np.ndarray) -> None:
            nonlocal offset
            if offset >= self._input_dim:
                return
            features[:, offset] = column
            offset += 1

        put(to_np("street", np.int32))
        put(to_np("pot", np.float32))
        put(to_np("stack_p0", np.float32))
        put(to_np("stack_p1", np.float32))
        put(to_np("board_size", np.int32))
        put(to_np("player_to_act", np.int32))
        put(to_np("terminal_payoff", np.float32))
        put(to_np("is_terminal", np.bool_).astype(np.float32))
        put(to_np("is_frontier", np.bool_).astype(np.float32))
        put(np.clip(to_np("infoset_id", np.int32), -1, 1_000_000))
        put(to_np("reach_p0", np.float32))
        put(to_np("reach_p1", np.float32))
        put(to_np("reach_p2", np.float32))
        normalized = normalize_feature_batch(ValueFeatureBatch(features), self._normalizer).values
        values = infer_value(self._model, normalized)
        return LeafValueBatch(
            values=np.asarray(values, dtype=np.float32),
            ev_player0=np.asarray(values[:, 0], dtype=np.float32),
            ev_player1=np.asarray(values[:, 1], dtype=np.float32),
            ev_player2=np.asarray(values[:, 2], dtype=np.float32)
            if values.shape[1] > 2
            else None,
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
    if output_dim != getattr(feature_spec, "player_count", output_dim):
        raise ValueError("runtime value network output dimension mismatch")
    if target_kind is not ValueTargetKind.SCALAR_EV:
        raise ValueError("runtime value network requires scalar EV targets")


def _build_runtime_leaf_features(
    batch: LeafFeatureBatch,
    feature_spec: ValueFeatureSpec,
    target_dim: int,
) -> np.ndarray:
    features = np.zeros((batch.size, target_dim), dtype=np.float32)
    offset = 0

    def put(column: np.ndarray) -> None:
        nonlocal offset
        if offset >= target_dim:
            return
        features[:, offset] = np.asarray(column, dtype=np.float32)
        offset += 1

    put(batch.street)
    put(batch.pot)
    put(batch.stack_p0)
    put(batch.stack_p1)
    put(batch.board_size)
    put(batch.player_to_act)
    put(batch.is_terminal.astype(np.float32))
    put(batch.is_frontier.astype(np.float32))
    put(np.clip(batch.infoset_id, -1, 1_000_000))
    put(batch.reach_p0)
    put(batch.reach_p1)
    put(batch.reach_p2)
    return features

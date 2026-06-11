from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

if TYPE_CHECKING:
    import torch
else:
    try:
        import torch
    except Exception:  # pragma: no cover
        torch = None  # type: ignore[assignment]

from .dataset import FeatureNormalizer, FeatureNormalizerPayload
from .model import ValueNetworkConfig
from .target import ValueFeatureSpec, ValueTargetKind


class FeatureSpecPayload(TypedDict):
    player_count: int
    max_history_length: int
    include_player_mask: bool
    include_ranges: bool


class ModelConfigPayload(TypedDict):
    input_dim: int
    hidden_dim: int
    hidden_layers: int
    output_dim: int
    dropout: float


class CheckpointPayload(TypedDict):
    step: int
    best_metric: float
    feature_spec: FeatureSpecPayload
    target_kind: str
    target_bucket_count: int
    model_config: ModelConfigPayload
    normalizer: FeatureNormalizerPayload
    model_state: dict[str, object]
    optimizer_state: dict[str, object]


class StateDictModelProtocol(Protocol):
    def state_dict(self) -> dict[str, object]: ...


class StateDictOptimizerProtocol(Protocol):
    def state_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class ValueCheckpoint:
    step: int
    best_metric: float
    feature_spec: ValueFeatureSpec
    target_kind: ValueTargetKind
    target_bucket_count: int
    model_config: ValueNetworkConfig
    normalizer: FeatureNormalizer


def save_checkpoint(
    path: Path,
    model: StateDictModelProtocol,
    optimizer: StateDictOptimizerProtocol,
    checkpoint: ValueCheckpoint,
) -> None:
    if torch is None:
        raise RuntimeError("torch is required for checkpointing")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: CheckpointPayload = {
        "step": checkpoint.step,
        "best_metric": checkpoint.best_metric,
        "feature_spec": {
            "player_count": checkpoint.feature_spec.player_count,
            "max_history_length": checkpoint.feature_spec.max_history_length,
            "include_player_mask": checkpoint.feature_spec.include_player_mask,
            "include_ranges": checkpoint.feature_spec.include_ranges,
        },
        "target_kind": checkpoint.target_kind.value,
        "target_bucket_count": checkpoint.target_bucket_count,
        "model_config": {
            "input_dim": checkpoint.model_config.input_dim,
            "hidden_dim": checkpoint.model_config.hidden_dim,
            "hidden_layers": checkpoint.model_config.hidden_layers,
            "output_dim": checkpoint.model_config.output_dim,
            "dropout": checkpoint.model_config.dropout,
        },
        "normalizer": checkpoint.normalizer.to_json(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(path: Path) -> tuple[ValueCheckpoint, 
                                         dict[str, object], 
                                         dict[str, object]]:
    if torch is None:
        raise RuntimeError("torch is required for checkpointing")
    payload = cast(CheckpointPayload, 
                   torch.load(path, map_location="cpu", 
                              weights_only=False))
    feature_spec_payload = payload["feature_spec"]
    model_config_payload = payload["model_config"]
    checkpoint = ValueCheckpoint(
        step=int(payload["step"]),
        best_metric=float(payload["best_metric"]),
        feature_spec=ValueFeatureSpec(
            player_count=int(feature_spec_payload["player_count"]),
            max_history_length=int(feature_spec_payload["max_history_length"]),
            include_player_mask=bool(feature_spec_payload["include_player_mask"]),
            include_ranges=bool(feature_spec_payload["include_ranges"]),
        ),
        target_kind=ValueTargetKind(str(payload["target_kind"])),
        target_bucket_count=int(payload["target_bucket_count"]),
        model_config=ValueNetworkConfig(
            input_dim=int(model_config_payload["input_dim"]),
            hidden_dim=int(model_config_payload["hidden_dim"]),
            hidden_layers=int(model_config_payload["hidden_layers"]),
            output_dim=int(model_config_payload["output_dim"]),
            dropout=float(model_config_payload["dropout"]),
        ),
        normalizer=FeatureNormalizer.from_json(payload["normalizer"]),
    )
    return (
        checkpoint,
        payload["model_state"],
        payload["optimizer_state"],
    )

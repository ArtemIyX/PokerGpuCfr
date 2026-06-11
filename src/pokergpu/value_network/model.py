from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch
    from torch import Tensor
    from torch.nn import Module
else:
    try:
        import torch
        from torch import Tensor
        from torch.nn import Module
    except Exception:  # pragma: no cover
        torch = None  # type: ignore[assignment]
        Tensor = object  # type: ignore[assignment]
        Module = object  # type: ignore[assignment]

from .target import ValueFeatureSpec, ValueTargetKind


def default_value_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass(frozen=True, slots=True)
class ValueNetworkConfig:
    input_dim: int
    hidden_dim: int = 256
    hidden_layers: int = 4
    output_dim: int = 1
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input dimension must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden dimension must be positive")
        if self.hidden_layers <= 0:
            raise ValueError("hidden layers must be positive")
        if self.output_dim <= 0:
            raise ValueError("output dimension must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class ValueMLP(Module):
    def __init__(self, config: ValueNetworkConfig) -> None:
        if torch is None:
            raise RuntimeError("torch is required for ValueMLP")
        super().__init__()
        layers: list[torch.nn.Module] = []
        in_dim = config.input_dim
        for _ in range(config.hidden_layers):
            layers.append(torch.nn.Linear(in_dim, config.hidden_dim))
            layers.append(torch.nn.SiLU())
            if config.dropout > 0.0:
                layers.append(torch.nn.Dropout(config.dropout))
            in_dim = config.hidden_dim
        layers.append(torch.nn.Linear(in_dim, config.output_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.network(inputs))


def build_value_network_config(
    feature_spec: ValueFeatureSpec,
    target_kind: ValueTargetKind,
    bucket_count: int = 1,
) -> ValueNetworkConfig:
    input_dim = (
        1
        + 1
        + 5
        + (feature_spec.player_count if feature_spec.include_player_mask else 0)
        + (feature_spec.player_count * 1326 if feature_spec.include_ranges else 0)
        + feature_spec.player_count
        + feature_spec.max_history_length
    )
    output_dim = (feature_spec.player_count 
                  if target_kind is ValueTargetKind.SCALAR_EV 
                  else bucket_count)
    return ValueNetworkConfig(input_dim=input_dim, output_dim=output_dim)


def build_value_model(config: ValueNetworkConfig, 
                      device: str | None = None) -> ValueMLP:
    model = ValueMLP(config)
    if torch is not None and device is not None:
        model = model.to(torch.device(device))
    return model


def model_device(model: ValueMLP) -> "torch.device":
    return next(model.parameters()).device


def infer_value(
    model: ValueMLP,
    features: NDArray[np.float32],
) -> NDArray[np.float32]:
    if torch is None:
        raise RuntimeError("torch is required for inference")
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=model_device(model),
        )
        output = model(tensor)
        return np.asarray(
            output.detach().to("cpu", dtype=torch.float32).numpy(),
            dtype=np.float32,
        )


def train_value_step(
    model: ValueMLP,
    optimizer: torch.optim.Optimizer,
    features: NDArray[np.float32],
    targets: NDArray[np.float32],
    amp: bool = False,
) -> float:
    if torch is None:
        raise RuntimeError("torch is required for training")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    feature_tensor = torch.as_tensor(
        features,
        dtype=torch.float32,
        device=model_device(model),
    )
    target_tensor = torch.as_tensor(
        targets,
        dtype=torch.float32,
        device=feature_tensor.device,
    )
    if amp and feature_tensor.device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        with torch.cuda.amp.autocast():
            prediction = model(feature_tensor)
            loss: Tensor = torch.nn.functional.mse_loss(prediction, target_tensor)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        prediction = model(feature_tensor)
        loss = torch.nn.functional.mse_loss(prediction, target_tensor)
        cast(Any, loss).backward()
        optimizer.step()
    return float(loss.detach().cpu().item())

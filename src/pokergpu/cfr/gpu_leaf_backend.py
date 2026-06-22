from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import cast
from typing import TYPE_CHECKING
from typing import Protocol

import numpy as np

from pokergpu.cfr.leaf_eval import LeafEvalBatchInput
from pokergpu.cfr.leaf_eval import LeafEvalBatchOutput
from pokergpu.cfr.leaf_eval import LeafEvalBackend
from pokergpu.cfr.leaf_model_spec import GpuLeafModelSpec
from pokergpu.cfr.triton_leaf_backend import TritonLeafKernel

if TYPE_CHECKING:
    import torch.nn as nn

_TORCH_IMPORT_ERROR: Exception | None
_torch: Any

try:
    import torch as _torch
except ModuleNotFoundError as exc:  # pragma: no cover - import-time guard
    _torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


class GpuLeafKernel(Protocol):
    def __call__(self, batch: LeafEvalBatchInput) -> LeafEvalBatchOutput:
        """Evaluate a fixed leaf batch on GPU and return batched outputs."""


class TorchLeafKernel:
    def __init__(self, spec: GpuLeafModelSpec = GpuLeafModelSpec()) -> None:
        if _torch is None:
            raise ModuleNotFoundError("torch is required for TorchLeafKernel") from _TORCH_IMPORT_ERROR
        self.spec = spec
        layers: list["nn.Module"] = []
        in_width = spec.input_width
        for hidden_width in spec.hidden_widths:
            layers.append(_torch.nn.Linear(in_width, hidden_width))
            layers.append(_torch.nn.ReLU())
            in_width = hidden_width
        layers.append(_torch.nn.Linear(in_width, spec.output_width))
        self.network = _torch.nn.Sequential(*layers)
        self._cuda_network: "nn.Module" | None = None

    def __call__(self, batch: LeafEvalBatchInput) -> LeafEvalBatchOutput:
        if _torch is None:
            raise ModuleNotFoundError("torch is required for TorchLeafKernel") from _TORCH_IMPORT_ERROR
        if not _torch.cuda.is_available():
            raise RuntimeError("torch GPU backend requires CUDA")
        device = _torch.device("cuda")
        if self._cuda_network is None:
            self._cuda_network = self.network.to(device=device, dtype=_torch.float32)
            self._cuda_network.eval()
        inputs = _torch.from_numpy(batch.features).to(device=device, dtype=_torch.float32, non_blocking=True)
        outputs = self._cuda_network(inputs)
        values = cast(
            np.ndarray,
            outputs.detach().to(device="cpu", dtype=_torch.float32).numpy(),
        )
        return LeafEvalBatchOutput(node_ids=batch.node_ids, values=values)


@dataclass(slots=True, frozen=True)
class GpuLeafBackend(LeafEvalBackend):
    kernel: GpuLeafKernel
    spec: GpuLeafModelSpec = GpuLeafModelSpec()

    def evaluate(self, batch: LeafEvalBatchInput) -> LeafEvalBatchOutput:
        if batch.features.shape[1] != self.spec.input_width:
            raise ValueError("gpu leaf backend received an unexpected input width")
        values = self.kernel(batch)
        if values.node_ids != batch.node_ids:
            raise ValueError("gpu leaf backend must preserve node ordering")
        return values


def create_default_leaf_backend(spec: GpuLeafModelSpec = GpuLeafModelSpec()) -> GpuLeafBackend:
    if _torch is None:
        raise ModuleNotFoundError("torch is required for GPU leaf evaluation") from _TORCH_IMPORT_ERROR
    return GpuLeafBackend(kernel=TorchLeafKernel(spec=spec), spec=spec)


def create_triton_leaf_backend(spec: GpuLeafModelSpec = GpuLeafModelSpec()) -> GpuLeafBackend:
    return GpuLeafBackend(kernel=TritonLeafKernel(spec=spec), spec=spec)

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True, frozen=True)
class EvalDeviceConfig:
    mode: str = "auto"


def resolve_eval_device(config: EvalDeviceConfig) -> torch.device:
    mode = config.mode.lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda requested but not available")
        return torch.device("cuda")
    if mode != "auto":
        raise ValueError("mode must be auto, cpu, or cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

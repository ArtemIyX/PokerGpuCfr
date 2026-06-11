from __future__ import annotations

import torch

from .types import LeafFeatureBatch


def build_gpu_leaf_tensors(
    batch: LeafFeatureBatch,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "player_to_act": torch.as_tensor(
            batch.player_to_act, dtype=torch.int64, device=device
        ).contiguous(),
        "street": torch.as_tensor(
            batch.street, dtype=torch.int64, device=device
        ).contiguous(),
        "pot": torch.as_tensor(
            batch.pot, dtype=torch.float32, device=device
        ).contiguous(),
        "stack_p0": torch.as_tensor(
            batch.stack_p0, dtype=torch.float32, device=device
        ).contiguous(),
        "stack_p1": torch.as_tensor(
            batch.stack_p1, dtype=torch.float32, device=device
        ).contiguous(),
        "board_size": torch.as_tensor(
            batch.board_size, dtype=torch.int64, device=device
        ).contiguous(),
        "reach_p0": torch.as_tensor(
            batch.reach_p0, dtype=torch.float32, device=device
        ).contiguous(),
        "reach_p1": torch.as_tensor(
            batch.reach_p1, dtype=torch.float32, device=device
        ).contiguous(),
        "is_terminal": torch.as_tensor(
            batch.is_terminal, dtype=torch.bool, device=device
        ).contiguous(),
        "is_frontier": torch.as_tensor(
            batch.is_frontier, dtype=torch.bool, device=device
        ).contiguous(),
        "infoset_id": torch.as_tensor(
            batch.infoset_id, dtype=torch.int64, device=device
        ).contiguous(),
    }

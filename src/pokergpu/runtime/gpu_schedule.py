from __future__ import annotations

import torch

from pokergpu.cfr import InfosetLayout


_GPU_MIN_LEVEL_WORK = 256
_GPU_MAX_LEVEL_MERGE = 12


def compact_level_schedule(levels: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    if not levels:
        return ()
    merged: list[tuple[int, ...]] = []
    pending: list[int] = []
    pending_work = 0
    for level_index, level in enumerate(levels):
        pending.append(level_index)
        pending_work += len(level)
        if pending_work >= _GPU_MIN_LEVEL_WORK or len(pending) >= _GPU_MAX_LEVEL_MERGE:
            merged.append(tuple(pending))
            pending = []
            pending_work = 0
    if pending:
        merged.append(tuple(pending))
    return tuple(merged)


def build_infoset_blocks(
    layout: InfosetLayout,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    blocks: list[torch.Tensor] = []
    for infoset_index, action_count in enumerate(layout.action_counts):
        start = int(layout.offsets[infoset_index])
        blocks.append(torch.arange(start, start + int(action_count), dtype=torch.int64, device=device))
    return tuple(blocks)

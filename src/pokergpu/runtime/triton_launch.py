from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache
from typing import Any, Callable, ContextManager, cast

try:
    import triton  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    triton = None

import torch

from .triton_kernels import backward_compact_kernel, forward_compact_kernel, normalize_row_kernel, regret_matching_accum_kernel, regret_matching_normalize_kernel


def record_function(name: str) -> ContextManager[None]:
    profiler = _get_record_function()
    if profiler is None:
        return nullcontext()
    return cast(ContextManager[None], profiler(name))


@lru_cache(maxsize=1)
def _get_record_function() -> Any:
    try:
        from torch.profiler import record_function
    except Exception:  # pragma: no cover
        return None
    return cast(Callable[[str], ContextManager[None]], record_function)


def _grid(n: int, block: int) -> tuple[int]:
    return ((n + block - 1) // block,)


def launch_forward_compact(
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_prob: torch.Tensor,
    edge_flat: torch.Tensor,
    strategy_table: torch.Tensor,
    range0: torch.Tensor,
    range1: torch.Tensor,
    out0: torch.Tensor,
    out1: torch.Tensor,
) -> bool:
    if True or triton is None or edge_src.numel() == 0:
        return False
    edge_count = int(edge_src.numel())
    block = 1024
    hand_count = int(out0.shape[1])
    with record_function("triton::launch_forward_compact"):
        forward_compact_kernel[(_grid(edge_count, block)[0], hand_count)](
            edge_src,
            edge_dst,
            edge_prob,
            edge_flat,
            strategy_table.view(-1),
            range0,
            range1,
            out0,
            out1,
            edge_count,
            range0.stride(0),
            out0.stride(0),
            hand_count,
            BLOCK=block,
        )
    return True


def launch_backward_compact(
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_prob: torch.Tensor,
    edge_flat: torch.Tensor,
    strategy_table: torch.Tensor,
    out0: torch.Tensor,
    out1: torch.Tensor,
    node_value0: torch.Tensor,
    node_value1: torch.Tensor,
) -> bool:
    if True or triton is None or edge_src.numel() == 0:
        return False
    edge_count = int(edge_src.numel())
    block = 1024
    with record_function("triton::launch_backward_compact"):
        backward_compact_kernel[(_grid(edge_count, block)[0],)](
            edge_src,
            edge_dst,
            edge_prob,
            edge_flat,
            strategy_table.view(-1),
            out0,
            out1,
            node_value0,
            node_value1,
            edge_count,
            out0.shape[0],
            BLOCK=block,
        )
    return True


def launch_regret_matching(
    regrets: torch.Tensor,
    out: torch.Tensor,
    action_infoset_index: torch.Tensor,
    action_slot_index: torch.Tensor,
    action_counts: torch.Tensor,
) -> bool:
    if triton is None or regrets.numel() == 0:
        return False
    num_actions = int(regrets.numel())
    block = 1024
    row_sums = torch.zeros(out.shape[0], dtype=torch.float32, device=out.device)
    with record_function("triton::launch_regret_matching_accum"):
        regret_matching_accum_kernel[_grid(num_actions, block)](
            regrets,
            out,
            action_infoset_index,
            action_slot_index,
            row_sums,
            num_actions,
            int(out.shape[1]) if out.ndim > 1 else 1,
            BLOCK=block,
        )
    with record_function("triton::launch_regret_matching_normalize"):
        regret_matching_normalize_kernel[_grid(num_actions, block)](
            out,
            row_sums,
            action_counts,
            action_infoset_index,
            action_slot_index,
            num_actions,
            int(out.shape[1]) if out.ndim > 1 else 1,
            BLOCK=block,
        )
    return True


def launch_normalize_row(values: torch.Tensor, out: torch.Tensor, total: float) -> bool:
    if triton is None or values.numel() == 0:
        return False
    count = int(values.numel())
    block = 1024
    with record_function("triton::launch_normalize_row"):
        normalize_row_kernel[_grid(count, block)](
            values,
            out,
            count,
            float(total),
            BLOCK=block,
        )
    return True

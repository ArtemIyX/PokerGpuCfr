from __future__ import annotations

try:
    import triton  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    triton = None

import torch

from .triton_kernels import backward_compact_kernel, forward_compact_kernel, regret_matching_accum_kernel, regret_matching_normalize_kernel


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
    backward_compact_kernel[_grid(edge_count, block)](
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

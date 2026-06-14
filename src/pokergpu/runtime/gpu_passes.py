from __future__ import annotations

import time
from typing import Any, cast

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"torch is required for GPU postflop solving: {exc}") from exc

from pokergpu.abstraction.hands import RangeVector
from pokergpu.cfr import InfosetLayout, InfosetStore
from pokergpu.core.board import Street
from pokergpu.core.payouts import compute_payouts
from pokergpu.core.state import GameState, HandPhase
from pokergpu.eval import LeafEvaluator
from pokergpu.eval.types import LeafFeatureBatch, LeafValueBatch
from pokergpu.runtime.cache import PackedGpuSolveState
from pokergpu.tree import NodeType, PublicTree

from .gpu_plan import concat_compact_level_edges, concat_level_edges, propagate_node_ranges

__all__ = [
    "regret_matching_table_inplace",
    "average_strategy_from_gpu",
    "make_cpu_store",
    "update_regrets_gpu",
    "update_regrets_compact_group",
    "forward_pass_gpu",
    "forward_pass_compact_group",
    "backward_pass_gpu",
    "backward_pass_compact_group",
    "evaluate_frontier_leaves",
    "concat_level_edges",
    "propagate_node_ranges",
]


def regret_matching_table_inplace(
    out: torch.Tensor,
    regrets: torch.Tensor,
    action_infoset_index: torch.Tensor,
    action_slot_index: torch.Tensor,
    action_counts: torch.Tensor,
    legal_action_mask: torch.Tensor | None = None,
    infoset_blocks: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    out.zero_()
    num_infosets = int(action_counts.numel())
    max_actions = int(out.shape[1])
    if num_infosets == 0:
        return out
    limit = min(int(regrets.numel()), int(action_infoset_index.numel()), int(action_slot_index.numel()))
    if limit == 0:
        return out
    action_infoset_index = action_infoset_index[:limit]
    action_slot_index = action_slot_index[:limit]
    regrets = regrets[:limit]
    valid = (action_infoset_index >= 0) & (action_infoset_index < num_infosets) & (action_slot_index >= 0) & (action_slot_index < max_actions)
    if legal_action_mask is not None and legal_action_mask.numel() >= limit:
        valid = valid & legal_action_mask[:limit]
    if not bool(valid.any()):
        return out
    if infoset_blocks is not None and len(infoset_blocks) == num_infosets:
        for infoset_index, block in enumerate(infoset_blocks):
            if block.numel() == 0:
                continue
            block = block.to(device=out.device)
            block_limit = min(int(block.numel()), max_actions)
            flat = block[:block_limit]
            if flat.numel() == 0:
                continue
            block_infosets = action_infoset_index[flat]
            block_slots = action_slot_index[flat]
            block_regrets = torch.clamp(regrets[flat], min=0.0)
            block_valid = (
                (block_infosets == infoset_index)
                & (block_slots >= 0)
                & (block_slots < max_actions)
            )
            if legal_action_mask is not None and legal_action_mask.numel() >= limit:
                block_valid = block_valid & legal_action_mask[flat]
            if not bool(block_valid.any()):
                continue
            slots = block_slots[block_valid]
            values = block_regrets[block_valid]
            out[infoset_index, slots] = values
            total = float(values.sum().item())
            if total > 0.0:
                out[infoset_index, : max_actions] = out[infoset_index, : max_actions] / total
            else:
                count = int(action_counts[infoset_index].clamp_min(1).item())
                out[infoset_index, :count] = 1.0 / float(count)
        return out
    infosets = action_infoset_index[valid]
    slots = action_slot_index[valid]
    values = torch.clamp(regrets[valid], min=0.0)
    out.index_put_((infosets, slots), values, accumulate=False)
    totals = torch.zeros(num_infosets, dtype=torch.float32, device=out.device)
    totals.scatter_add_(0, infosets, values)
    nonzero = totals > 0
    if bool(nonzero.any()):
        out[nonzero] = out[nonzero] / totals[nonzero].unsqueeze(1)
    zero_rows = ~nonzero
    if bool(zero_rows.any()):
        counts = action_counts[zero_rows].clamp_min(1).to(torch.float32)
        out[zero_rows] = 0.0
        idx = torch.nonzero(zero_rows, as_tuple=False).flatten()
        for i, count in zip(idx.tolist(), counts.tolist(), strict=False):
            out[i, : int(count)] = 1.0 / float(count)
    return out


def average_strategy_from_gpu(
    strategy_sums: torch.Tensor,
    action_counts: torch.Tensor,
    action_offsets: torch.Tensor,
    infoset_index: int,
) -> np.ndarray:
    count = int(action_counts[infoset_index].item())
    start = int(action_offsets[infoset_index].item())
    values = strategy_sums[start : start + count].detach().cpu().numpy().astype(np.float32, copy=False)
    total = float(np.sum(values, dtype=np.float64))
    if total <= 0.0 and count > 0:
        return np.full(count, 1.0 / count, dtype=np.float32)
    if total <= 0.0:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(values / np.float32(total), dtype=np.float32)


def make_cpu_store(
    layout: InfosetLayout,
    regrets: torch.Tensor,
    strategy_sums: torch.Tensor,
    *,
    device: torch.device,
) -> InfosetStore:
    del device
    return InfosetStore(
        layout=layout,
        regrets=regrets.detach().cpu().numpy().astype(np.float32, copy=True),
        strategy_sums=strategy_sums.detach().cpu().numpy().astype(np.float32, copy=True),
    )


def update_regrets_gpu(
    state: PackedGpuSolveState,
    regrets: torch.Tensor,
    strategy_sums: torch.Tensor,
    strategy_table: torch.Tensor,
    node_values_p0: torch.Tensor,
    node_values_p1: torch.Tensor,
    timings: list[float] | None = None,
) -> None:
    for index, _level_indices in enumerate(state.compact_backward_levels):
        started = time.monotonic()
        update_regrets_compact_group(state, index, regrets, strategy_sums, strategy_table, node_values_p0, node_values_p1)
        if timings is not None and index < len(timings):
            timings[index] += time.monotonic() - started


def _process_compact_edges(
    state: PackedGpuSolveState,
    compact_level_index: int,
    strategy_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_src, edge_dst, edge_infoset, edge_slot, edge_kind, edge_prob = concat_compact_level_edges(state, compact_level_index)
    valid = (
        (edge_infoset >= 0)
        & (edge_infoset < strategy_table.shape[0])
        & (edge_slot >= 0)
        & (edge_slot < strategy_table.shape[1])
    )
    if not bool(valid.any()):
        empty_i64 = torch.empty(0, dtype=torch.int64, device=strategy_table.device)
        empty_f32 = torch.empty(0, dtype=torch.float32, device=strategy_table.device)
        return empty_i64, empty_i64, empty_i64, empty_i64, empty_i64, empty_f32
    return (
        edge_src[valid],
        edge_dst[valid],
        edge_infoset[valid],
        edge_slot[valid],
        edge_kind[valid],
        edge_prob[valid],
    )


_COMPACT_MODE_FORWARD = 0
_COMPACT_MODE_BACKWARD = 1
_COMPACT_MODE_REGRET = 2


def _compact_pass(
    state: PackedGpuSolveState,
    compact_level_index: int,
    strategy_table: torch.Tensor,
    mode: int,
    node_range_p0: torch.Tensor | None = None,
    node_range_p1: torch.Tensor | None = None,
    out_p0: torch.Tensor | None = None,
    out_p1: torch.Tensor | None = None,
    regrets: torch.Tensor | None = None,
    strategy_sums: torch.Tensor | None = None,
    node_values_p0: torch.Tensor | None = None,
    node_values_p1: torch.Tensor | None = None,
) -> None:
    edge_src, edge_dst, edge_infoset, edge_slot, edge_kind, edge_prob = _process_compact_edges(
        state, compact_level_index, strategy_table
    )
    if edge_src.numel() == 0:
        return
    if mode == _COMPACT_MODE_FORWARD:
        if node_range_p0 is None or node_range_p1 is None:
            return
        chance_mask = edge_kind == 0
        if bool(chance_mask.any()):
            src = edge_src[chance_mask]
            dst = edge_dst[chance_mask]
            probs = edge_prob[chance_mask].unsqueeze(1)
            node_range_p0.index_add_(0, dst, node_range_p0[src] * probs)
            node_range_p1.index_add_(0, dst, node_range_p1[src] * probs)
        player_mask = edge_kind == 1
        if bool(player_mask.any()):
            src = edge_src[player_mask]
            dst = edge_dst[player_mask]
            infosets = edge_infoset[player_mask]
            slots = edge_slot[player_mask]
            probs = strategy_table[infosets, slots]
            weights = probs.unsqueeze(1)
            node_range_p0.index_add_(0, dst, node_range_p0[src] * weights)
            node_range_p1.index_add_(0, dst, node_range_p1[src])
        player_mask = edge_kind == 2
        if bool(player_mask.any()):
            src = edge_src[player_mask]
            dst = edge_dst[player_mask]
            infosets = edge_infoset[player_mask]
            slots = edge_slot[player_mask]
            probs = strategy_table[infosets, slots]
            node_range_p0.index_add_(0, dst, node_range_p0[src])
            node_range_p1.index_add_(0, dst, node_range_p1[src] * probs.unsqueeze(1))
        return
    if mode == _COMPACT_MODE_BACKWARD:
        if out_p0 is None or out_p1 is None:
            return
        child_p0 = out_p0[edge_dst]
        child_p1 = out_p1[edge_dst]
        chance_mask = edge_kind == 0
        if bool(chance_mask.any()):
            out_p0.index_add_(0, edge_src[chance_mask], edge_prob[chance_mask] * child_p0[chance_mask])
            out_p1.index_add_(0, edge_src[chance_mask], edge_prob[chance_mask] * child_p1[chance_mask])
        player_mask = edge_kind == 1
        if bool(player_mask.any()):
            probs = strategy_table[edge_infoset[player_mask], edge_slot[player_mask]]
            out_p0.index_add_(0, edge_src[player_mask], probs * child_p0[player_mask])
            out_p1.index_add_(0, edge_src[player_mask], probs * child_p1[player_mask])
        player_mask = edge_kind == 2
        if bool(player_mask.any()):
            probs = strategy_table[edge_infoset[player_mask], edge_slot[player_mask]]
            out_p0.index_add_(0, edge_src[player_mask], probs * child_p0[player_mask])
            out_p1.index_add_(0, edge_src[player_mask], probs * child_p1[player_mask])
        return
    if mode == _COMPACT_MODE_REGRET:
        if regrets is None or strategy_sums is None or node_values_p0 is None or node_values_p1 is None:
            return
        flat = state.action_offsets[edge_infoset] + edge_slot
        child_values = torch.where(edge_kind == 1, node_values_p0[edge_dst], node_values_p1[edge_dst])
        strat = strategy_table[edge_infoset, edge_slot]
        infoset_values = torch.zeros(state.action_counts.numel(), dtype=torch.float32, device=regrets.device)
        infoset_values.scatter_add_(0, edge_infoset, strat * child_values)
        regrets.index_add_(0, flat, child_values - infoset_values[edge_infoset])
        strategy_sums.index_add_(0, flat, strat)


def update_regrets_compact_group(
    state: PackedGpuSolveState,
    compact_level_index: int,
    regrets: torch.Tensor,
    strategy_sums: torch.Tensor,
    strategy_table: torch.Tensor,
    node_values_p0: torch.Tensor,
    node_values_p1: torch.Tensor,
) -> None:
    _compact_pass(
        state,
        compact_level_index,
        strategy_table,
        _COMPACT_MODE_REGRET,
        regrets=regrets,
        strategy_sums=strategy_sums,
        node_values_p0=node_values_p0,
        node_values_p1=node_values_p1,
    )


def forward_pass_gpu(
    state: PackedGpuSolveState,
    strategy_table: torch.Tensor,
    node_range_p0: torch.Tensor,
    node_range_p1: torch.Tensor,
    timings: list[float] | None = None,
) -> None:
    propagate_node_ranges(state)
    for index, _level_indices in enumerate(state.compact_forward_levels):
        started = time.monotonic()
        forward_pass_compact_group(state, index, strategy_table, node_range_p0, node_range_p1)
        if timings is not None and index < len(timings):
            timings[index] += time.monotonic() - started


def forward_pass_compact_group(
    state: PackedGpuSolveState,
    compact_level_index: int,
    strategy_table: torch.Tensor,
    node_range_p0: torch.Tensor,
    node_range_p1: torch.Tensor,
) -> None:
    _compact_pass(
        state,
        compact_level_index,
        strategy_table,
        _COMPACT_MODE_FORWARD,
        node_range_p0=node_range_p0,
        node_range_p1=node_range_p1,
    )


def backward_pass_gpu(
    state: PackedGpuSolveState,
    strategy_table: torch.Tensor,
    evaluator: LeafEvaluator,
    out_p0: torch.Tensor,
    out_p1: torch.Tensor,
    timings: list[float] | None = None,
) -> None:
    out_p0.zero_()
    out_p1.zero_()
    frontier_nodes = state.frontier_nodes
    if int(frontier_nodes.numel()) > 0:
        leaf_values = evaluate_frontier_leaves(state, evaluator)
        out_p0[frontier_nodes] = torch.as_tensor(leaf_values.ev_player0, dtype=torch.float32, device=out_p0.device)
        out_p1[frontier_nodes] = torch.as_tensor(leaf_values.ev_player1, dtype=torch.float32, device=out_p1.device)
    for index, _level_indices in enumerate(reversed(state.compact_backward_levels)):
        started = time.monotonic()
        backward_pass_compact_group(state, len(state.compact_backward_levels) - 1 - index, strategy_table, out_p0, out_p1)
        if timings is not None and index < len(timings):
            timings[index] += time.monotonic() - started


def backward_pass_compact_group(
    state: PackedGpuSolveState,
    compact_level_index: int,
    strategy_table: torch.Tensor,
    out_p0: torch.Tensor,
    out_p1: torch.Tensor,
) -> None:
    _compact_pass(
        state,
        compact_level_index,
        strategy_table,
        _COMPACT_MODE_BACKWARD,
        out_p0=out_p0,
        out_p1=out_p1,
    )


def evaluate_frontier_leaves(
    state: PackedGpuSolveState,
    evaluator: LeafEvaluator,
) -> LeafValueBatch:
    if state.frontier_leaf_tensors is None:
        tensors = {
            "range_p0": state.node_range_p0[state.frontier_nodes],
            "range_p1": state.node_range_p1[state.frontier_nodes],
            "range_p2": state.node_range_p2[state.frontier_nodes],
            "street": torch.zeros(state.frontier_nodes.shape[0], dtype=torch.int32, device=state.node_range_p0.device),
            "pot": torch.zeros(state.frontier_nodes.shape[0], dtype=torch.float32, device=state.node_range_p0.device),
            "stack_p0": torch.zeros(state.frontier_nodes.shape[0], dtype=torch.float32, device=state.node_range_p0.device),
            "stack_p1": torch.zeros(state.frontier_nodes.shape[0], dtype=torch.float32, device=state.node_range_p0.device),
            "board_size": torch.zeros(state.frontier_nodes.shape[0], dtype=torch.int32, device=state.node_range_p0.device),
            "player_to_act": torch.zeros(state.frontier_nodes.shape[0], dtype=torch.int32, device=state.node_range_p0.device),
            "terminal_payoff": torch.zeros(state.frontier_nodes.shape[0], dtype=torch.float32, device=state.node_range_p0.device),
            "is_terminal": torch.zeros(state.frontier_nodes.shape[0], dtype=torch.bool, device=state.node_range_p0.device),
            "is_frontier": torch.ones(state.frontier_nodes.shape[0], dtype=torch.bool, device=state.node_range_p0.device),
            "infoset_id": torch.zeros(state.frontier_nodes.shape[0], dtype=torch.int32, device=state.node_range_p0.device),
        }
    else:
        tensors = state.frontier_leaf_tensors
        tensors["range_p0"] = state.node_range_p0[state.frontier_nodes]
        tensors["range_p1"] = state.node_range_p1[state.frontier_nodes]
        tensors["range_p2"] = state.node_range_p2[state.frontier_nodes]
    evaluate_tensors = getattr(evaluator, "evaluate_tensors", None)
    if evaluate_tensors is not None:
        try:
            return cast(LeafValueBatch, evaluate_tensors(tensors))
        except Exception:
            pass
    evaluate = getattr(evaluator, "evaluate", None)
    if evaluate is not None:
        return cast(LeafValueBatch, evaluate(state.packed.leaf_feature_batch))
    raise RuntimeError("GPU leaf evaluation requires evaluate_tensors support")

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from functools import lru_cache
from typing import Any, Callable, ContextManager, cast

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

from .gpu_plan import concat_compact_level_edges, concat_level_edges

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

_COMPACT_COMPILE_ENABLED = os.getenv("POKERGPU_COMPACT_COMPILE", "1").strip().lower() not in {"0", "false", "no", "off"}
__all__ = [
    "regret_matching_table_inplace",
    "average_strategy_from_gpu",
    "make_cpu_store",
    "run_compact_iteration_gpu",
    "update_regrets_gpu",
    "update_regrets_compact_group",
    "propagate_node_ranges_compact",
    "forward_pass_gpu",
    "forward_pass_compact_group",
    "backward_pass_gpu",
    "backward_pass_compact_group",
    "evaluate_frontier_leaves",
    "concat_level_edges",
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
    return _regret_matching_table_inplace_impl(
        out,
        regrets,
        action_infoset_index,
        action_slot_index,
        action_counts,
        legal_action_mask,
        infoset_blocks,
    )


def _regret_matching_table_inplace_impl(
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
    values = strategy_sums.narrow(0, start, count)
    total = float(values.sum().item())
    if total <= 0.0 and count > 0:
        return np.full(count, 1.0 / count, dtype=np.float32)
    if total <= 0.0:
        return np.zeros(0, dtype=np.float32)
    return values.div(total).detach().cpu().numpy().astype(np.float32, copy=False)


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


def _run_compact_iteration_core(
    state: PackedGpuSolveState,
    strategy_table: torch.Tensor,
    *,
    node_range_p0: torch.Tensor,
    node_range_p1: torch.Tensor,
    out_p0: torch.Tensor,
    out_p1: torch.Tensor,
    regrets: torch.Tensor,
    strategy_sums: torch.Tensor,
    node_values_p0: torch.Tensor,
    node_values_p1: torch.Tensor,
) -> None:
    compact_level_count = min(
        len(state.compact_level_edge_src),
        len(state.compact_level_edge_dst),
        len(state.compact_level_edge_infoset),
        len(state.compact_level_edge_slot),
        len(state.compact_level_edge_kind),
        len(state.compact_level_edge_prob),
    )
    compact_backward_level_count = min(
        len(state.compact_backward_edge_src),
        len(state.compact_backward_edge_dst),
        len(state.compact_backward_edge_infoset),
        len(state.compact_backward_edge_slot),
        len(state.compact_backward_edge_kind),
        len(state.compact_backward_edge_prob),
    )
    node_range_p0.zero_()
    node_range_p1.zero_()
    if node_range_p0.shape[0] > 0:
        node_range_p0[0].fill_(1.0)
        node_range_p1[0].fill_(1.0)
    for index in range(compact_level_count):
        edge_src = state.compact_level_edge_src[index]
        edge_dst = state.compact_level_edge_dst[index]
        edge_infoset = state.compact_level_edge_infoset[index]
        edge_slot = state.compact_level_edge_slot[index]
        edge_kind = state.compact_level_edge_kind[index]
        edge_prob = state.compact_level_edge_prob[index]
        if edge_src.numel() == 0:
            continue
        _compact_pass_impl(
            edge_src,
            edge_dst,
            edge_infoset,
            edge_slot,
            edge_kind,
            edge_prob,
            strategy_table,
            _COMPACT_MODE_FORWARD,
            node_range_p0=node_range_p0,
            node_range_p1=node_range_p1,
        )
    node_types = state.packed.node_type
    first_child = state.packed.first_child
    child_count = state.packed.child_count
    edge_child = state.packed.children
    edge_chance_prob = state.packed.chance_prob
    for node_index in range(int(node_types.numel()) - 1, -1, -1):
        start = int(first_child[node_index])
        count = int(child_count[node_index])
        if count <= 0:
            continue
        child_slice = edge_child[start : start + count]
        child_p0 = out_p0[child_slice]
        child_p1 = out_p1[child_slice]
        node_type = int(node_types[node_index].item())
        if node_type == 0:
            probs = edge_chance_prob[start : start + count]
            out_p0[node_index] = torch.sum(probs * child_p0)
            out_p1[node_index] = torch.sum(probs * child_p1)
            continue
        if node_type == 1 or node_type == 2:
            infoset = int(state.packed.infoset_ids[node_index].item())
            slots = torch.arange(count, device=strategy_table.device, dtype=torch.int64)
            probs = strategy_table[infoset, slots]
            out_p0[node_index] = torch.sum(probs * child_p0)
            out_p1[node_index] = torch.sum(probs * child_p1)
            continue
        if node_type == 4:
            continue
    for index in range(compact_backward_level_count):
        edge_src = state.compact_backward_edge_src[index]
        edge_dst = state.compact_backward_edge_dst[index]
        edge_infoset = state.compact_backward_edge_infoset[index]
        edge_slot = state.compact_backward_edge_slot[index]
        edge_kind = state.compact_backward_edge_kind[index]
        edge_prob = state.compact_backward_edge_prob[index]
        if edge_src.numel() == 0:
            continue
        _compact_pass_impl(
            edge_src,
            edge_dst,
            edge_infoset,
            edge_slot,
            edge_kind,
            edge_prob,
            strategy_table,
            _COMPACT_MODE_REGRET,
            regrets=regrets,
            strategy_sums=strategy_sums,
            node_values_p0=node_values_p0,
            node_values_p1=node_values_p1,
            action_offsets=state.action_offsets,
            action_counts=state.action_counts,
        )


def _compiled_iteration_core() -> Any:
    return _run_compact_iteration_core


_COMPACT_ITERATION_CORE: Any = _compiled_iteration_core()


def run_compact_iteration_gpu(
    state: PackedGpuSolveState,
    strategy_table: torch.Tensor,
    *,
    node_range_p0: torch.Tensor,
    node_range_p1: torch.Tensor,
    out_p0: torch.Tensor,
    out_p1: torch.Tensor,
    regrets: torch.Tensor,
    strategy_sums: torch.Tensor,
    node_values_p0: torch.Tensor,
    node_values_p1: torch.Tensor,
    evaluator: LeafEvaluator,
    debug: bool = False,
) -> None:
    with record_function("solve::leaf_eval"):
        if state.frontier_nodes.numel() > 0:
            leaf_values = evaluator.evaluate(state.packed.leaf_feature_batch)
            leaf_start = int(state.frontier_nodes[0].item())
            leaf_count = int(state.frontier_nodes.numel())
            leaf_slice = slice(leaf_start, leaf_start + leaf_count)
            out_p0[leaf_slice] = torch.as_tensor(leaf_values.ev_player0, dtype=torch.float32, device=out_p0.device)
            out_p1[leaf_slice] = torch.as_tensor(leaf_values.ev_player1, dtype=torch.float32, device=out_p1.device)
            if debug:
                print(
                    "debug::leaf",
                    float(torch.sum(torch.abs(out_p0[leaf_slice])).item()),
                    float(torch.sum(torch.abs(out_p1[leaf_slice])).item()),
                    int(leaf_count),
                )
    with record_function("solve::cfr_core"):
        _COMPACT_ITERATION_CORE(
            state,
            strategy_table,
            node_range_p0=node_range_p0,
            node_range_p1=node_range_p1,
            out_p0=out_p0,
            out_p1=out_p1,
            regrets=regrets,
            strategy_sums=strategy_sums,
            node_values_p0=node_values_p0,
            node_values_p1=node_values_p1,
        )
    if debug:
        root_children = state.root_child_nodes
        edge_src0 = state.compact_backward_edge_src[0] if len(state.compact_backward_edge_src) > 0 else None
        edge_dst0 = state.compact_backward_edge_dst[0] if len(state.compact_backward_edge_dst) > 0 else None
        if edge_src0 is None or edge_dst0 is None:
            print("warn::missing_compact_edge_block", len(state.compact_backward_edge_src), len(state.compact_backward_edge_dst))
        if root_children is None or int(root_children.numel()) == 0:
            print("warn::missing_root_children")
        nonzero = torch.nonzero(out_p0 != 0, as_tuple=False).flatten()
        if int(nonzero.numel()) == 0:
            print("warn::backward_p0_all_zero")
        if root_children is not None and int(root_children.numel()) > 0 and len(state.compact_backward_edge_dst) > 0:
            hits = torch.zeros(root_children.shape[0], dtype=torch.bool, device=root_children.device)
            for edge_dst_block in state.compact_backward_edge_dst:
                hits = hits | torch.isin(root_children, edge_dst_block)
            if not bool(hits.any()):
                print("warn::root_children_missing_from_backward_blocks")


def propagate_node_ranges_compact(
    state: PackedGpuSolveState,
    strategy_table: torch.Tensor,
) -> None:
    state.node_range_p0.zero_()
    state.node_range_p1.zero_()
    state.node_range_p2.zero_()
    state.node_range_p0[0] = torch.ones_like(state.node_range_p0[0])
    state.node_range_p1[0] = torch.ones_like(state.node_range_p1[0])
    state.node_range_p2[0] = torch.ones_like(state.node_range_p2[0])
    for index, _level_indices in enumerate(state.compact_forward_levels):
        _compact_pass(
            state,
            index,
            strategy_table,
            _COMPACT_MODE_FORWARD,
            node_range_p0=state.node_range_p0,
            node_range_p1=state.node_range_p1,
        )


_COMPACT_MODE_FORWARD = 0
_COMPACT_MODE_BACKWARD = 1
_COMPACT_MODE_REGRET = 2


def _compact_pass_impl(
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_infoset: torch.Tensor,
    edge_slot: torch.Tensor,
    edge_kind: torch.Tensor,
    edge_prob: torch.Tensor,
    strategy_table: torch.Tensor,
    mode: int,
    *,
    node_range_p0: torch.Tensor | None = None,
    node_range_p1: torch.Tensor | None = None,
    out_p0: torch.Tensor | None = None,
    out_p1: torch.Tensor | None = None,
    action_offsets: torch.Tensor | None = None,
    action_counts: torch.Tensor | None = None,
    regrets: torch.Tensor | None = None,
    strategy_sums: torch.Tensor | None = None,
    node_values_p0: torch.Tensor | None = None,
    node_values_p1: torch.Tensor | None = None,
) -> None:
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
        if (
            regrets is None
            or strategy_sums is None
            or node_values_p0 is None
            or node_values_p1 is None
            or action_offsets is None
            or action_counts is None
        ):
            return
        valid = (
            (edge_infoset >= 0)
            & (edge_infoset < action_counts.numel())
            & (edge_slot >= 0)
            & (edge_slot < strategy_table.shape[1])
        )
        if not bool(valid.any()):
            return
        edge_infoset = edge_infoset[valid]
        edge_slot = edge_slot[valid]
        edge_dst = edge_dst[valid]
        edge_kind = edge_kind[valid]
        flat = action_offsets[edge_infoset] + edge_slot
        child_values = torch.where(edge_kind == 1, node_values_p0[edge_dst], node_values_p1[edge_dst])
        strat = strategy_table[edge_infoset, edge_slot]
        infoset_values = torch.zeros(action_counts.numel(), dtype=torch.float32, device=regrets.device)
        infoset_values.scatter_add_(0, edge_infoset, strat * child_values)
        regrets.index_add_(0, flat, child_values - infoset_values[edge_infoset])
        strategy_sums.index_add_(0, flat, strat)


def _compiled_compact_pass_impl() -> Any:
    return _compact_pass_impl


_COMPACT_PASS_IMPL = _compiled_compact_pass_impl()


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
    edge_src = state.compact_level_edge_src[compact_level_index]
    if edge_src.numel() == 0:
        return
    edge_dst = state.compact_level_edge_dst[compact_level_index]
    edge_infoset = state.compact_level_edge_infoset[compact_level_index]
    edge_slot = state.compact_level_edge_slot[compact_level_index]
    edge_kind = state.compact_level_edge_kind[compact_level_index]
    edge_prob = state.compact_level_edge_prob[compact_level_index]
    _COMPACT_PASS_IMPL(
        edge_src,
        edge_dst,
        edge_infoset,
        edge_slot,
        edge_kind,
        edge_prob,
        strategy_table,
        mode,
        node_range_p0=node_range_p0,
        node_range_p1=node_range_p1,
        out_p0=out_p0,
        out_p1=out_p1,
        action_offsets=state.action_offsets,
        action_counts=state.action_counts,
        regrets=regrets,
        strategy_sums=strategy_sums,
        node_values_p0=node_values_p0,
        node_values_p1=node_values_p1,
    )


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
    started = time.monotonic()
    propagate_node_ranges_compact(state, strategy_table)
    if timings is not None:
        timings.append(time.monotonic() - started)


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
    started = time.monotonic()
    out_p0.zero_()
    out_p1.zero_()
    frontier_nodes = state.frontier_nodes
    if int(frontier_nodes.numel()) > 0:
        leaf_values = evaluator.evaluate(state.packed.leaf_feature_batch)
        leaf_start = int(frontier_nodes[0].item())
        leaf_count = int(frontier_nodes.numel())
        leaf_slice = slice(leaf_start, leaf_start + leaf_count)
        out_p0[leaf_slice] = torch.as_tensor(leaf_values.ev_player0, dtype=torch.float32, device=out_p0.device)
        out_p1[leaf_slice] = torch.as_tensor(leaf_values.ev_player1, dtype=torch.float32, device=out_p1.device)
    for index, _level_indices in enumerate(reversed(state.compact_backward_levels)):
        backward_pass_compact_group(state, len(state.compact_backward_levels) - 1 - index, strategy_table, out_p0, out_p1)
    if timings is not None:
        timings.append(time.monotonic() - started)


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



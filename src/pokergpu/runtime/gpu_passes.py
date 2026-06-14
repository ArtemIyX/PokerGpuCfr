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
from .gpu_types import CompactEdgeGroup

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
    infosets = action_infoset_index.clamp(0, num_infosets - 1)
    slots = action_slot_index.clamp(0, max_actions - 1)
    values = torch.clamp(regrets, min=0.0) * valid.to(regrets.dtype)
    out.index_put_((infosets, slots), values, accumulate=False)
    totals = torch.zeros(num_infosets, dtype=torch.float32, device=out.device)
    totals.scatter_add_(0, infosets, values)
    totals_safe = totals.clamp_min(1.0).unsqueeze(1)
    out /= totals_safe
    zero_rows = totals <= 0
    if zero_rows.numel() > 0:
        max_actions = out.shape[1]
        slots_mask = torch.arange(max_actions, device=out.device).unsqueeze(0)
        counts = action_counts.clamp_min(1).to(out.device).unsqueeze(1)
        uniform = (slots_mask < counts).to(out.dtype) / counts.to(out.dtype)
        out[zero_rows] = uniform[zero_rows]
    return out


def average_strategy_from_gpu(
    strategy_sums: torch.Tensor,
    action_counts: torch.Tensor,
    action_offsets: torch.Tensor,
    infoset_index: int,
) -> np.ndarray:
    count = int(action_counts[infoset_index].detach().cpu().item())
    start = int(action_offsets[infoset_index].detach().cpu().item())
    values = strategy_sums.narrow(0, start, count)
    total = float(values.sum().detach().cpu().item())
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
    debug: bool = False,
) -> None:
    with record_function("solve::forward"):
        node_range_p0.zero_()
        node_range_p1.zero_()
        if node_range_p0.shape[0] > 0:
            node_range_p0[0].fill_(1.0)
            node_range_p1[0].fill_(1.0)
        for group in state.compact_forward_groups:
            _compact_group_pass(group, strategy_table, _COMPACT_MODE_FORWARD, node_range_p0=node_range_p0, node_range_p1=node_range_p1)
    with record_function("solve::backward"):
        for group in state.compact_backward_groups:
            _compact_group_pass(group, strategy_table, _COMPACT_MODE_BACKWARD, out_p0=out_p0, out_p1=out_p1)
    with record_function("solve::regret"):
        for group in state.compact_backward_groups:
            _compact_group_pass(group, strategy_table, _COMPACT_MODE_REGRET, regrets=regrets, strategy_sums=strategy_sums, node_values_p0=node_values_p0, node_values_p1=node_values_p1)


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
        frontier_count = state.frontier_count
        if frontier_count > 0:
            leaf_values = evaluator.evaluate(state.packed.leaf_feature_batch)
            leaf_slice = slice(state.frontier_start, state.frontier_start + frontier_count)
            ev0 = torch.as_tensor(leaf_values.ev_player0, dtype=torch.float32, device=out_p0.device)
            ev1 = torch.as_tensor(leaf_values.ev_player1, dtype=torch.float32, device=out_p1.device)
            out_p0[leaf_slice].copy_(ev0)
            out_p1[leaf_slice].copy_(ev1)
            if debug:
                print(
                    "debug::leaf",
                    float(torch.sum(torch.abs(out_p0[leaf_slice])).detach().cpu()),
                    float(torch.sum(torch.abs(out_p1[leaf_slice])).detach().cpu()),
                    int(frontier_count),
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
            debug=debug,
        )
    if debug:
        root_children = state.root_child_nodes
        edge_src0 = state.compact_backward_edge_src[0] if len(state.compact_backward_edge_src) > 0 else None
        edge_dst0 = state.compact_backward_edge_dst[0] if len(state.compact_backward_edge_dst) > 0 else None
        if edge_src0 is None or edge_dst0 is None:
            print("warn::missing_compact_edge_block", len(state.compact_backward_edge_src), len(state.compact_backward_edge_dst))
        if root_children is None or int(root_children.numel()) == 0:
            print("warn::missing_root_children")
        if out_p0.numel() == 0:
            return
        if float(torch.sum(torch.abs(out_p0)).detach().cpu()) == 0.0:
            print("warn::backward_p0_all_zero")
        if root_children is not None and int(root_children.numel()) > 0 and len(state.compact_backward_edge_dst) > 0:
            hits = torch.zeros(root_children.shape[0], dtype=torch.bool, device=root_children.device)
            for edge_dst_block in state.compact_backward_edge_dst:
                hits = hits | torch.isin(root_children, edge_dst_block)
            if hits.numel() > 0 and not bool(hits.any().detach().cpu()):
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
    for src, dst, infoset, slot, kind, prob, flat, chance_src, chance_dst, chance_prob, p0_src, p0_dst, p0_infoset, p0_slot, p0_flat, p0_prob, p1_src, p1_dst, p1_infoset, p1_slot, p1_flat, p1_prob in zip(
        state.compact_level_edge_src,
        state.compact_level_edge_dst,
        state.compact_level_edge_infoset,
        state.compact_level_edge_slot,
        state.compact_level_edge_kind,
        state.compact_level_edge_prob,
        state.compact_level_edge_flat_p0,
        state.compact_level_edge_src_chance,
        state.compact_level_edge_dst_chance,
        state.compact_level_edge_prob_chance,
        state.compact_level_edge_src_p0,
        state.compact_level_edge_dst_p0,
        state.compact_level_edge_infoset_p0,
        state.compact_level_edge_slot_p0,
        state.compact_level_edge_flat_p0,
        state.compact_level_edge_prob_p0,
        state.compact_level_edge_src_p1,
        state.compact_level_edge_dst_p1,
        state.compact_level_edge_infoset_p1,
        state.compact_level_edge_slot_p1,
        state.compact_level_edge_flat_p1,
        state.compact_level_edge_prob_p1,
        strict=True,
    ):
        _compact_group_pass(
            CompactEdgeGroup(
                src, dst, infoset, slot, kind, prob, flat,
                chance_src, chance_dst, chance_prob,
                p0_src, p0_dst, p0_infoset, p0_slot, p0_flat, p0_prob,
                p1_src, p1_dst, p1_infoset, p1_slot, p1_flat, p1_prob,
            ),
            strategy_table,
            _COMPACT_MODE_FORWARD,
            node_range_p0=state.node_range_p0,
            node_range_p1=state.node_range_p1,
        )


_COMPACT_MODE_FORWARD = 0
_COMPACT_MODE_BACKWARD = 1
_COMPACT_MODE_REGRET = 2


def _make_compact_group(
    src: torch.Tensor,
    dst: torch.Tensor,
    infoset: torch.Tensor,
    slot: torch.Tensor,
    kind: torch.Tensor,
    prob: torch.Tensor,
    *,
    device: torch.device,
) -> CompactEdgeGroup:
    empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
    empty_f32 = torch.empty(0, dtype=torch.float32, device=device)
    return CompactEdgeGroup(
        src=src,
        dst=dst,
        infoset=infoset,
        slot=slot,
        kind=kind,
        prob=prob,
        flat=empty_i64,
        chance_src=empty_i64,
        chance_dst=empty_i64,
        chance_prob=empty_f32,
        p0_src=empty_i64,
        p0_dst=empty_i64,
        p0_infoset=empty_i64,
        p0_slot=empty_i64,
        p0_flat=empty_i64,
        p0_prob=empty_f32,
        p1_src=empty_i64,
        p1_dst=empty_i64,
        p1_infoset=empty_i64,
        p1_slot=empty_i64,
        p1_flat=empty_i64,
        p1_prob=empty_f32,
    )


def _compact_pass_impl(
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_infoset: torch.Tensor | None,
    edge_slot: torch.Tensor | None,
    edge_flat: torch.Tensor | None,
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
    debug: bool = False,
) -> None:
    if edge_src.numel() == 0:
        return
    if mode == _COMPACT_MODE_FORWARD:
        if node_range_p0 is None or node_range_p1 is None:
            return
        edge_src = edge_src.clamp(0, node_range_p0.numel() - 1)
        if edge_infoset is None or edge_slot is None:
            probs = edge_prob.unsqueeze(1)
            edge_dst = edge_dst.clamp(0, node_range_p0.numel() - 1)
            node_range_p0.index_add_(0, edge_dst, node_range_p0[edge_src] * probs)
            node_range_p1.index_add_(0, edge_dst, node_range_p1[edge_src] * probs)
            return
        infoset_limit = int(strategy_table.shape[0])
        slot_limit = int(strategy_table.shape[1]) if strategy_table.ndim > 1 else 0
        if infoset_limit <= 0 or slot_limit <= 0:
            return
        infoset_valid = (edge_infoset >= 0) & (edge_infoset < infoset_limit)
        slot_valid = (edge_slot >= 0) & (edge_slot < slot_limit)
        flat_valid = torch.ones_like(infoset_valid, dtype=torch.bool)
        if edge_flat is not None:
            flat_valid = (edge_flat >= 0) & (edge_flat < strategy_table.numel())
        valid = infoset_valid & slot_valid & flat_valid
        if not bool(valid.any()):
            return
        if edge_flat is None:
            probs = strategy_table[edge_infoset[valid], edge_slot[valid]]
        else:
            probs = strategy_table.view(-1).index_select(0, edge_flat[valid])
        weights = probs.unsqueeze(1)
        edge_dst = edge_dst[valid].clamp(0, node_range_p0.numel() - 1)
        edge_src = edge_src[valid].clamp(0, node_range_p0.numel() - 1)
        node_range_p0.index_add_(0, edge_dst, node_range_p0[edge_src] * weights)
        node_range_p1.index_add_(0, edge_dst, node_range_p1[edge_src] * weights)
        return
    if mode == _COMPACT_MODE_BACKWARD:
        if out_p0 is None or out_p1 is None:
            return
        src_valid = (edge_src >= 0) & (edge_src < out_p0.numel())
        dst_valid = (edge_dst >= 0) & (edge_dst < out_p0.numel())
        valid = src_valid & dst_valid
        edge_src_valid = edge_src[valid]
        edge_dst_valid = edge_dst[valid]
        child_p0 = out_p0[edge_dst_valid]
        child_p1 = out_p1[edge_dst_valid]
        if edge_infoset is None or edge_slot is None:
            out_p0.index_add_(0, edge_src_valid, edge_prob[valid] * child_p0)
            out_p1.index_add_(0, edge_src_valid, edge_prob[valid] * child_p1)
            return
        infoset_limit = int(strategy_table.shape[0])
        slot_limit = int(strategy_table.shape[1]) if strategy_table.ndim > 1 else 0
        if infoset_limit <= 0 or slot_limit <= 0:
            return
        edge_infoset_valid = edge_infoset[valid]
        edge_slot_valid = edge_slot[valid]
        flat_valid = torch.ones_like(edge_infoset_valid, dtype=torch.bool)
        if edge_flat is not None:
            flat_valid = (edge_flat[valid] >= 0) & (edge_flat[valid] < strategy_table.numel())
        strat_valid = (
            (edge_infoset_valid >= 0)
            & (edge_infoset_valid < infoset_limit)
            & (edge_slot_valid >= 0)
            & (edge_slot_valid < slot_limit)
            & flat_valid
        )
        edge_src_valid = edge_src_valid[strat_valid]
        child_p0 = child_p0[strat_valid]
        child_p1 = child_p1[strat_valid]
        if edge_flat is None:
            probs = strategy_table[edge_infoset_valid[strat_valid], edge_slot_valid[strat_valid]]
        else:
            probs = strategy_table.view(-1).index_select(0, edge_flat[valid][strat_valid])
        out_p0.index_add_(0, edge_src_valid, probs * child_p0)
        out_p1.index_add_(0, edge_src_valid, probs * child_p1)
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
        if edge_flat is None:
            return
        if edge_infoset is None or edge_slot is None:
            return
        infoset_limit = int(action_offsets.numel())
        action_limit = int(action_counts.numel())
        if infoset_limit <= 0 or action_limit <= 0:
            return
        edge_infoset_safe = edge_infoset.clamp(0, infoset_limit - 1)
        action_counts_safe = action_counts[edge_infoset_safe].clamp_min(1)
        slot_valid = (edge_slot >= 0) & (edge_slot < action_counts_safe)
        dst_valid = (edge_dst >= 0) & (edge_dst < node_values_p0.numel())
        infoset_valid = (edge_infoset >= 0) & (edge_infoset < infoset_limit) & (edge_infoset < action_limit)
        valid = slot_valid & dst_valid & infoset_valid
        edge_dst_valid = edge_dst[valid]
        edge_infoset_valid = edge_infoset_safe[valid]
        edge_slot_valid = edge_slot[valid]
        child_values = node_values_p0[edge_dst_valid]
        flat = action_offsets[edge_infoset_valid] + edge_slot_valid
        strat = strategy_table.view(-1).index_select(0, flat)
        infoset_values = torch.zeros(action_counts.numel(), dtype=torch.float32, device=regrets.device)
        infoset_values.scatter_add_(0, edge_infoset_valid, strat * child_values)
        regrets.index_add_(0, flat, child_values - infoset_values[edge_infoset_valid])
        strategy_sums.index_add_(0, flat, strat)


def _compiled_compact_pass_impl() -> Any:
    return _compact_pass_impl


_COMPACT_PASS_IMPL = _compiled_compact_pass_impl()


def _compact_group_pass(
    group: CompactEdgeGroup,
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
    edge_src = group.src
    if edge_src.numel() == 0:
        return
    edge_dst = group.dst
    edge_infoset = group.infoset
    edge_slot = group.slot
    edge_kind = group.kind
    edge_prob = group.prob
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
    group = CompactEdgeGroup(
        src=state.compact_backward_edge_src[compact_level_index],
        dst=state.compact_backward_edge_dst[compact_level_index],
        infoset=state.compact_backward_edge_infoset[compact_level_index],
        slot=state.compact_backward_edge_slot[compact_level_index],
        kind=state.compact_backward_edge_kind[compact_level_index],
        prob=state.compact_backward_edge_prob[compact_level_index],
        flat=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        chance_src=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        chance_dst=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        chance_prob=torch.empty(0, dtype=torch.float32, device=state.regrets.device),
        p0_src=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p0_dst=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p0_infoset=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p0_slot=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p0_flat=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p0_prob=torch.empty(0, dtype=torch.float32, device=state.regrets.device),
        p1_src=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p1_dst=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p1_infoset=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p1_slot=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p1_flat=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p1_prob=torch.empty(0, dtype=torch.float32, device=state.regrets.device),
    )
    _compact_group_pass(
        group,
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
    group = CompactEdgeGroup(
        src=state.compact_level_edge_src[compact_level_index],
        dst=state.compact_level_edge_dst[compact_level_index],
        infoset=state.compact_level_edge_infoset[compact_level_index],
        slot=state.compact_level_edge_slot[compact_level_index],
        kind=state.compact_level_edge_kind[compact_level_index],
        prob=state.compact_level_edge_prob[compact_level_index],
        flat=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        chance_src=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        chance_dst=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        chance_prob=torch.empty(0, dtype=torch.float32, device=state.regrets.device),
        p0_src=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p0_dst=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p0_infoset=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p0_slot=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p0_flat=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p0_prob=torch.empty(0, dtype=torch.float32, device=state.regrets.device),
        p1_src=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p1_dst=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p1_infoset=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p1_slot=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p1_flat=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p1_prob=torch.empty(0, dtype=torch.float32, device=state.regrets.device),
    )
    _compact_group_pass(
        group,
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
    frontier_count = state.frontier_count
    if frontier_count > 0:
        leaf_values = evaluator.evaluate(state.packed.leaf_feature_batch)
        leaf_slice = slice(state.frontier_start, state.frontier_start + frontier_count)
        out_p0[leaf_slice].copy_(torch.from_numpy(np.asarray(leaf_values.ev_player0, dtype=np.float32)))
        out_p1[leaf_slice].copy_(torch.from_numpy(np.asarray(leaf_values.ev_player1, dtype=np.float32)))
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
    group = CompactEdgeGroup(
        src=state.compact_backward_edge_src[compact_level_index],
        dst=state.compact_backward_edge_dst[compact_level_index],
        infoset=state.compact_backward_edge_infoset[compact_level_index],
        slot=state.compact_backward_edge_slot[compact_level_index],
        kind=state.compact_backward_edge_kind[compact_level_index],
        prob=state.compact_backward_edge_prob[compact_level_index],
        flat=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        chance_src=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        chance_dst=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        chance_prob=torch.empty(0, dtype=torch.float32, device=state.regrets.device),
        p0_src=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p0_dst=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p0_infoset=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p0_slot=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p0_flat=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p0_prob=torch.empty(0, dtype=torch.float32, device=state.regrets.device),
        p1_src=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p1_dst=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p1_infoset=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p1_slot=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p1_flat=torch.empty(0, dtype=torch.int64, device=state.regrets.device),
        p1_prob=torch.empty(0, dtype=torch.float32, device=state.regrets.device),
    )
    _compact_group_pass(
        group,
        strategy_table,
        _COMPACT_MODE_BACKWARD,
        out_p0=out_p0,
        out_p1=out_p1,
    )



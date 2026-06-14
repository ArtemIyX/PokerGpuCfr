from __future__ import annotations

import time
from typing import Any
from typing import cast

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"torch is required for GPU postflop solving: {exc}") from exc

from pokergpu.abstraction.actions import (
    BaselineActionAbstraction,
    make_postflop_mvp_profile,
)
from pokergpu.abstraction.hands import RangeVector
from pokergpu.cfr import InfosetLayout, InfosetStore, TreeLevels, build_tree_levels
from pokergpu.core.board import Street
from pokergpu.core.canonical import canonical_board_key
from pokergpu.core.payouts import compute_payouts
from pokergpu.core.state import GameState, HandPhase
from pokergpu.eval.cpu_stub import CpuStubLeafEvaluator
from pokergpu.eval import LeafEvaluator
from pokergpu.eval.types import LeafFeatureBatch, LeafValueBatch
from pokergpu.eval.tensor_builder import build_gpu_leaf_tensors
from pokergpu.runtime.cache import LruCache, PackedGpuSolveState, PackedGpuSubtree
from pokergpu.runtime.caching import TreeTemplateKey, make_warm_start_state
from pokergpu.tree import NodeType, PublicTree
from pokergpu.tree.builder import BuiltPublicTree, TreeBuildConfig, build_public_tree
from pokergpu.tree.public_tree import PublicTreeTemplate

from .gpu_compile import compile_packed_subtree
from .gpu_passes import (
    average_strategy_from_gpu,
    make_cpu_store,
    regret_matching_table_inplace,
    run_compact_iteration_gpu,
)
from .gpu_plan import build_batched_gpu_plan, init_gpu_ranges
from .gpu_types import (
    BatchedGpuPlan,
    BatchedGpuSolveInput,
    GpuSolveStats,
    GpuSolveTrace,
    PackedGpuSolve,
)
from .gpu_schedule import build_infoset_blocks, compact_level_schedule
from .postflop import PostflopResolveResult, PostflopResolveSpec, _summarize_root_ev

__all__ = [
    "BatchedGpuPlan",
    "BatchedGpuSolveInput",
    "GpuSolveStats",
    "GpuSolveTrace",
    "PackedGpuSolve",
    "resolve_postflop_gpu",
    "resolve_postflop_gpu_batch",
    "resolve_postflop_gpu_batch_inputs",
    "resolve_postflop_gpu_many",
]


_PRIVATE_HAND_COUNT = 1326


_GPU_PLAN_CACHE: LruCache[PackedGpuSolve] = LruCache(max_entries=64)
_GPU_BATCH_MIN_SOLVES = 4
_GPU_MIN_LEAFS = 1024
_GPU_MIN_INFOSSETS = 1024
def resolve_postflop_gpu_many(
    specs: tuple[PostflopResolveSpec, ...],
    *,
    evaluator: LeafEvaluator | None = None,
) -> tuple[PostflopResolveResult, ...]:
    if not specs:
        return ()
    inputs = tuple(_prepare_batched_input(spec) for spec in specs)
    return resolve_postflop_gpu_batch_inputs(inputs, evaluator=evaluator)


def resolve_postflop_gpu_batch(
    specs: tuple[PostflopResolveSpec, ...],
    *,
    evaluator: LeafEvaluator | None = None,
) -> tuple[PostflopResolveResult, ...]:
    if not specs:
        return ()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for resolve_postflop_gpu_batch")
    inputs = tuple(_prepare_batched_input(spec) for spec in specs)
    return resolve_postflop_gpu_batch_inputs(inputs, evaluator=evaluator)


def resolve_postflop_gpu_batch_inputs(
    items: tuple[BatchedGpuSolveInput, ...],
    *,
    evaluator: LeafEvaluator | None = None,
) -> tuple[PostflopResolveResult, ...]:
    if not items:
        return ()
    evaluator_impl = evaluator or CpuStubLeafEvaluator()
    groups = _group_batched_gpu_inputs(items)
    results: dict[int, PostflopResolveResult] = {}
    for group_items in groups.values():
        packed = tuple(_prepare_gpu_solve(item.spec, template=item.template) for item in group_items)
        solved = tuple(_finish_gpu_solve(item, evaluator_impl) for item in packed)
        for group_item, result in zip(group_items, solved, strict=True):
            results[id(group_item)] = result
    return tuple(results[id(item)] for item in items)


def resolve_postflop_gpu(
    spec: PostflopResolveSpec,
    *,
    evaluator: LeafEvaluator | None = None,
    strict_gpu: bool = False,
) -> PostflopResolveResult:
    if spec.state.player_count != 2:
        raise ValueError("GPU postflop solver currently supports heads-up only")
    if spec.state.current_street is Street.PREFLOP:
        raise ValueError("postflop resolver requires a postflop state")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for resolve_postflop_gpu")

    evaluator_impl = evaluator or CpuStubLeafEvaluator()
    packed = _prepare_gpu_solve(spec)
    return _finish_gpu_solve(packed, evaluator_impl)


def _should_use_gpu(packed: PackedGpuSolve) -> bool:
    return (
        packed.tree.tree.node_count >= _GPU_MIN_INFOSSETS
        and int(packed.packed_subtree.leaf_count) >= _GPU_MIN_LEAFS
    )


def _prepare_batched_input(spec: PostflopResolveSpec) -> BatchedGpuSolveInput:
    built = _build_or_load_tree_template(spec)
    return BatchedGpuSolveInput(spec=spec, template=built.template, cache_key=built.template.tree_key)


def _group_batched_gpu_inputs(
    items: tuple[BatchedGpuSolveInput, ...],
) -> dict[str, tuple[BatchedGpuSolveInput, ...]]:
    grouped: dict[str, list[BatchedGpuSolveInput]] = {}
    for item in items:
        grouped.setdefault(item.template.tree_key, []).append(item)
    return {key: tuple(value) for key, value in grouped.items()}


def _prepare_gpu_solve(spec: PostflopResolveSpec, *, template: PublicTreeTemplate | None = None) -> PackedGpuSolve:
    built = _build_or_load_tree_template(spec, template=template)
    cache_key = _gpu_plan_key(spec, template=built.template)
    cached = _GPU_PLAN_CACHE.get(cache_key)
    if cached is not None:
        return PackedGpuSolve(
            spec=spec,
            tree=cached.tree,
            plan=cached.plan,
            layout=cached.layout,
            root_infoset=cached.root_infoset,
            root_actions=cached.root_actions,
            packed_subtree=cached.packed_subtree,
            gpu_state=_make_gpu_state(
                cached.packed_subtree,
                cached.layout,
                cached.root_ranges,
                plan=cached.plan,
                compact_forward_levels=cached.plan.compact_forward_levels,
                compact_backward_levels=cached.plan.compact_backward_levels,
            ),
        )
    tree = built
    packed_subtree = compile_packed_subtree(tree, spot_key=cache_key, device="cuda")
    action_counts = _build_compact_infoset_action_counts(packed_subtree)
    layout = InfosetLayout.from_action_counts(action_counts)
    root_infoset = tree.tree.infoset_ids[0]
    if root_infoset is None:
        raise ValueError("root node must be a player infoset")
    levels = build_tree_levels(tree.tree)
    plan = build_batched_gpu_plan(
        tree,
        tree.actions_by_node,
        levels,
        layout,
        frontier_leaf_batch=packed_subtree.leaf_feature_batch,
        device=torch.device("cuda"),
    )
    packed = PackedGpuSolve(
        spec=spec,
        tree=tree,
        plan=plan,
        layout=layout,
        root_infoset=int(root_infoset),
        root_actions=tuple(_format_action(action) for action in tree.actions_by_node[0]),
        packed_subtree=packed_subtree,
        root_ranges=_spec_root_ranges(spec),
    )
    packed = PackedGpuSolve(
        spec=packed.spec,
        tree=packed.tree,
        plan=packed.plan,
        layout=packed.layout,
        root_infoset=packed.root_infoset,
        root_actions=packed.root_actions,
        packed_subtree=packed.packed_subtree,
        gpu_state=_make_gpu_state(
            packed.packed_subtree,
            packed.layout,
            packed.root_ranges,
            plan=plan,
            compact_forward_levels=plan.compact_forward_levels,
            compact_backward_levels=plan.compact_backward_levels,
        ),
    )
    _load_warm_start_into_gpu_state(packed, spec)
    state = packed.gpu_state
    if state is None:
        raise RuntimeError("gpu state must be initialized")
    regret_matching_table_inplace(
        state.strategy_table,
        state.regrets,
        state.action_infoset_index,
        state.action_slot_index,
        state.action_counts,
    )
    _GPU_PLAN_CACHE.put(cache_key, packed)
    if spec.cache_state is not None:
        spec.cache_state.store_tree_template(_tree_template_cache_key(spec), tree.template)
    return packed


def _build_or_load_tree_template(
    spec: PostflopResolveSpec,
    *,
    template: PublicTreeTemplate | None = None,
) -> BuiltPublicTree:
    cache_key = _tree_template_cache_key(spec)
    if template is not None:
        return _rebuilt_tree_from_template(spec, template)
    if spec.cache_state is not None:
        cached_template = spec.cache_state.lookup_tree_template(cache_key)
        if cached_template is not None:
            return _rebuilt_tree_from_template(spec, cached_template)
    built = build_public_tree(
        spec.state,
        abstraction=BaselineActionAbstraction(profile=make_postflop_mvp_profile()),
        config=TreeBuildConfig(
            max_depth=spec.max_depth,
            max_nodes=spec.max_nodes,
            min_reach_prob=spec.min_reach_prob,
        ),
    )
    if spec.cache_state is not None:
        spec.cache_state.store_tree_template(cache_key, built.template)
    return built


def _rebuilt_tree_from_template(
    spec: PostflopResolveSpec,
    template: PublicTreeTemplate,
) -> BuiltPublicTree:
    built = build_public_tree(
        spec.state,
        abstraction=BaselineActionAbstraction(profile=make_postflop_mvp_profile()),
        config=TreeBuildConfig(
            max_depth=spec.max_depth,
            max_nodes=spec.max_nodes,
            min_reach_prob=spec.min_reach_prob,
        ),
    )
    return BuiltPublicTree(
        tree=built.tree,
        template=template,
        flat_view=getattr(template, "flat_view", None) or built.template.flat_view,
        level_schedule=getattr(template, "level_schedule", None) or built.template.level_schedule,
        node_states=built.node_states,
        actions_by_node=built.actions_by_node,
        action_abstraction_id=built.action_abstraction_id,
        canonical_board_key=built.canonical_board_key,
        player_count=built.player_count,
        active_players=built.active_players,
    )


def _make_gpu_state(
    packed: PackedGpuSubtree,
    layout: InfosetLayout,
    root_ranges: tuple[RangeVector, ...],
    *,
    plan: BatchedGpuPlan | None = None,
    compact_forward_levels: tuple[tuple[int, ...], ...] = (),
    compact_backward_levels: tuple[tuple[int, ...], ...] = (),
) -> PackedGpuSolveState:
    device = packed.node_type.device
    action_count = int(layout.total_actions)
    regrets = torch.zeros(action_count, dtype=torch.float32, device=device)
    strategy_sums = torch.zeros_like(regrets)
    strategy_table = torch.zeros((packed.infoset_count, max(packed.max_actions, 1)), dtype=torch.float32, device=device)
    ranges = init_gpu_ranges(root_ranges, device=device)
    node_range_p0 = torch.zeros((packed.node_count, _PRIVATE_HAND_COUNT), dtype=torch.float32, device=device)
    node_range_p1 = torch.zeros_like(node_range_p0)
    node_range_p2 = torch.zeros_like(node_range_p0)
    backward_p0 = torch.zeros(packed.node_count, dtype=torch.float32, device=device)
    backward_p1 = torch.zeros_like(backward_p0)
    level_edge_start = plan.level_edge_start if plan is not None else tuple()
    level_edge_count = plan.level_edge_count if plan is not None else tuple()
    level_edge_src = plan.level_edge_src if plan is not None else tuple()
    level_edge_dst = plan.level_edge_dst if plan is not None else tuple()
    level_edge_infoset = plan.level_edge_infoset if plan is not None else tuple()
    level_edge_slot = plan.level_edge_slot if plan is not None else tuple()
    level_edge_kind = plan.level_edge_kind if plan is not None else tuple()
    level_edge_prob = plan.level_edge_prob if plan is not None else tuple()
    compact_level_edge_src = plan.compact_level_edge_src if plan is not None else tuple()
    compact_level_edge_dst = plan.compact_level_edge_dst if plan is not None else tuple()
    compact_level_edge_infoset = plan.compact_level_edge_infoset if plan is not None else tuple()
    compact_level_edge_slot = plan.compact_level_edge_slot if plan is not None else tuple()
    compact_level_edge_kind = plan.compact_level_edge_kind if plan is not None else tuple()
    compact_level_edge_prob = plan.compact_level_edge_prob if plan is not None else tuple()
    compact_level_edge_src_chance = plan.compact_level_edge_src_chance if plan is not None else tuple()
    compact_level_edge_dst_chance = plan.compact_level_edge_dst_chance if plan is not None else tuple()
    compact_level_edge_prob_chance = plan.compact_level_edge_prob_chance if plan is not None else tuple()
    compact_level_edge_src_p0 = plan.compact_level_edge_src_p0 if plan is not None else tuple()
    compact_level_edge_dst_p0 = plan.compact_level_edge_dst_p0 if plan is not None else tuple()
    compact_level_edge_infoset_p0 = plan.compact_level_edge_infoset_p0 if plan is not None else tuple()
    compact_level_edge_slot_p0 = plan.compact_level_edge_slot_p0 if plan is not None else tuple()
    compact_level_edge_flat_p0 = plan.compact_level_edge_flat_p0 if plan is not None else tuple()
    compact_level_edge_prob_p0 = plan.compact_level_edge_prob_p0 if plan is not None else tuple()
    compact_level_edge_src_p1 = plan.compact_level_edge_src_p1 if plan is not None else tuple()
    compact_level_edge_dst_p1 = plan.compact_level_edge_dst_p1 if plan is not None else tuple()
    compact_level_edge_infoset_p1 = plan.compact_level_edge_infoset_p1 if plan is not None else tuple()
    compact_level_edge_slot_p1 = plan.compact_level_edge_slot_p1 if plan is not None else tuple()
    compact_level_edge_flat_p1 = plan.compact_level_edge_flat_p1 if plan is not None else tuple()
    compact_level_edge_prob_p1 = plan.compact_level_edge_prob_p1 if plan is not None else tuple()
    infoset_blocks = plan.infoset_blocks if plan is not None else tuple()
    frontier_leaf_tensors = build_gpu_leaf_tensors(packed.leaf_feature_batch, device) if packed.leaf_feature_batch.size > 0 else None
    if frontier_leaf_tensors is not None:
        frontier_leaf_tensors["range_p0"] = torch.zeros((packed.frontier_nodes.numel(), _PRIVATE_HAND_COUNT), dtype=torch.float32, device=device)
        frontier_leaf_tensors["range_p1"] = torch.zeros_like(frontier_leaf_tensors["range_p0"])
        frontier_leaf_tensors["range_p2"] = torch.zeros_like(frontier_leaf_tensors["range_p0"])
    frontier_range_nodes = packed.frontier_nodes
    frontier_start = int(frontier_range_nodes[0].item()) if frontier_range_nodes.numel() > 0 else 0
    frontier_count = int(frontier_range_nodes.numel())
    frontier_range_p0 = torch.zeros((packed.frontier_nodes.numel(), _PRIVATE_HAND_COUNT), dtype=torch.float32, device=device)
    frontier_range_p1 = torch.zeros_like(frontier_range_p0)
    frontier_range_p2 = torch.zeros_like(frontier_range_p0)
    return PackedGpuSolveState(
        packed=packed,
        regrets=regrets,
        strategy_sums=strategy_sums,
        strategy_table=strategy_table,
        node_range_p0=node_range_p0,
        node_range_p1=node_range_p1,
        node_range_p2=node_range_p2,
        action_infoset_index=packed.action_infoset_index,
        action_slot_index=packed.action_slot_index,
        action_offsets=torch.as_tensor(layout.offsets, dtype=torch.int64, device=device),
        action_counts=torch.as_tensor(layout.action_counts, dtype=torch.int64, device=device),
        backward_p0=backward_p0,
        backward_p1=backward_p1,
        frontier_nodes=packed.frontier_nodes,
        frontier_start=frontier_start,
        frontier_count=frontier_count,
        node_type=packed.node_type,
        node_first_child=packed.first_child,
        node_child_count=packed.child_count,
        node_parent=tuple(),
        node_infoset=packed.infoset_ids,
        node_street=packed.street,
        node_depth=packed.node_depth,
        node_terminal_payoff=packed.terminal_payoffs,
        node_is_frontier=packed.is_frontier,
        node_player=tuple(),
        edge_parent=tuple(),
        edge_child=packed.children,
        edge_node_type=tuple(),
        edge_infoset=tuple(),
        edge_action_slot=packed.action_slot,
        edge_chance_prob=packed.chance_prob,
        level_nodes=tuple(),
        level_frontier_mask=tuple(),
        level_player_mask=tuple(),
        level_legal_action_mask=tuple(),
        level_card_removal_mask=tuple(),
        level_edge_start=level_edge_start,
        level_edge_count=level_edge_count,
        level_edge_src=level_edge_src,
        level_edge_dst=level_edge_dst,
        level_edge_infoset=level_edge_infoset,
        level_edge_slot=level_edge_slot,
        level_edge_kind=level_edge_kind,
        level_edge_prob=level_edge_prob,
        compact_level_edge_src=compact_level_edge_src,
        compact_level_edge_dst=compact_level_edge_dst,
        compact_level_edge_infoset=compact_level_edge_infoset,
        compact_level_edge_slot=compact_level_edge_slot,
        compact_level_edge_kind=compact_level_edge_kind,
        compact_level_edge_prob=compact_level_edge_prob,
        compact_level_edge_src_chance=compact_level_edge_src_chance,
        compact_level_edge_dst_chance=compact_level_edge_dst_chance,
        compact_level_edge_prob_chance=compact_level_edge_prob_chance,
        compact_level_edge_src_p0=compact_level_edge_src_p0,
        compact_level_edge_dst_p0=compact_level_edge_dst_p0,
        compact_level_edge_infoset_p0=compact_level_edge_infoset_p0,
        compact_level_edge_slot_p0=compact_level_edge_slot_p0,
        compact_level_edge_flat_p0=compact_level_edge_flat_p0,
        compact_level_edge_prob_p0=compact_level_edge_prob_p0,
        compact_level_edge_src_p1=compact_level_edge_src_p1,
        compact_level_edge_dst_p1=compact_level_edge_dst_p1,
        compact_level_edge_infoset_p1=compact_level_edge_infoset_p1,
        compact_level_edge_slot_p1=compact_level_edge_slot_p1,
        compact_level_edge_flat_p1=compact_level_edge_flat_p1,
        compact_level_edge_prob_p1=compact_level_edge_prob_p1,
        compact_backward_edge_src=plan.compact_backward_edge_src if plan is not None else tuple(),
        compact_backward_edge_dst=plan.compact_backward_edge_dst if plan is not None else tuple(),
        compact_backward_edge_infoset=plan.compact_backward_edge_infoset if plan is not None else tuple(),
        compact_backward_edge_slot=plan.compact_backward_edge_slot if plan is not None else tuple(),
        compact_backward_edge_kind=plan.compact_backward_edge_kind if plan is not None else tuple(),
        compact_backward_edge_prob=plan.compact_backward_edge_prob if plan is not None else tuple(),
        compact_backward_edge_src_chance=plan.compact_backward_edge_src_chance if plan is not None else tuple(),
        compact_backward_edge_dst_chance=plan.compact_backward_edge_dst_chance if plan is not None else tuple(),
        compact_backward_edge_prob_chance=plan.compact_backward_edge_prob_chance if plan is not None else tuple(),
        compact_backward_edge_src_p0=plan.compact_backward_edge_src_p0 if plan is not None else tuple(),
        compact_backward_edge_dst_p0=plan.compact_backward_edge_dst_p0 if plan is not None else tuple(),
        compact_backward_edge_infoset_p0=plan.compact_backward_edge_infoset_p0 if plan is not None else tuple(),
        compact_backward_edge_slot_p0=plan.compact_backward_edge_slot_p0 if plan is not None else tuple(),
        compact_backward_edge_flat_p0=plan.compact_backward_edge_flat_p0 if plan is not None else tuple(),
        compact_backward_edge_prob_p0=plan.compact_backward_edge_prob_p0 if plan is not None else tuple(),
        compact_backward_edge_src_p1=plan.compact_backward_edge_src_p1 if plan is not None else tuple(),
        compact_backward_edge_dst_p1=plan.compact_backward_edge_dst_p1 if plan is not None else tuple(),
        compact_backward_edge_infoset_p1=plan.compact_backward_edge_infoset_p1 if plan is not None else tuple(),
        compact_backward_edge_slot_p1=plan.compact_backward_edge_slot_p1 if plan is not None else tuple(),
        compact_backward_edge_flat_p1=plan.compact_backward_edge_flat_p1 if plan is not None else tuple(),
        compact_backward_edge_prob_p1=plan.compact_backward_edge_prob_p1 if plan is not None else tuple(),
        compact_forward_levels=compact_forward_levels,
        compact_backward_levels=compact_backward_levels,
        infoset_blocks=infoset_blocks,
        frontier_leaf_tensors=frontier_leaf_tensors,
        frontier_range_nodes=frontier_range_nodes,
        frontier_range_p0=frontier_range_p0,
        frontier_range_p1=frontier_range_p1,
        frontier_range_p2=frontier_range_p2,
        root_child_nodes=plan.root_child_nodes if plan is not None else tuple(),
    )


def _finish_gpu_solve(
    packed: PackedGpuSolve,
    evaluator_impl: LeafEvaluator,
) -> PostflopResolveResult:
    trace = _run_gpu_solve(packed, evaluator_impl)
    return PostflopResolveResult(
        root_infoset_id=trace.packed.root_infoset,
        root_actions=trace.packed.root_actions,
        root_strategy=trace.root_strategy,
        root_action_ev_player0=trace.root_action_ev_player0,
        root_action_ev_player1=trace.root_action_ev_player1,
        root_ev_player0=trace.root_ev_player0,
        root_ev_player1=trace.root_ev_player1,
        iterations=trace.iterations,
        elapsed_seconds=trace.elapsed_seconds,
        node_count=trace.node_count,
        leaf_count=trace.leaf_count,
    )


def _run_gpu_solve(
    packed: PackedGpuSolve,
    evaluator_impl: LeafEvaluator,
    *,
    debug: bool = False,
) -> GpuSolveTrace:
    spec = packed.spec
    tree = packed.tree
    layout = packed.layout
    root_infoset = packed.root_infoset
    started_at = time.monotonic()
    device = torch.device("cuda")
    node_count = tree.tree.node_count
    state = packed.gpu_state
    if state is None:
        state = _make_gpu_state(
            packed.packed_subtree,
            packed.layout,
            packed.root_ranges,
            plan=packed.plan,
            compact_forward_levels=packed.plan.compact_forward_levels,
            compact_backward_levels=packed.plan.compact_backward_levels,
        )
        packed = PackedGpuSolve(
            spec=packed.spec,
            tree=packed.tree,
            plan=packed.plan,
            layout=packed.layout,
            root_infoset=packed.root_infoset,
            root_actions=packed.root_actions,
            packed_subtree=packed.packed_subtree,
            gpu_state=state,
        )
    leaf_count = int(state.frontier_nodes.numel())
    regrets = state.regrets
    strategy_sums = state.strategy_sums
    strategy_table = state.strategy_table
    backward_p0 = state.backward_p0
    backward_p1 = state.backward_p1

    iterations = 0
    phase_seconds = {
        "strategy": 0.0,
        "forward": 0.0,
        "backward": 0.0,
        "regret": 0.0,
        "finalize": 0.0,
    }
    level_node_counts = tuple(int(level.numel()) for level in packed.plan.level_nodes)
    level_edge_counts = tuple(int(level.numel()) for level in packed.plan.level_edge_dst)
    level_frontier_counts = tuple(int(mask.sum().item()) for mask in packed.plan.level_frontier_mask)
    compact_forward_level_sizes = tuple(len(level) for level in packed.plan.compact_forward_levels)
    compact_backward_level_sizes = tuple(len(level) for level in packed.plan.compact_backward_levels)
    deadline = time.monotonic() + max(0.0, spec.time_budget_sec)
    target_iterations = max(0, int(spec.iterations))
    while (
        (target_iterations > 0 and iterations < target_iterations)
        or (target_iterations <= 0 and (time.monotonic() < deadline or iterations == 0))
    ):
        started_phase = time.monotonic()
        run_compact_iteration_gpu(
            state,
            strategy_table,
            node_range_p0=state.node_range_p0,
            node_range_p1=state.node_range_p1,
            out_p0=backward_p0,
            out_p1=backward_p1,
            regrets=regrets,
            strategy_sums=strategy_sums,
            node_values_p0=backward_p0,
            node_values_p1=backward_p1,
            evaluator=evaluator_impl,
            debug=debug,
        )
        phase_seconds["strategy"] += time.monotonic() - started_phase
        iterations += 1
        if target_iterations > 0 and iterations >= target_iterations:
            break
        if spec.time_budget_sec <= 0.0 and target_iterations <= 0:
            break

    started_phase = time.monotonic()
    root_strategy = average_strategy_from_gpu(
        strategy_sums,
        state.action_counts,
        state.action_offsets,
        root_infoset,
    )
    root_action_ev_p0 = _root_action_values_from_backward(
        tree.tree,
        packed.plan,
        backward_p0,
        backward_p1,
        root_infoset,
    )
    if debug:
        print("debug::root_action_raw", root_action_ev_p0.tolist())
    phase_seconds["finalize"] += time.monotonic() - started_phase
    root_action_ev_p1 = -root_action_ev_p0
    bb_scale = float(spec.state.betting_round.blinds.big_blind)
    if bb_scale <= 0.0:
        bb_scale = 1.0
    root_action_ev_p0 = np.asarray(root_action_ev_p0 / bb_scale, dtype=np.float32)
    root_action_ev_p1 = np.asarray(root_action_ev_p1 / bb_scale, dtype=np.float32)
    root_ev_p0 = float(_summarize_root_ev(root_strategy, root_action_ev_p0))
    root_ev_p1 = -root_ev_p0
    if spec.cache_state is not None:
        spec.cache_state.store_warm_start(
            _gpu_plan_key(spec, template=packed.tree.template),
            make_warm_start_state(
                regret=tuple(float(value) for value in regrets.detach().cpu().tolist()),
                strategy_sum=tuple(float(value) for value in strategy_sums.detach().cpu().tolist()),
                source_key=_gpu_plan_key(spec, template=packed.tree.template),
                blend_alpha=1.0,
            ),
        )
    return GpuSolveTrace(
        packed=packed,
        iterations=iterations,
        elapsed_seconds=time.monotonic() - started_at,
        phase_seconds=phase_seconds,
        level_node_counts=level_node_counts,
        level_edge_counts=level_edge_counts,
        level_frontier_counts=level_frontier_counts,
        compact_forward_level_sizes=compact_forward_level_sizes,
        compact_backward_level_sizes=compact_backward_level_sizes,
        compact_phase_seconds={},
        node_count=node_count,
        leaf_count=leaf_count,
        root_strategy=root_strategy,
        root_action_ev_player0=root_action_ev_p0,
        root_action_ev_player1=root_action_ev_p1,
        root_ev_player0=root_ev_p0,
        root_ev_player1=root_ev_p1,
        gpu_backward_p0=backward_p0.detach().cpu().numpy(),
        gpu_backward_p1=backward_p1.detach().cpu().numpy(),
    )


def _gpu_plan_key(spec: PostflopResolveSpec, *, template: PublicTreeTemplate | None = None) -> str:
    template_key = template.tree_key if template is not None else ""
    return "|".join(
        [
            "gpu-pack-v2",
            spec.solver_version,
            str(spec.state.player_count),
            str(spec.state.current_street.value),
            str(spec.state.betting_round.pot.amount),
            str(spec.state.betting_round.to_act),
            str(spec.max_depth),
            str(spec.max_nodes),
            str(spec.min_reach_prob),
            str(spec.state.board.cards),
            str(tuple(int(player.player) for player in spec.state.active_players)),
            template_key,
        ]
    )


def _tree_template_cache_key(spec: PostflopResolveSpec) -> str:
    return TreeTemplateKey(
        street=spec.state.current_street.value,
        canonical_board=canonical_board_key(spec.state.board),
        pot=int(spec.state.betting_round.pot.amount),
        stacks=tuple(int(stack.stack) for stack in spec.state.betting_round.stacks),
        to_act=int(spec.state.betting_round.to_act),
        action_abstraction_id=BaselineActionAbstraction(profile=make_postflop_mvp_profile()).abstraction_id(spec.state),
    ).digest()


def _load_warm_start_into_gpu_state(
    packed: PackedGpuSolve,
    spec: PostflopResolveSpec,
) -> None:
    if spec.cache_state is None or packed.gpu_state is None:
        return
    warm_start = spec.cache_state.lookup_warm_start(_gpu_plan_key(spec, template=packed.tree.template))
    if warm_start is None:
        return
    state = packed.gpu_state
    regret = np.asarray(warm_start.regret, dtype=np.float32)
    strategy_sum = np.asarray(warm_start.strategy_sum, dtype=np.float32)
    if regret.size == state.regrets.numel():
        state.regrets.copy_(torch.as_tensor(regret, dtype=torch.float32, device=state.regrets.device))
    if strategy_sum.size == state.strategy_sums.numel():
        state.strategy_sums.copy_(torch.as_tensor(strategy_sum, dtype=torch.float32, device=state.strategy_sums.device))


def _spec_root_ranges(spec: PostflopResolveSpec) -> tuple[RangeVector, ...]:
    ranges = [spec.range_p0, spec.range_p1]
    if spec.range_p2 is not None:
        ranges.append(spec.range_p2)
    return tuple(ranges)


def _update_store_from_gpu(
    tree: PublicTree,
    store: InfosetStore,
    node_values_p0: torch.Tensor,
    node_values_p1: torch.Tensor,
) -> None:
    node_values_p0_cpu = node_values_p0.detach().cpu().numpy()
    node_values_p1_cpu = node_values_p1.detach().cpu().numpy()
    for node_index, infoset_id in enumerate(tree.infoset_ids):
        if infoset_id is None or tree.node_types[node_index] not in {NodeType.PLAYER0, NodeType.PLAYER1}:
            continue
        action_values = []
        start = tree.first_child[node_index]
        count = tree.child_count[node_index]
        children = tree.children[start : start + count]
        for link in children:
            if tree.node_types[node_index] is NodeType.PLAYER0:
                action_values.append(float(node_values_p0_cpu[int(link.child)]))
            else:
                action_values.append(float(node_values_p1_cpu[int(link.child)]))
        action_values_arr = np.asarray(action_values, dtype=np.float32)
        if action_values_arr.size == 0:
            continue
        strategy = store.current_strategy(int(infoset_id))
        limit = min(strategy.shape[0], action_values_arr.shape[0])
        if limit == 0:
            continue
        action_values_arr = action_values_arr[:limit]
        regrets = store.regrets_for_infoset(int(infoset_id))
        regrets_slice = regrets[:limit]
        infoset_value = np.float32(
            np.sum(strategy[:limit] * action_values_arr, dtype=np.float64)
        )
        regrets_slice += action_values_arr - infoset_value
        np.maximum(regrets_slice, 0.0, out=regrets_slice)
        store.strategy_sums_for_infoset(int(infoset_id))[:] += strategy


def _build_infoset_action_counts(
    tree: PublicTree,
    actions_by_node: tuple[tuple[object, ...], ...],
) -> tuple[int, ...]:
    max_infoset = -1
    counts: dict[int, int] = {}
    for node_index, infoset_id in enumerate(tree.infoset_ids):
        if infoset_id is None:
            continue
        infoset_index = int(infoset_id)
        max_infoset = max(max_infoset, infoset_index)
        counts[infoset_index] = max(counts.get(infoset_index, 0), len(actions_by_node[node_index]) or 1)
    return tuple(counts.get(index, 1) for index in range(max_infoset + 1))


def _build_compact_infoset_action_counts(packed: PackedGpuSubtree) -> tuple[int, ...]:
    if packed.infoset_count <= 0 or packed.action_infoset_index.numel() == 0:
        return ()
    counts = [0] * packed.infoset_count
    action_infoset_index = packed.action_infoset_index.detach().cpu().tolist()
    action_slot_index = packed.action_slot_index.detach().cpu().tolist()
    for infoset_index, slot_index in zip(action_infoset_index, action_slot_index, strict=True):
        if infoset_index < 0 or infoset_index >= packed.infoset_count:
            continue
        counts[infoset_index] = max(counts[infoset_index], int(slot_index) + 1)
    return tuple(count if count > 0 else 1 for count in counts)


def _format_action(action: object) -> str:
    amount = getattr(action, "amount", None)
    action_type = getattr(action, "action_type", None)
    if amount is None:
        return str(getattr(action_type, "value", action_type))
    return f"{getattr(action_type, 'value', action_type)}({int(amount)})"


def _terminal_value_player0(state: GameState) -> float:
    if state.phase is not HandPhase.TERMINAL:
        return 0.0
    payouts = compute_payouts(state)
    player0_payout = next(
        (payout.amount for payout in payouts if payout.player == 0),
        0,
    )
    other_payouts = sum(payout.amount for payout in payouts if payout.player != 0)
    return float(player0_payout - other_payouts)


def _root_action_values_from_backward(
    tree: PublicTree,
    plan: BatchedGpuPlan,
    backward_p0: torch.Tensor,
    backward_p1: torch.Tensor,
    root_infoset: int,
) -> np.ndarray:
    del tree, root_infoset
    limit = int(plan.root_child_nodes.numel())
    if limit <= 0:
        return np.zeros(0, dtype=np.float32)
    child_nodes = plan.root_child_nodes[:limit]
    valid = (child_nodes >= 0) & (child_nodes < backward_p0.numel())
    safe_child_nodes = child_nodes.clamp(0, max(0, backward_p0.numel() - 1))
    values = backward_p0[safe_child_nodes] * valid.to(backward_p0.dtype)
    return values.detach().cpu().numpy().astype(np.float32, copy=False)


def debug_first_gpu_cpu_divergence(
    spec: PostflopResolveSpec,
    *,
    evaluator: LeafEvaluator | None = None,
) -> None:
    evaluator_impl = evaluator or CpuStubLeafEvaluator()
    packed = _prepare_gpu_solve(spec)
    trace = _run_gpu_solve(packed, evaluator_impl)
    tree = packed.tree
    gpu_nodes = trace.root_action_ev_player0
    print("root_child_nodes", packed.plan.root_child_nodes.detach().cpu().tolist())
    print("gpu_root_action_ev", gpu_nodes.tolist())
    print("status", "CPU comparison removed")


def _print_first_branch_divergence(
    tree: PublicTree,
    node_states: tuple[GameState, ...],
    start_node: int,
    trace: GpuSolveTrace,
) -> None:
    gpu_values = trace.gpu_backward_p0
    if gpu_values is None:
        return
    stack = [start_node]
    visited: set[int] = set()
    while stack:
        node_index = stack.pop()
        if node_index in visited:
            continue
        visited.add(node_index)
        node_type = tree.node_types[node_index]
        if node_type in {NodeType.LEAF, NodeType.TERMINAL}:
            continue
        if node_index < len(gpu_values):
            gpu_value = float(gpu_values[node_index])
            print(
                "gpu_branch_value",
                {
                    "node": node_index,
                    "gpu": gpu_value,
                    "node_type": str(node_type),
                    "state": node_states[node_index],
                },
            )
            return
        start = tree.first_child[node_index]
        count = tree.child_count[node_index]
        for child_link in reversed(tree.children[start : start + count]):
            stack.append(int(child_link.child))

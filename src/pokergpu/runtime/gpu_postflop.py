from __future__ import annotations

import time
from dataclasses import dataclass
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
from pokergpu.eval import LeafEvaluator
from pokergpu.eval.types import LeafFeatureBatch, LeafValueBatch
from pokergpu.runtime.cache import LruCache, PackedGpuSolveState, PackedGpuSubtree
from pokergpu.runtime.caching import TreeTemplateKey, make_warm_start_state
from pokergpu.tree import NodeType, PublicTree
from pokergpu.tree.builder import BuiltPublicTree, TreeBuildConfig, build_public_tree
from pokergpu.tree.public_tree import PublicTreeTemplate

from .gpu_compile import compile_packed_subtree
from .postflop import PostflopResolveResult, PostflopResolveSpec, _summarize_root_ev
from .value_network import default_postflop_leaf_evaluator


@dataclass(frozen=True, slots=True)
class GpuSolveStats:
    iterations: int
    node_count: int
    leaf_count: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class PackedGpuSolve:
    spec: PostflopResolveSpec
    tree: BuiltPublicTree
    plan: "BatchedGpuPlan"
    layout: InfosetLayout
    root_infoset: int
    root_actions: tuple[str, ...]
    packed_subtree: PackedGpuSubtree
    gpu_state: PackedGpuSolveState | None = None
    root_ranges: tuple[RangeVector, ...] = ()


@dataclass(frozen=True, slots=True)
class BatchedGpuSolveInput:
    spec: PostflopResolveSpec
    template: PublicTreeTemplate
    cache_key: str = ""


@dataclass(frozen=True, slots=True)
class BatchedGpuPlan:
    node_type: torch.Tensor
    node_first_child: torch.Tensor
    node_child_count: torch.Tensor
    node_parent: torch.Tensor
    node_infoset: torch.Tensor
    node_street: torch.Tensor
    node_depth: torch.Tensor
    node_terminal_payoff: torch.Tensor
    node_is_frontier: torch.Tensor
    forward_levels: tuple[dict[str, torch.Tensor], ...]
    backward_levels: tuple[dict[str, torch.Tensor], ...]
    level_frontier_mask: tuple[torch.Tensor, ...]
    level_player_mask: tuple[torch.Tensor, ...]
    edge_parent: torch.Tensor
    edge_child: torch.Tensor
    edge_node_type: torch.Tensor
    edge_infoset: torch.Tensor
    edge_action_slot: torch.Tensor
    edge_chance_prob: torch.Tensor
    level_nodes: tuple[torch.Tensor, ...]
    node_player: torch.Tensor
    level_edge_start: tuple[torch.Tensor, ...]
    level_edge_count: tuple[torch.Tensor, ...]
    level_edge_src: tuple[torch.Tensor, ...]
    level_edge_dst: tuple[torch.Tensor, ...]
    level_edge_infoset: tuple[torch.Tensor, ...]
    level_edge_slot: tuple[torch.Tensor, ...]
    level_edge_kind: tuple[torch.Tensor, ...]
    level_edge_prob: tuple[torch.Tensor, ...]
    frontier_nodes: torch.Tensor
    frontier_leaf_batch: LeafFeatureBatch
    root_child_nodes: torch.Tensor
    root_child_parent_infoset: int
    action_counts: torch.Tensor
    action_offsets: torch.Tensor


@dataclass(frozen=True, slots=True)
class GpuSolveTrace:
    packed: PackedGpuSolve
    iterations: int
    elapsed_seconds: float
    phase_seconds: dict[str, float]
    node_count: int
    leaf_count: int
    root_strategy: np.ndarray
    root_action_ev_player0: np.ndarray
    root_action_ev_player1: np.ndarray
    root_ev_player0: float
    root_ev_player1: float
    gpu_backward_p0: np.ndarray
    gpu_backward_p1: np.ndarray


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
    evaluator_impl = evaluator or default_postflop_leaf_evaluator()
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

    evaluator_impl = evaluator or default_postflop_leaf_evaluator()
    packed = _prepare_gpu_solve(spec)
    return _finish_gpu_solve(packed, evaluator_impl)


def _should_use_gpu(packed: PackedGpuSolve) -> bool:
    return (
        packed.tree.tree.node_count >= _GPU_MIN_INFOSSETS
        and int(packed.packed_subtree.leaf_count) >= _GPU_MIN_LEAFS
    )


def _root_child_nodes(tree: BuiltPublicTree) -> tuple[int, ...]:
    start = tree.tree.first_child[0]
    count = tree.tree.child_count[0]
    return tuple(int(tree.tree.children[start + index].child) for index in range(count))


def _build_batched_gpu_plan(
    tree: BuiltPublicTree,
    actions_by_node: tuple[tuple[object, ...], ...],
    levels: TreeLevels,
    layout: InfosetLayout,
    *,
    frontier_leaf_batch: LeafFeatureBatch,
    device: torch.device,
) -> BatchedGpuPlan:
    forward_levels: list[dict[str, torch.Tensor]] = []
    backward_levels: list[dict[str, torch.Tensor]] = []
    node_infoset: list[int] = []
    node_player: list[int] = []
    node_parent: list[int] = [-1] * tree.tree.node_count
    flat_view = tree.template.flat_view
    frontier_nodes = [
        index
        for index in range(tree.tree.node_count)
        if flat_view.is_frontier[index] and flat_view.node_type[index] is not NodeType.TERMINAL
    ]
    root_children = _root_child_nodes(tree)
    action_counts = torch.as_tensor(layout.action_counts, dtype=torch.int64, device=device)
    action_offsets = torch.as_tensor(layout.offsets, dtype=torch.int64, device=device)
    node_type_tensor = torch.as_tensor([_node_type_code(nt) for nt in flat_view.node_type], dtype=torch.int64, device=device)
    node_first_child = torch.as_tensor(flat_view.first_child, dtype=torch.int64, device=device)
    node_child_count = torch.as_tensor(flat_view.child_count, dtype=torch.int64, device=device)
    node_street = torch.as_tensor(flat_view.street, dtype=torch.int64, device=device)
    node_depth = torch.as_tensor(flat_view.node_depth, dtype=torch.int64, device=device)
    node_terminal_payoff = torch.as_tensor(flat_view.terminal_payoff, dtype=torch.float32, device=device)
    node_is_frontier = torch.as_tensor(flat_view.is_frontier, dtype=torch.bool, device=device)
    edge_parent = torch.as_tensor(flat_view.edge_parent, dtype=torch.int64, device=device)
    edge_child = torch.as_tensor(flat_view.edge_child, dtype=torch.int64, device=device)
    edge_action_slot = torch.as_tensor(flat_view.edge_action_slot, dtype=torch.int64, device=device)
    edge_chance_prob = torch.as_tensor(flat_view.edge_chance_prob, dtype=torch.float32, device=device)
    edge_infoset = torch.as_tensor(flat_view.edge_infoset_id, dtype=torch.int64, device=device)
    edge_node_type_tensor = torch.as_tensor(flat_view.edge_player, dtype=torch.int64, device=device)
    level_nodes = tuple(
        torch.as_tensor(level, dtype=torch.int64, device=device)
        for level in levels.forward_levels
    )
    level_frontier_mask: list[torch.Tensor] = []
    level_player_mask: list[torch.Tensor] = []
    level_edge_start: list[torch.Tensor] = []
    level_edge_count: list[torch.Tensor] = []
    level_edge_src: list[torch.Tensor] = []
    level_edge_dst: list[torch.Tensor] = []
    level_edge_infoset: list[torch.Tensor] = []
    level_edge_slot: list[torch.Tensor] = []
    level_edge_kind: list[torch.Tensor] = []
    level_edge_prob: list[torch.Tensor] = []
    for level in levels.forward_levels:
        forward_levels.append(_build_level_plan(tree, level, actions_by_node, device=device))
    for level in levels.backward_levels:
        backward_levels.append(_build_level_plan(tree, level, actions_by_node, device=device))
    for level_tensor in level_nodes:
        level_set = set(int(i) for i in level_tensor.tolist())
        level_frontier_mask.append(
            torch.as_tensor([flat_view.is_frontier[int(i)] for i in level_tensor.tolist()], dtype=torch.bool, device=device)
        )
        level_player_mask.append(
            torch.as_tensor(
                [flat_view.node_type[int(i)] in {NodeType.PLAYER0, NodeType.PLAYER1} for i in level_tensor.tolist()],
                dtype=torch.bool,
                device=device,
            )
        )
        starts = [flat_view.first_child[int(i)] for i in level_tensor.tolist()]
        counts = [flat_view.child_count[int(i)] for i in level_tensor.tolist()]
        level_edge_start.append(torch.as_tensor(starts, dtype=torch.int64, device=device))
        level_edge_count.append(torch.as_tensor(counts, dtype=torch.int64, device=device))
        edge_indices = [i for i, parent in enumerate(flat_view.edge_parent) if parent in level_set]
        level_edge_src.append(torch.as_tensor([flat_view.edge_parent[i] for i in edge_indices], dtype=torch.int64, device=device))
        level_edge_dst.append(torch.as_tensor([flat_view.edge_child[i] for i in edge_indices], dtype=torch.int64, device=device))
        level_edge_infoset.append(torch.as_tensor([flat_view.edge_infoset_id[i] for i in edge_indices], dtype=torch.int64, device=device))
        level_edge_slot.append(torch.as_tensor([flat_view.edge_action_slot[i] for i in edge_indices], dtype=torch.int64, device=device))
        level_edge_kind.append(torch.as_tensor([flat_view.edge_player[i] for i in edge_indices], dtype=torch.int64, device=device))
        level_edge_prob.append(torch.as_tensor([flat_view.edge_chance_prob[i] for i in edge_indices], dtype=torch.float32, device=device))
    for node_index in range(tree.tree.node_count):
        node_type = tree.tree.node_types[node_index]
        node_player.append(0 if node_type is NodeType.PLAYER0 else 1 if node_type is NodeType.PLAYER1 else -1)
        infoset_id = tree.tree.infoset_ids[node_index]
        node_infoset.append(int(infoset_id) if infoset_id is not None else -1)
    for parent, child in enumerate(flat_view.edge_child):
        node_parent[int(child)] = int(flat_view.edge_parent[parent])
    return BatchedGpuPlan(
        node_type=node_type_tensor,
        node_first_child=node_first_child,
        node_child_count=node_child_count,
        node_parent=torch.as_tensor(node_parent, dtype=torch.int64, device=device),
        node_infoset=torch.as_tensor(node_infoset, dtype=torch.int64, device=device),
        node_street=node_street,
        node_depth=node_depth,
        node_terminal_payoff=node_terminal_payoff,
        node_is_frontier=node_is_frontier,
        forward_levels=tuple(forward_levels),
        backward_levels=tuple(backward_levels),
        level_frontier_mask=tuple(level_frontier_mask),
        level_player_mask=tuple(level_player_mask),
        edge_parent=edge_parent,
        edge_child=edge_child,
        edge_node_type=edge_node_type_tensor,
        edge_infoset=edge_infoset,
        edge_action_slot=edge_action_slot,
        edge_chance_prob=edge_chance_prob,
        level_nodes=level_nodes,
        node_player=torch.as_tensor(node_player, dtype=torch.int64, device=device),
        level_edge_start=tuple(level_edge_start),
        level_edge_count=tuple(level_edge_count),
        level_edge_src=tuple(level_edge_src),
        level_edge_dst=tuple(level_edge_dst),
        level_edge_infoset=tuple(level_edge_infoset),
        level_edge_slot=tuple(level_edge_slot),
        level_edge_kind=tuple(level_edge_kind),
        level_edge_prob=tuple(level_edge_prob),
        frontier_nodes=torch.as_tensor(frontier_nodes, dtype=torch.int64, device=device),
        frontier_leaf_batch=frontier_leaf_batch,
        root_child_nodes=torch.as_tensor(root_children, dtype=torch.int64, device=device),
        root_child_parent_infoset=int(tree.tree.infoset_ids[0] or 0),
        action_counts=action_counts,
        action_offsets=action_offsets,
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
            gpu_state=_make_gpu_state(cached.packed_subtree, cached.layout, cached.root_ranges),
        )
    tree = built
    packed_subtree = compile_packed_subtree(tree, spot_key=cache_key, device="cuda")
    action_counts = _build_compact_infoset_action_counts(packed_subtree)
    layout = InfosetLayout.from_action_counts(action_counts)
    root_infoset = tree.tree.infoset_ids[0]
    if root_infoset is None:
        raise ValueError("root node must be a player infoset")
    levels = build_tree_levels(tree.tree)
    plan = _build_batched_gpu_plan(
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
        gpu_state=_make_gpu_state(packed.packed_subtree, packed.layout, packed.root_ranges),
    )
    _load_warm_start_into_gpu_state(packed, spec)
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
) -> PackedGpuSolveState:
    device = packed.node_type.device
    action_count = int(layout.total_actions)
    regrets = torch.zeros(action_count, dtype=torch.float32, device=device)
    strategy_sums = torch.zeros_like(regrets)
    strategy_table = torch.zeros((packed.infoset_count, max(packed.max_actions, 1)), dtype=torch.float32, device=device)
    ranges = _init_gpu_ranges(root_ranges, device=device)
    node_range_p0 = torch.zeros((packed.node_count, _PRIVATE_HAND_COUNT), dtype=torch.float32, device=device)
    node_range_p1 = torch.zeros_like(node_range_p0)
    node_range_p2 = torch.zeros_like(node_range_p0)
    backward_p0 = torch.zeros(packed.node_count, dtype=torch.float32, device=device)
    backward_p1 = torch.zeros_like(backward_p0)
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
        level_edge_start=tuple(),
        level_edge_count=tuple(),
        level_edge_src=tuple(),
        level_edge_dst=tuple(),
        level_edge_infoset=tuple(),
        level_edge_slot=tuple(),
        level_edge_kind=tuple(),
        level_edge_prob=tuple(),
        root_child_nodes=tuple(),
    )


def _regret_matching_table_inplace(
    out: torch.Tensor,
    regrets: torch.Tensor,
    action_infoset_index: torch.Tensor,
    action_slot_index: torch.Tensor,
    action_counts: torch.Tensor,
    legal_action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    out.zero_()
    num_infosets = int(action_counts.numel())
    if num_infosets == 0 or action_infoset_index.numel() == 0:
        return out
    max_actions = int(out.shape[1])
    valid = (action_infoset_index >= 0) & (action_infoset_index < num_infosets) & (action_slot_index >= 0) & (action_slot_index < max_actions)
    if legal_action_mask is not None and legal_action_mask.numel() == valid.numel():
        valid = valid & legal_action_mask
    if not bool(valid.any()):
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
        state = _make_gpu_state(packed.packed_subtree, packed.layout, packed.root_ranges)
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
    deadline = time.monotonic() + max(0.0, spec.time_budget_sec)
    target_iterations = max(0, int(spec.iterations))
    while (
        (target_iterations > 0 and iterations < target_iterations)
        or (target_iterations <= 0 and (time.monotonic() < deadline or iterations == 0))
    ):
        started_phase = time.monotonic()
        _regret_matching_table_inplace(
            strategy_table,
            regrets,
            state.action_infoset_index,
            state.action_slot_index,
            state.action_counts,
            state.packed.legal_action_mask,
        )
        phase_seconds["strategy"] += time.monotonic() - started_phase

        started_phase = time.monotonic()
        _propagate_node_ranges(state)
        _forward_pass_gpu(state, strategy_table, state.node_range_p0, state.node_range_p1)
        phase_seconds["forward"] += time.monotonic() - started_phase

        started_phase = time.monotonic()
        backward_p0.zero_()
        backward_p1.zero_()
        _backward_pass_gpu(
            state,
            strategy_table,
            evaluator_impl,
            backward_p0,
            backward_p1,
        )
        phase_seconds["backward"] += time.monotonic() - started_phase

        started_phase = time.monotonic()
        _update_regrets_gpu(
            state,
            regrets,
            strategy_sums,
            strategy_table,
            backward_p0,
            backward_p1,
        )
        phase_seconds["regret"] += time.monotonic() - started_phase
        iterations += 1
        if target_iterations > 0 and iterations >= target_iterations:
            break
        if spec.time_budget_sec <= 0.0 and target_iterations <= 0:
            break

    started_phase = time.monotonic()
    _regret_matching_table_inplace(
        strategy_table,
        regrets,
        state.action_infoset_index,
        state.action_slot_index,
        state.action_counts,
        state.packed.legal_action_mask,
    )
    _propagate_node_ranges(state)
    _forward_pass_gpu(state, strategy_table, state.node_range_p0, state.node_range_p1)
    backward_p0.zero_()
    backward_p1.zero_()
    _backward_pass_gpu(
        state,
        strategy_table,
        evaluator_impl,
        backward_p0,
        backward_p1,
    )
    phase_seconds["finalize"] += time.monotonic() - started_phase

    root_strategy = _average_strategy_from_gpu(
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


def _build_level_plan(
    tree: BuiltPublicTree,
    level: tuple[int, ...],
    actions_by_node: tuple[tuple[object, ...], ...],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    parents: list[int] = []
    children: list[int] = []
    node_types: list[int] = []
    infosets: list[int] = []
    action_slots: list[int] = []
    chance_probs: list[float] = []
    for node_index in level:
        if tree.tree.is_frontier[node_index]:
            continue
        node_type = tree.tree.node_types[node_index]
        start = tree.tree.first_child[node_index]
        count = tree.tree.child_count[node_index]
        if count <= 0:
            continue
        links = tree.tree.children[start : start + count]
        infoset_id = tree.tree.infoset_ids[node_index]
        for action_index, link in enumerate(links):
            parents.append(node_index)
            children.append(int(link.child))
            node_types.append(0 if node_type is NodeType.CHANCE else 1 if node_type is NodeType.PLAYER0 else 2 if node_type is NodeType.PLAYER1 else 3)
            infosets.append(int(infoset_id) if infoset_id is not None else -1)
            action_slots.append(action_index)
            chance_probs.append(float(link.chance_prob or 0.0))
    return {
        "parents": torch.as_tensor(parents, dtype=torch.int64, device=device),
        "children": torch.as_tensor(children, dtype=torch.int64, device=device),
        "node_types": torch.as_tensor(node_types, dtype=torch.int64, device=device),
        "infosets": torch.as_tensor(infosets, dtype=torch.int64, device=device),
        "action_slots": torch.as_tensor(action_slots, dtype=torch.int64, device=device),
        "chance_probs": torch.as_tensor(chance_probs, dtype=torch.float32, device=device),
    }


def _node_type_code(node_type: NodeType) -> int:
    if node_type is NodeType.CHANCE:
        return 0
    if node_type is NodeType.PLAYER0:
        return 1
    if node_type is NodeType.PLAYER1:
        return 2
    if node_type is NodeType.TERMINAL:
        return 3
    return 4


def _regret_matching_table(
    regrets: torch.Tensor,
    action_infoset_index: torch.Tensor,
    action_slot_index: torch.Tensor,
    action_counts: torch.Tensor,
) -> torch.Tensor:
    num_infosets = int(action_counts.numel())
    max_actions = int(action_counts.max().item()) if num_infosets else 1
    table = torch.zeros((num_infosets, max_actions), dtype=torch.float32, device=regrets.device)
    if num_infosets == 0:
        return table
    if action_infoset_index.numel() == 0:
        return table
    valid = (action_infoset_index >= 0) & (action_infoset_index < num_infosets) & (action_slot_index >= 0) & (action_slot_index < max_actions)
    if not bool(valid.any()):
        return table
    infosets = action_infoset_index[valid]
    slots = action_slot_index[valid]
    values = torch.clamp(regrets[valid], min=0.0)
    table.index_put_((infosets, slots), values, accumulate=False)
    totals = torch.zeros(num_infosets, dtype=torch.float32, device=regrets.device)
    totals.scatter_add_(0, infosets, values)
    nonzero = totals > 0
    if bool(nonzero.any()):
        table[nonzero] = table[nonzero] / totals[nonzero].unsqueeze(1)
    zero_rows = ~nonzero
    if bool(zero_rows.any()):
        counts = action_counts[zero_rows].clamp_min(1).to(torch.float32)
        table[zero_rows] = 0.0
        idx = torch.nonzero(zero_rows, as_tuple=False).flatten()
        for i, count in zip(idx.tolist(), counts.tolist(), strict=False):
            table[i, : int(count)] = 1.0 / float(count)
    return table


def _average_strategy_from_gpu(
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


def _make_cpu_store(
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


def _update_regrets_gpu(
    state: PackedGpuSolveState,
    regrets: torch.Tensor,
    strategy_sums: torch.Tensor,
    strategy_table: torch.Tensor,
    node_values_p0: torch.Tensor,
    node_values_p1: torch.Tensor,
) -> None:
    for level_index in range(len(state.level_edge_dst)):
        edge_src = state.level_edge_src[level_index]
        edge_dst = state.level_edge_dst[level_index]
        edge_infoset = state.level_edge_infoset[level_index]
        edge_slot = state.level_edge_slot[level_index]
        edge_kind = state.level_edge_kind[level_index]
        valid = (
            (edge_infoset >= 0)
            & (edge_infoset < strategy_table.shape[0])
            & (edge_slot >= 0)
            & (edge_slot < strategy_table.shape[1])
        )
        if not bool(valid.any()):
            continue
        edge_src = edge_src[valid]
        edge_dst = edge_dst[valid]
        edge_infoset = edge_infoset[valid]
        edge_slot = edge_slot[valid]
        edge_kind = edge_kind[valid]
        flat = state.action_offsets[edge_infoset] + edge_slot
        child_values = torch.where(edge_kind == 1, node_values_p0[edge_dst], node_values_p1[edge_dst])
        strat = strategy_table[edge_infoset, edge_slot]
        infoset_value = torch.zeros(state.action_counts.numel(), dtype=torch.float32, device=regrets.device)
        infoset_value.scatter_add_(0, edge_infoset, strat * child_values)
        regrets.index_add_(0, flat, child_values - infoset_value[edge_infoset])
        strategy_sums.index_add_(0, flat, strat)


def _update_regrets_gpu_batched(
    state: PackedGpuSolveState,
    regrets: torch.Tensor,
    strategy_sums: torch.Tensor,
    strategy_table: torch.Tensor,
    node_values_p0: torch.Tensor,
    node_values_p1: torch.Tensor,
) -> None:
    for level_index in range(len(state.level_edge_dst)):
        edge_src = state.level_edge_src[level_index]
        edge_dst = state.level_edge_dst[level_index]
        edge_infoset = state.level_edge_infoset[level_index]
        edge_slot = state.level_edge_slot[level_index]
        edge_kind = state.level_edge_kind[level_index]
        valid = (
            (edge_infoset >= 0)
            & (edge_infoset < strategy_table.shape[0])
            & (edge_slot >= 0)
            & (edge_slot < strategy_table.shape[1])
        )
        if not bool(valid.any()):
            continue
        edge_src = edge_src[valid]
        edge_dst = edge_dst[valid]
        edge_infoset = edge_infoset[valid]
        edge_slot = edge_slot[valid]
        edge_kind = edge_kind[valid]
        flat = state.action_offsets[edge_infoset] + edge_slot
        child_values = torch.where(edge_kind == 1, node_values_p0[edge_dst], node_values_p1[edge_dst])
        strat = strategy_table[edge_infoset, edge_slot]
        infoset_values = torch.zeros(state.action_counts.numel(), dtype=torch.float32, device=regrets.device)
        infoset_values.scatter_add_(0, edge_infoset, strat * child_values)
        strategy_sums.scatter_add_(0, flat, strat)
        regrets.index_add_(0, flat, child_values - infoset_values[edge_infoset])


def _forward_pass_gpu(*args: Any) -> None:
    if len(args) == 4 and isinstance(args[0], PackedGpuSolveState):
        state, strategy_table, node_range_p0, node_range_p1 = args
        _forward_pass_gpu_batched(state, strategy_table, node_range_p0, node_range_p1)
        return
    raise TypeError("_forward_pass_gpu received unsupported arguments")


def _forward_pass_gpu_batched(
    state: PackedGpuSolveState,
    strategy_table: torch.Tensor,
    node_range_p0: torch.Tensor,
    node_range_p1: torch.Tensor,
) -> None:
    _propagate_node_ranges(state)
    for level_index in range(len(state.level_edge_dst)):
        edge_src = state.level_edge_src[level_index]
        edge_dst = state.level_edge_dst[level_index]
        edge_infoset = state.level_edge_infoset[level_index]
        edge_slot = state.level_edge_slot[level_index]
        edge_kind = state.level_edge_kind[level_index]
        edge_prob = state.level_edge_prob[level_index]
        valid = (
            (edge_infoset >= 0)
            & (edge_infoset < strategy_table.shape[0])
            & (edge_slot >= 0)
            & (edge_slot < strategy_table.shape[1])
        )
        if not bool(valid.any()):
            continue
        edge_src = edge_src[valid]
        edge_dst = edge_dst[valid]
        edge_infoset = edge_infoset[valid]
        edge_slot = edge_slot[valid]
        edge_kind = edge_kind[valid]
        edge_prob = edge_prob[valid]
        chance_mask = edge_kind == 0
        if bool(chance_mask.any()):
            src = edge_src[chance_mask]
            dst = edge_dst[chance_mask]
            probs = edge_prob[chance_mask]
            node_range_p0.index_add_(0, dst, node_range_p0[src] * probs)
            node_range_p1.index_add_(0, dst, node_range_p1[src] * probs)
        player_mask = edge_kind == 1
        if bool(player_mask.any()):
            src = edge_src[player_mask]
            dst = edge_dst[player_mask]
            infosets = edge_infoset[player_mask]
            slots = edge_slot[player_mask]
            probs = strategy_table[infosets, slots]
            node_range_p0.index_add_(0, dst, node_range_p0[src] * probs)
            node_range_p1.index_add_(0, dst, node_range_p1[src])
        player_mask = edge_kind == 2
        if bool(player_mask.any()):
            src = edge_src[player_mask]
            dst = edge_dst[player_mask]
            infosets = edge_infoset[player_mask]
            slots = edge_slot[player_mask]
            probs = strategy_table[infosets, slots]
            node_range_p0.index_add_(0, dst, node_range_p0[src])
            node_range_p1.index_add_(0, dst, node_range_p1[src] * probs)


def _backward_pass_gpu(*args: Any) -> None:
    if len(args) == 5 and isinstance(args[0], PackedGpuSolveState):
        state, strategy_table, evaluator, out_p0, out_p1 = args
        _backward_pass_gpu_batched(state, strategy_table, evaluator, out_p0, out_p1)
        return
    raise TypeError("_backward_pass_gpu received unsupported arguments")


def _backward_pass_gpu_batched(
    state: PackedGpuSolveState,
    strategy_table: torch.Tensor,
    evaluator: LeafEvaluator,
    out_p0: torch.Tensor,
    out_p1: torch.Tensor,
) -> None:
    out_p0.zero_()
    out_p1.zero_()

    frontier_nodes = state.frontier_nodes
    if int(frontier_nodes.numel()) > 0:
        leaf_values = _evaluate_frontier_leaves(state, evaluator)
        out_p0[frontier_nodes] = torch.as_tensor(leaf_values.ev_player0, dtype=torch.float32, device=out_p0.device)
        out_p1[frontier_nodes] = torch.as_tensor(leaf_values.ev_player1, dtype=torch.float32, device=out_p1.device)

    for level_index in reversed(range(len(state.level_edge_dst))):
        edge_src = state.level_edge_src[level_index]
        edge_dst = state.level_edge_dst[level_index]
        edge_infoset = state.level_edge_infoset[level_index]
        edge_slot = state.level_edge_slot[level_index]
        edge_kind = state.level_edge_kind[level_index]
        edge_prob = state.level_edge_prob[level_index]
        valid = (
            (edge_infoset >= 0)
            & (edge_infoset < strategy_table.shape[0])
            & (edge_slot >= 0)
            & (edge_slot < strategy_table.shape[1])
        )
        if not bool(valid.any()):
            continue
        edge_src = edge_src[valid]
        edge_dst = edge_dst[valid]
        edge_infoset = edge_infoset[valid]
        edge_slot = edge_slot[valid]
        edge_kind = edge_kind[valid]
        edge_prob = edge_prob[valid]
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


def _evaluate_frontier_leaves(
    state: PackedGpuSolveState,
    evaluator: LeafEvaluator,
) -> LeafValueBatch:
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
    evaluate_tensors = getattr(evaluator, "evaluate_tensors", None)
    if evaluate_tensors is not None:
        try:
            return cast(LeafValueBatch, evaluate_tensors(tensors))
        except Exception:
            pass
    raise RuntimeError("GPU leaf evaluation requires evaluate_tensors support")


def _init_gpu_ranges(
    root_ranges: tuple[RangeVector, ...],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = []
    for index in range(6):
        if index < len(root_ranges):
            values = root_ranges[index].values
        else:
            values = np.zeros(1326, dtype=np.float32)
        tensors.append(torch.as_tensor(values, dtype=torch.float32, device=device))
    return tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5]


def _propagate_node_ranges(state: PackedGpuSolveState) -> None:
    state.node_range_p0.zero_()
    state.node_range_p1.zero_()
    state.node_range_p2.zero_()
    state.node_range_p0[0] = torch.ones_like(state.node_range_p0[0])
    state.node_range_p1[0] = torch.ones_like(state.node_range_p1[0])
    state.node_range_p2[0] = torch.ones_like(state.node_range_p2[0])
    for level_index in range(len(state.level_edge_dst)):
        edge_src = state.level_edge_src[level_index]
        edge_dst = state.level_edge_dst[level_index]
        edge_kind = state.level_edge_kind[level_index]
        edge_prob = state.level_edge_prob[level_index]
        if edge_src.numel() == 0:
            continue
        for src, dst, kind, prob in zip(edge_src.tolist(), edge_dst.tolist(), edge_kind.tolist(), edge_prob.tolist(), strict=True):
            src_i = int(src)
            dst_i = int(dst)
            if kind == 0:
                state.node_range_p0[dst_i] = state.node_range_p0[src_i] * prob
                state.node_range_p1[dst_i] = state.node_range_p1[src_i] * prob
                state.node_range_p2[dst_i] = state.node_range_p2[src_i] * prob
            elif kind == 1:
                probs = state.strategy_table[state.edge_infoset[dst_i] if dst_i < state.edge_infoset.numel() else 0]
                state.node_range_p0[dst_i] = state.node_range_p0[src_i]
                state.node_range_p1[dst_i] = state.node_range_p1[src_i]
                state.node_range_p2[dst_i] = state.node_range_p2[src_i]
            else:
                state.node_range_p0[dst_i] = state.node_range_p0[src_i]
                state.node_range_p1[dst_i] = state.node_range_p1[src_i]
                state.node_range_p2[dst_i] = state.node_range_p2[src_i]


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
    root_node = 0
    start = int(tree.first_child[root_node])
    count = int(tree.child_count[root_node])
    values: list[float] = []
    for action_index in range(min(count, int(plan.root_child_nodes.numel()))):
        child_node = int(tree.children[start + action_index].child)
        if child_node < 0 or child_node >= backward_p0.numel():
            values.append(0.0)
            continue
        values.append(float(backward_p0[child_node].item()))
    return np.asarray(values, dtype=np.float32)


def debug_first_gpu_cpu_divergence(
    spec: PostflopResolveSpec,
    *,
    evaluator: LeafEvaluator | None = None,
) -> None:
    evaluator_impl = evaluator or default_postflop_leaf_evaluator()
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

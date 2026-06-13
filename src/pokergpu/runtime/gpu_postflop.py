from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"torch is required for GPU postflop solving: {exc}") from exc

from pokergpu.abstraction.actions import (
    BaselineActionAbstraction,
    make_postflop_mvp_profile,
)
from pokergpu.cfr import InfosetLayout, InfosetStore, TreeLevels, build_tree_levels
from pokergpu.cfr.traversal import (
    build_leaf_feature_batch,
)
from pokergpu.core.board import Street
from pokergpu.core.canonical import canonical_board_key
from pokergpu.core.payouts import compute_payouts
from pokergpu.core.state import GameState, HandPhase
from pokergpu.eval import LeafEvaluator
from pokergpu.eval.types import LeafFeatureBatch
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
class BatchedGpuPlan:
    forward_levels: tuple[dict[str, torch.Tensor], ...]
    backward_levels: tuple[dict[str, torch.Tensor], ...]
    level_nodes: tuple[torch.Tensor, ...]
    edge_parent: torch.Tensor
    edge_child: torch.Tensor
    edge_node_type: torch.Tensor
    edge_infoset: torch.Tensor
    edge_action_slot: torch.Tensor
    edge_chance_prob: torch.Tensor
    node_infoset: torch.Tensor
    node_type: torch.Tensor
    frontier_nodes: torch.Tensor
    frontier_leaf_batch: LeafFeatureBatch
    root_child_nodes: torch.Tensor
    root_child_parent_infoset: int
    action_counts: torch.Tensor
    action_offsets: torch.Tensor


@dataclass(frozen=True, slots=True)
class PackedGpuSolve:
    spec: PostflopResolveSpec
    tree: BuiltPublicTree
    plan: BatchedGpuPlan
    layout: InfosetLayout
    root_infoset: int
    root_actions: tuple[str, ...]
    packed_subtree: PackedGpuSubtree
    gpu_state: PackedGpuSolveState | None = None


@dataclass(frozen=True, slots=True)
class BatchedGpuSolveInput:
    spec: PostflopResolveSpec
    template: PublicTreeTemplate
    cache_key: str = ""


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
    cpu_backward_p0: np.ndarray
    cpu_backward_p1: np.ndarray


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


def _root_child_nodes(tree: PublicTree) -> tuple[int, ...]:
    start = tree.first_child[0]
    count = tree.child_count[0]
    return tuple(int(tree.children[start + index].child) for index in range(count))


def _build_batched_gpu_plan(
    tree: PublicTree,
    actions_by_node: tuple[tuple[object, ...], ...],
    levels: TreeLevels,
    layout: InfosetLayout,
    *,
    frontier_leaf_batch: LeafFeatureBatch,
    device: torch.device,
) -> BatchedGpuPlan:
    forward_levels: list[dict[str, torch.Tensor]] = []
    backward_levels: list[dict[str, torch.Tensor]] = []
    edge_parent: list[int] = []
    edge_child: list[int] = []
    edge_node_type: list[int] = []
    edge_infoset: list[int] = []
    edge_action_slot: list[int] = []
    edge_chance_prob: list[float] = []
    node_infoset: list[int] = []
    node_type: list[int] = []
    frontier_nodes = [index for index in range(tree.node_count) if tree.is_frontier[index] and tree.node_types[index] is not NodeType.TERMINAL]
    root_children = _root_child_nodes(tree)
    action_counts = torch.as_tensor(layout.action_counts, dtype=torch.int64, device=device)
    action_offsets = torch.as_tensor(layout.offsets, dtype=torch.int64, device=device)
    level_nodes = tuple(
        torch.as_tensor(level, dtype=torch.int64, device=device)
        for level in levels.forward_levels
    )
    for level in levels.forward_levels:
        forward_levels.append(_build_level_plan(tree, level, actions_by_node, device=device))
    for level in levels.backward_levels:
        backward_levels.append(_build_level_plan(tree, level, actions_by_node, device=device))
    for node_index in range(tree.node_count):
        code = _node_type_code(tree.node_types[node_index])
        node_type.append(code)
        infoset_id = tree.infoset_ids[node_index]
        node_infoset.append(int(infoset_id) if infoset_id is not None else -1)
        start = tree.first_child[node_index]
        count = tree.child_count[node_index]
        for action_index, link in enumerate(tree.children[start : start + count]):
            edge_parent.append(node_index)
            edge_child.append(int(link.child))
            edge_node_type.append(code)
            edge_infoset.append(int(infoset_id) if infoset_id is not None else -1)
            edge_action_slot.append(action_index)
            edge_chance_prob.append(float(link.chance_prob or 0.0))
    return BatchedGpuPlan(
        forward_levels=tuple(forward_levels),
        backward_levels=tuple(backward_levels),
        edge_parent=torch.as_tensor(edge_parent, dtype=torch.int64, device=device),
        edge_child=torch.as_tensor(edge_child, dtype=torch.int64, device=device),
        edge_node_type=torch.as_tensor(edge_node_type, dtype=torch.int64, device=device),
        edge_infoset=torch.as_tensor(edge_infoset, dtype=torch.int64, device=device),
        edge_action_slot=torch.as_tensor(edge_action_slot, dtype=torch.int64, device=device),
        edge_chance_prob=torch.as_tensor(edge_chance_prob, dtype=torch.float32, device=device),
        level_nodes=level_nodes,
        node_infoset=torch.as_tensor(node_infoset, dtype=torch.int64, device=device),
        node_type=torch.as_tensor(node_type, dtype=torch.int64, device=device),
        frontier_nodes=torch.as_tensor(frontier_nodes, dtype=torch.int64, device=device),
        frontier_leaf_batch=frontier_leaf_batch,
        root_child_nodes=torch.as_tensor(root_children, dtype=torch.int64, device=device),
        root_child_parent_infoset=int(tree.infoset_ids[0] or 0),
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
            gpu_state=_make_gpu_state(cached.packed_subtree, cached.layout),
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
        tree.tree,
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
    )
    packed = PackedGpuSolve(
        spec=packed.spec,
        tree=packed.tree,
        plan=packed.plan,
        layout=packed.layout,
        root_infoset=packed.root_infoset,
        root_actions=packed.root_actions,
        packed_subtree=packed.packed_subtree,
        gpu_state=_make_gpu_state(packed.packed_subtree, packed.layout),
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


def _make_gpu_state(packed: PackedGpuSubtree, layout: InfosetLayout) -> PackedGpuSolveState:
    device = packed.node_type.device
    action_count = int(layout.total_actions)
    regrets = torch.zeros(action_count, dtype=torch.float32, device=device)
    strategy_sums = torch.zeros_like(regrets)
    strategy_table = torch.zeros((packed.infoset_count, max(packed.max_actions, 1)), dtype=torch.float32, device=device)
    strategy_flat = torch.zeros(action_count, dtype=torch.float32, device=device)
    forward_reach_p0 = torch.zeros(packed.node_count, dtype=torch.float32, device=device)
    forward_reach_p1 = torch.zeros_like(forward_reach_p0)
    backward_p0 = torch.zeros_like(forward_reach_p0)
    backward_p1 = torch.zeros_like(forward_reach_p0)
    leaf_values_p0 = torch.zeros(packed.leaf_count, dtype=torch.float32, device=device)
    leaf_values_p1 = torch.zeros_like(leaf_values_p0)
    root_action_ev_p0 = torch.zeros(packed.max_actions, dtype=torch.float32, device=device)
    root_action_ev_p1 = torch.zeros_like(root_action_ev_p0)
    root_strategy = torch.zeros_like(root_action_ev_p0)
    return PackedGpuSolveState(
        packed=packed,
        regrets=regrets,
        strategy_sums=strategy_sums,
        strategy_table=strategy_table,
        strategy_flat=strategy_flat,
        action_infoset_index=packed.action_infoset_index,
        action_slot_index=packed.action_slot_index,
        action_offsets=torch.as_tensor(layout.offsets, dtype=torch.int64, device=device),
        action_counts=torch.as_tensor(layout.action_counts, dtype=torch.int64, device=device),
        forward_reach_p0=forward_reach_p0,
        forward_reach_p1=forward_reach_p1,
        backward_p0=backward_p0,
        backward_p1=backward_p1,
        leaf_values_p0=leaf_values_p0,
        leaf_values_p1=leaf_values_p1,
        root_action_ev_p0=root_action_ev_p0,
        root_action_ev_p1=root_action_ev_p1,
        root_strategy=root_strategy,
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
) -> GpuSolveTrace:
    spec = packed.spec
    tree = packed.tree
    plan = packed.plan
    layout = packed.layout
    root_infoset = packed.root_infoset
    started_at = time.monotonic()
    device = torch.device("cuda")
    node_count = tree.tree.node_count
    leaf_count = int(plan.frontier_nodes.numel())
    state = packed.gpu_state
    if state is None:
        state = _make_gpu_state(packed.packed_subtree, packed.layout)
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
    regrets = state.regrets
    strategy_sums = state.strategy_sums
    forward_reach_p0 = state.forward_reach_p0
    forward_reach_p1 = state.forward_reach_p1
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
        strategy_table = _regret_matching_table(
            regrets,
            state.action_infoset_index,
            state.action_slot_index,
            plan.action_counts,
        )
        phase_seconds["strategy"] += time.monotonic() - started_phase

        started_phase = time.monotonic()
        forward_reach_p0.zero_()
        forward_reach_p1.zero_()
        forward_reach_p0[0] = 1.0
        forward_reach_p1[0] = 1.0
        _forward_pass_gpu(tree.tree, plan, strategy_table, forward_reach_p0, forward_reach_p1)
        phase_seconds["forward"] += time.monotonic() - started_phase

        started_phase = time.monotonic()
        backward_p0.zero_()
        backward_p1.zero_()
        _backward_pass_gpu(
            tree.tree,
            plan,
            strategy_table,
            evaluator_impl,
            tree.node_states,
            backward_p0,
            backward_p1,
        )
        phase_seconds["backward"] += time.monotonic() - started_phase

        started_phase = time.monotonic()
        _update_regrets_gpu(
            tree.tree,
            plan,
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
    strategy_table = _regret_matching_table(
        regrets,
        state.action_infoset_index,
        state.action_slot_index,
        plan.action_counts,
    )
    forward_reach_p0.zero_()
    forward_reach_p1.zero_()
    forward_reach_p0[0] = 1.0
    forward_reach_p1[0] = 1.0
    _forward_pass_gpu(tree.tree, plan, strategy_table, forward_reach_p0, forward_reach_p1)
    backward_p0.zero_()
    backward_p1.zero_()
    _backward_pass_gpu(
        tree.tree,
        plan,
        strategy_table,
        evaluator_impl,
        tree.node_states,
        backward_p0,
        backward_p1,
    )
    phase_seconds["finalize"] += time.monotonic() - started_phase

    root_strategy = _average_strategy_from_gpu(
        strategy_sums,
        plan.action_counts,
        plan.action_offsets,
        root_infoset,
    )
    root_action_ev_p0 = _root_action_values_from_backward(
        tree.tree,
        plan,
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
        cpu_backward_p0=backward_p0.detach().cpu().numpy(),
        cpu_backward_p1=backward_p1.detach().cpu().numpy(),
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
    tree: PublicTree,
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
        if tree.is_frontier[node_index]:
            continue
        node_type = tree.node_types[node_index]
        start = tree.first_child[node_index]
        count = tree.child_count[node_index]
        if count <= 0:
            continue
        links = tree.children[start : start + count]
        infoset_id = tree.infoset_ids[node_index]
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
    totals = torch.zeros(num_infosets, dtype=torch.float32, device=regrets.device)
    valid_rows: list[tuple[int, int, float]] = []
    for i in range(int(action_infoset_index.numel())):
        infoset_index = int(action_infoset_index[i].item())
        slot_index = int(action_slot_index[i].item())
        if infoset_index < 0 or infoset_index >= num_infosets:
            continue
        if slot_index < 0 or slot_index >= max_actions:
            continue
        value = float(torch.clamp(regrets[i], min=0.0).item())
        totals[infoset_index] += value
        valid_rows.append((infoset_index, slot_index, value))
    if not valid_rows:
        return table
    for infoset_index, slot_index, value in valid_rows:
        total = float(totals[infoset_index].item())
        table[infoset_index, slot_index] = 0.0 if total <= 0.0 else value / total
    for infoset_index in range(num_infosets):
        if float(totals[infoset_index].item()) > 0.0:
            continue
        count = int(action_counts[infoset_index].item())
        if count > 0:
            table[infoset_index, :count] = 1.0 / float(count)
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
    tree: PublicTree,
    plan: BatchedGpuPlan,
    regrets: torch.Tensor,
    strategy_sums: torch.Tensor,
    strategy_table: torch.Tensor,
    node_values_p0: torch.Tensor,
    node_values_p1: torch.Tensor,
) -> None:
    action_offsets = plan.action_offsets
    for level in plan.level_nodes:
        for node_index in level.tolist():
            infoset_index = int(plan.node_infoset[node_index].item())
            if infoset_index < 0 or infoset_index >= action_offsets.numel():
                continue
            node_type = tree.node_types[node_index]
            if node_type not in {NodeType.PLAYER0, NodeType.PLAYER1}:
                continue
            count = int(plan.action_counts[infoset_index].item())
            start = int(action_offsets[infoset_index].item())
            child_count = int(tree.child_count[node_index])
            limit = min(count, child_count)
            if limit <= 0:
                continue
            children_start = int(tree.first_child[node_index])
            child_values = torch.empty(limit, dtype=torch.float32, device=regrets.device)
            for action_index in range(limit):
                child = int(tree.children[children_start + action_index].child)
                child_values[action_index] = (
                    node_values_p0[child]
                    if node_type is NodeType.PLAYER0
                    else node_values_p1[child]
                )
            strat = strategy_table[infoset_index, :limit]
            infoset_value = torch.sum(strat * child_values)
            regrets[start : start + limit] += child_values - infoset_value
            strategy_sums[start : start + limit] += strat


def _update_regrets_gpu_batched(
    tree: PublicTree,
    plan: BatchedGpuPlan,
    regrets: torch.Tensor,
    strategy_sums: torch.Tensor,
    strategy_table: torch.Tensor,
    node_values_p0: torch.Tensor,
    node_values_p1: torch.Tensor,
) -> None:
    edge_child = plan.edge_child
    edge_infoset = plan.edge_infoset
    edge_action_slot = plan.edge_action_slot
    edge_node_type = plan.edge_node_type
    edge_offsets = plan.action_offsets
    if edge_child.numel() == 0:
        return
    infoset_values = torch.zeros(plan.action_counts.numel(), dtype=torch.float32, device=regrets.device)
    for i in range(int(edge_child.numel())):
        infoset_index = int(edge_infoset[i].item())
        slot_index = int(edge_action_slot[i].item())
        if infoset_index < 0 or infoset_index >= plan.action_counts.numel():
            continue
        if slot_index < 0 or slot_index >= int(plan.action_counts.max().item()):
            continue
        if slot_index >= int(plan.action_counts[infoset_index].item()):
            continue
        child_index = int(edge_child[i].item())
        if child_index < 0 or child_index >= node_values_p0.numel():
            continue
        if int(edge_node_type[i].item()) == 1:
            child_value = float(node_values_p0[child_index].item())
        else:
            child_value = float(node_values_p1[child_index].item())
        flat_index = int(edge_offsets[infoset_index].item()) + slot_index
        if flat_index < 0 or flat_index >= regrets.numel():
            continue
        strat = float(strategy_table[infoset_index, slot_index].item())
        infoset_values[infoset_index] += strat * child_value
        strategy_sums[flat_index] += strat
    for i in range(int(edge_child.numel())):
        infoset_index = int(edge_infoset[i].item())
        slot_index = int(edge_action_slot[i].item())
        if infoset_index < 0 or infoset_index >= plan.action_counts.numel():
            continue
        if slot_index < 0 or slot_index >= int(plan.action_counts.max().item()):
            continue
        if slot_index >= int(plan.action_counts[infoset_index].item()):
            continue
        flat_index = int(edge_offsets[infoset_index].item()) + slot_index
        if flat_index < 0 or flat_index >= regrets.numel():
            continue
        child_index = int(edge_child[i].item())
        if child_index < 0 or child_index >= node_values_p0.numel():
            continue
        if int(edge_node_type[i].item()) == 1:
            child_value = float(node_values_p0[child_index].item())
        else:
            child_value = float(node_values_p1[child_index].item())
        action_regret = child_value - float(infoset_values[infoset_index].item())
        regrets[flat_index] += action_regret


def _forward_pass_gpu(*args: Any) -> None:
    if len(args) == 5 and isinstance(args[1], BatchedGpuPlan):
        tree, plan, strategy_table, reach_p0, reach_p1 = args
        _forward_pass_gpu_batched(tree, plan, strategy_table, reach_p0, reach_p1)
        return
    if len(args) == 6:
        tree, levels, store, reach_p0, reach_p1, device = args
        _forward_pass_gpu_legacy(tree, levels, store, reach_p0, reach_p1, device)
        return
    raise TypeError("_forward_pass_gpu received unsupported arguments")


def _forward_pass_gpu_batched(
    tree: PublicTree,
    plan: BatchedGpuPlan,
    strategy_table: torch.Tensor,
    reach_p0: torch.Tensor,
    reach_p1: torch.Tensor,
) -> None:
    for level in plan.level_nodes:
        for node_index in level.tolist():
            if tree.is_frontier[node_index]:
                continue
            node_type = tree.node_types[node_index]
            if node_type in {NodeType.LEAF, NodeType.TERMINAL}:
                continue
            start = tree.first_child[node_index]
            count = tree.child_count[node_index]
            if count <= 0:
                continue
            links = tree.children[start : start + count]
            if node_type is NodeType.CHANCE:
                for link in links:
                    if link.chance_prob is None:
                        continue
                    child_index = int(link.child)
                    chance = float(link.chance_prob)
                    reach_p0[child_index] += reach_p0[node_index] * chance
                    reach_p1[child_index] += reach_p1[node_index] * chance
                continue
            infoset_id = tree.infoset_ids[node_index]
            if infoset_id is None:
                continue
            infoset_index = int(plan.node_infoset[node_index].item())
            if infoset_index < 0 or infoset_index >= strategy_table.shape[0]:
                continue
            strategy = strategy_table[infoset_index]
            limit = min(count, int(strategy.shape[0]))
            for action_index, link in enumerate(links[:limit]):
                child_index = int(link.child)
                prob = float(strategy[action_index].item())
                if node_type is NodeType.PLAYER0:
                    reach_p0[child_index] += reach_p0[node_index] * prob
                    reach_p1[child_index] += reach_p1[node_index]
                elif node_type is NodeType.PLAYER1:
                    reach_p0[child_index] += reach_p0[node_index]
                    reach_p1[child_index] += reach_p1[node_index] * prob
                else:
                    reach_p0[child_index] += reach_p0[node_index]
                    reach_p1[child_index] += reach_p1[node_index]


def _forward_pass_gpu_legacy(
    tree: PublicTree,
    levels: TreeLevels,
    store: InfosetStore,
    reach_p0: torch.Tensor,
    reach_p1: torch.Tensor,
    device: torch.device,
) -> None:
    for level in levels.forward_levels:
        for node_index in level:
            node_type = tree.node_types[node_index]
            if tree.is_frontier[node_index] or node_type in {NodeType.LEAF, NodeType.TERMINAL}:
                continue
            start = tree.first_child[node_index]
            count = tree.child_count[node_index]
            children = tree.children[start : start + count]
            if node_type is NodeType.CHANCE:
                for link in children:
                    if link.chance_prob is None:
                        continue
                    chance = torch.tensor(float(link.chance_prob), device=device)
                    reach_p0[int(link.child)] += reach_p0[node_index] * chance
                    reach_p1[int(link.child)] += reach_p1[node_index] * chance
                continue
            infoset_id = tree.infoset_ids[node_index]
            if infoset_id is None:
                continue
            strategy = torch.as_tensor(
                store.current_strategy(int(infoset_id)),
                dtype=torch.float32,
                device=device,
            )
            limit = min(len(children), int(strategy.shape[0]))
            for action_index, link in enumerate(children[:limit]):
                child_index = int(link.child)
                action_prob = strategy[action_index]
                if node_type is NodeType.PLAYER0:
                    reach_p0[child_index] += reach_p0[node_index] * action_prob
                    reach_p1[child_index] += reach_p1[node_index]
                else:
                    reach_p0[child_index] += reach_p0[node_index]
                    reach_p1[child_index] += reach_p1[node_index] * action_prob


def _backward_pass_gpu(*args: Any) -> None:
    if len(args) == 7 and isinstance(args[1], BatchedGpuPlan):
        tree, plan, strategy_table, evaluator, node_states, out_p0, out_p1 = args
        _backward_pass_gpu_batched(tree, plan, strategy_table, evaluator, node_states, out_p0, out_p1)
        return
    if len(args) == 10:
        tree, levels, store, reach_p0, reach_p1, evaluator, node_states, out_p0, out_p1, device = args
        _backward_pass_gpu_legacy(tree, levels, store, reach_p0, reach_p1, evaluator, node_states, out_p0, out_p1, device)
        return
    raise TypeError("_backward_pass_gpu received unsupported arguments")


def _backward_pass_gpu_batched(
    tree: PublicTree,
    plan: BatchedGpuPlan,
    strategy_table: torch.Tensor,
    evaluator: LeafEvaluator,
    node_states: tuple[GameState, ...],
    out_p0: torch.Tensor,
    out_p1: torch.Tensor,
) -> None:
    for node_index, payoff in enumerate(tree.terminal_payoffs):
        if payoff is not None:
            out_p0[node_index] = float(payoff)
            out_p1[node_index] = -float(payoff)

    frontier_nodes = tuple(int(index) for index in plan.frontier_nodes.tolist())
    if frontier_nodes:
        leaf_values = evaluator.evaluate(plan.frontier_leaf_batch)
        for row, node_index in enumerate(frontier_nodes):
            out_p0[node_index] = float(leaf_values.ev_player0[row])
            out_p1[node_index] = float(leaf_values.ev_player1[row])

    for level in reversed(plan.level_nodes):
        for node_index in level.tolist():
            if tree.is_frontier[node_index]:
                continue
            node_type = tree.node_types[node_index]
            if node_type in {NodeType.LEAF, NodeType.TERMINAL}:
                continue
            start = tree.first_child[node_index]
            count = tree.child_count[node_index]
            if count <= 0:
                continue
            child_links = tree.children[start : start + count]
            child_indices = torch.as_tensor(
                [int(link.child) for link in child_links],
                dtype=torch.int64,
                device=out_p0.device,
            )
            child_p0 = out_p0[child_indices]
            child_p1 = out_p1[child_indices]
            if node_type is NodeType.CHANCE:
                probs = torch.as_tensor(
                    [float(link.chance_prob or 0.0) for link in child_links],
                    dtype=torch.float32,
                    device=out_p0.device,
                )
                out_p0[node_index] = torch.sum(probs * child_p0)
                out_p1[node_index] = torch.sum(probs * child_p1)
                continue
            infoset_id = tree.infoset_ids[node_index]
            if infoset_id is None:
                continue
            infoset_index = int(plan.node_infoset[node_index].item())
            if infoset_index < 0 or infoset_index >= strategy_table.shape[0]:
                continue
            limit = min(count, int(strategy_table.shape[1]))
            if limit <= 0:
                continue
            strategy = strategy_table[infoset_index, :limit]
            out_p0[node_index] = torch.sum(strategy * child_p0[:limit])
            out_p1[node_index] = torch.sum(strategy * child_p1[:limit])


def _backward_pass_gpu_legacy(
    tree: PublicTree,
    levels: TreeLevels,
    store: InfosetStore,
    reach_p0: torch.Tensor,
    reach_p1: torch.Tensor,
    evaluator: LeafEvaluator,
    node_states: tuple[GameState, ...],
    out_p0: torch.Tensor,
    out_p1: torch.Tensor,
    device: torch.device,
) -> None:
    frontier_nodes = tuple(
        index
        for index in range(tree.node_count)
        if tree.is_frontier[index] and tree.node_types[index] is not NodeType.TERMINAL
    )
    if frontier_nodes:
        batch = build_leaf_feature_batch(
            tree,
            frontier_nodes,
            node_states=node_states,
            reach_p0=reach_p0.detach().cpu().numpy(),
            reach_p1=reach_p1.detach().cpu().numpy(),
        )
        leaf_values = evaluator.evaluate(batch)
        for row, node_index in enumerate(frontier_nodes):
            out_p0[node_index] = float(leaf_values.ev_player0[row])
            out_p1[node_index] = float(leaf_values.ev_player1[row])

    for level in levels.backward_levels:
        for node_index in level:
            node_type = tree.node_types[node_index]
            if node_type is NodeType.TERMINAL:
                out_p0[node_index] = float(tree.terminal_payoffs[node_index] or 0.0)
                out_p1[node_index] = -out_p0[node_index]
                continue
            if tree.is_frontier[node_index]:
                continue
            start = tree.first_child[node_index]
            count = tree.child_count[node_index]
            children = tree.children[start : start + count]
            if not children:
                continue
            child_p0 = torch.stack([out_p0[int(link.child)] for link in children])
            child_p1 = torch.stack([out_p1[int(link.child)] for link in children])
            if node_type is NodeType.CHANCE:
                probs = torch.tensor(
                    [float(link.chance_prob or 0.0) for link in children],
                    dtype=torch.float32,
                    device=device,
                )
                out_p0[node_index] = torch.sum(probs * child_p0)
                out_p1[node_index] = torch.sum(probs * child_p1)
                continue
            infoset_id = tree.infoset_ids[node_index]
            if infoset_id is None:
                continue
            strategy = torch.as_tensor(
                store.current_strategy(int(infoset_id)),
                dtype=torch.float32,
                device=device,
            )
            limit = min(int(strategy.shape[0]), child_p0.shape[0])
            out_p0[node_index] = torch.sum(strategy[:limit] * child_p0[:limit])
            out_p1[node_index] = torch.sum(strategy[:limit] * child_p1[:limit])


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
    cpu_nodes = trace.cpu_backward_p0[
        packed.plan.root_child_nodes.detach().cpu().numpy()
    ]
    print("root_child_nodes", packed.plan.root_child_nodes.detach().cpu().tolist())
    print("gpu_root_action_ev", gpu_nodes.tolist())
    print("cpu_root_action_ev", cpu_nodes.tolist())
    branch_node = None
    limit = min(gpu_nodes.shape[0], cpu_nodes.shape[0])
    for index in range(limit):
        if not np.isclose(gpu_nodes[index], cpu_nodes[index]):
            branch_node = int(packed.plan.root_child_nodes[index].item())
            print(
                "gpu_cpu_divergence",
                {
                    "index": index,
                    "gpu": float(gpu_nodes[index]),
                    "cpu": float(cpu_nodes[index]),
                    "root_infoset": packed.root_infoset,
                    "root_action": packed.root_actions[index] if index < len(packed.root_actions) else "",
                    "root_child": branch_node,
                    "child_node_type": str(packed.tree.tree.node_types[branch_node]),
                    "child_state": packed.tree.node_states[branch_node],
                },
            )
            break
    if branch_node is None and limit > 2:
        branch_node = int(packed.plan.root_child_nodes[2].item())
    if branch_node is not None:
        _print_first_branch_divergence(
            tree.tree,
            tree.node_states,
            branch_node,
            trace,
        )
        return
    print("gpu_cpu_divergence", {"status": "no divergence found"})


def _print_first_branch_divergence(
    tree: PublicTree,
    node_states: tuple[GameState, ...],
    start_node: int,
    trace: GpuSolveTrace,
) -> None:
    gpu_values = trace.gpu_backward_p0
    cpu_values = trace.cpu_backward_p0
    if gpu_values is None or cpu_values is None:
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
        if node_index < len(gpu_values) and node_index < len(cpu_values):
            gpu_value = float(gpu_values[node_index])
            cpu_value = float(cpu_values[node_index])
            if not np.isclose(gpu_value, cpu_value):
                print(
                    "gpu_cpu_branch_divergence",
                    {
                        "node": node_index,
                        "gpu": gpu_value,
                        "cpu": cpu_value,
                        "node_type": str(node_type),
                        "state": node_states[node_index],
                    },
                )
                return
        start = tree.first_child[node_index]
        count = tree.child_count[node_index]
        for child_link in reversed(tree.children[start : start + count]):
            stack.append(int(child_link.child))

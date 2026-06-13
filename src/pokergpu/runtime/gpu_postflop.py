from __future__ import annotations

import time
from dataclasses import dataclass

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
from pokergpu.cfr.traversal import build_leaf_feature_batch
from pokergpu.cfr.traversal import compute_counterfactual_values
from pokergpu.core.board import Street
from pokergpu.core.payouts import compute_payouts, total_pot
from pokergpu.core.state import GameState, HandPhase
from pokergpu.eval import LeafEvaluator
from pokergpu.tree import NodeType, PublicTree
from pokergpu.tree.builder import BuiltPublicTree, TreeBuildConfig, build_public_tree

from .postflop import PostflopResolveResult, PostflopResolveSpec
from .value_network import default_postflop_leaf_evaluator


@dataclass(frozen=True, slots=True)
class GpuSolveStats:
    iterations: int
    node_count: int
    leaf_count: int
    elapsed_seconds: float


def resolve_postflop_gpu(
    spec: PostflopResolveSpec,
    *,
    evaluator: LeafEvaluator | None = None,
) -> PostflopResolveResult:
    if spec.state.player_count != 2:
        raise ValueError("GPU postflop solver currently supports heads-up only")
    if spec.state.current_street is Street.PREFLOP:
        raise ValueError("postflop resolver requires a postflop state")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for resolve_postflop_gpu")

    evaluator_impl = evaluator or default_postflop_leaf_evaluator()
    started_at = time.monotonic()
    tree = build_public_tree(
        spec.state,
        abstraction=BaselineActionAbstraction(profile=make_postflop_mvp_profile()),
        config=TreeBuildConfig(
            max_depth=spec.max_depth,
            max_nodes=spec.max_nodes,
            min_reach_prob=spec.min_reach_prob,
        ),
    )
    levels = build_tree_levels(tree.tree)
    action_counts = _build_infoset_action_counts(tree.tree, tree.actions_by_node)
    store = InfosetStore.zeros(InfosetLayout.from_action_counts(action_counts))
    root_infoset = tree.tree.infoset_ids[0]
    if root_infoset is None:
        raise ValueError("root node must be a player infoset")
    root_actions = tuple(_format_action(action) for action in tree.actions_by_node[0])

    device = torch.device("cuda")
    node_count = tree.tree.node_count
    node_values_p0 = torch.zeros(node_count, dtype=torch.float32, device=device)
    node_values_p1 = torch.zeros(node_count, dtype=torch.float32, device=device)
    leaf_nodes = tuple(
        index
        for index in range(node_count)
        if tree.tree.is_frontier[index] and tree.tree.node_types[index] is not NodeType.TERMINAL
    )
    leaf_count = len(leaf_nodes)
    terminal_values_player0 = np.asarray(
        [
            np.float32(_terminal_value_player0(node_state))
            for node_state in tree.node_states
        ],
        dtype=np.float32,
    )

    root_strategy = np.full(len(root_actions), 1.0 / max(1, len(root_actions)), dtype=np.float32)
    forward_reach_p0 = torch.zeros(node_count, dtype=torch.float32, device=device)
    forward_reach_p1 = torch.zeros(node_count, dtype=torch.float32, device=device)
    backward_p0 = torch.zeros(node_count, dtype=torch.float32, device=device)
    backward_p1 = torch.zeros(node_count, dtype=torch.float32, device=device)

    iterations = 0
    deadline = time.monotonic() + max(0.0, spec.time_budget_sec)
    target_iterations = max(0, int(spec.iterations))
    while (
        (target_iterations > 0 and iterations < target_iterations)
        or (target_iterations <= 0 and (time.monotonic() < deadline or iterations == 0))
    ):
        forward_reach_p0.zero_()
        forward_reach_p1.zero_()
        forward_reach_p0[0] = 1.0
        forward_reach_p1[0] = 1.0
        _forward_pass_gpu(
            tree.tree,
            levels,
            store,
            forward_reach_p0,
            forward_reach_p1,
            device,
        )

        backward_p0.zero_()
        backward_p1.zero_()
        _backward_pass_gpu(
            tree.tree,
            levels,
            store,
            forward_reach_p0,
            forward_reach_p1,
            evaluator_impl,
            tree.node_states,
            backward_p0,
            backward_p1,
            device,
        )
        _update_store_from_gpu(tree.tree, store, backward_p0, backward_p1)
        iterations += 1
        if target_iterations > 0 and iterations >= target_iterations:
            break
        if spec.time_budget_sec <= 0.0 and target_iterations <= 0:
            break

    bb_scale = float(spec.state.betting_round.blinds.big_blind)
    if bb_scale <= 0.0:
        bb_scale = 1.0
    root_strategy = store.average_strategy(int(root_infoset))
    cpu_backward = compute_counterfactual_values(
        tree.tree,
        store,
        node_states=tree.node_states,
        reach_p0=forward_reach_p0.detach().cpu().numpy(),
        reach_p1=forward_reach_p1.detach().cpu().numpy(),
        terminal_values_player0=terminal_values_player0,
        evaluator=evaluator_impl,
    )
    root_action_nodes = _root_child_nodes(tree.tree)
    root_action_ev_p0 = np.asarray(
        [float(cpu_backward.node_values_player0[index]) for index in root_action_nodes],
        dtype=np.float32,
    )
    root_action_ev_p1 = -root_action_ev_p0
    root_action_ev_p0 = np.asarray(root_action_ev_p0 / bb_scale, dtype=np.float32)
    root_action_ev_p1 = np.asarray(root_action_ev_p1 / bb_scale, dtype=np.float32)
    root_ev_p0 = float(
        np.sum(
            root_strategy[: root_action_ev_p0.shape[0]] * root_action_ev_p0,
            dtype=np.float64,
        )
    )
    root_ev_p1 = -root_ev_p0
    return PostflopResolveResult(
        root_infoset_id=int(root_infoset),
        root_actions=root_actions,
        root_strategy=root_strategy,
        root_action_ev_player0=root_action_ev_p0,
        root_action_ev_player1=root_action_ev_p1,
        root_ev_player0=root_ev_p0,
        root_ev_player1=root_ev_p1,
        iterations=iterations,
        elapsed_seconds=time.monotonic() - started_at,
        node_count=node_count,
        leaf_count=leaf_count,
    )


def _root_child_nodes(tree: PublicTree) -> tuple[int, ...]:
    start = tree.first_child[0]
    count = tree.child_count[0]
    return tuple(int(tree.children[start + index].child) for index in range(count))


def _forward_pass_gpu(
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


def _backward_pass_gpu(
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
    for node_index, infoset_id in enumerate(tree.infoset_ids):
        if infoset_id is None or tree.node_types[node_index] not in {NodeType.PLAYER0, NodeType.PLAYER1}:
            continue
        action_values = []
        start = tree.first_child[node_index]
        count = tree.child_count[node_index]
        children = tree.children[start : start + count]
        for link in children:
            if tree.node_types[node_index] is NodeType.PLAYER0:
                action_values.append(float(node_values_p0[int(link.child)]))
            else:
                action_values.append(float(node_values_p1[int(link.child)]))
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
        counts.setdefault(infoset_index, len(actions_by_node[node_index]) or 1)
    return tuple(counts.get(index, 1) for index in range(max_infoset + 1))


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

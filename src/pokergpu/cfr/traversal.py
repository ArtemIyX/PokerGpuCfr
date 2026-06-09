from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pokergpu.tree import NodeId, NodeType, PublicTree

from .infosets import InfosetStore


@dataclass(frozen=True, slots=True)
class ForwardPassResult:
    player0_reach: NDArray[np.float32]
    player1_reach: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class BackwardPassResult:
    node_values_player0: NDArray[np.float32]
    node_values_player1: NDArray[np.float32]
    infoset_action_values: dict[int, NDArray[np.float32]]


@dataclass(frozen=True, slots=True)
class RegretUpdateResult:
    infoset_values: NDArray[np.float32]
    updated_infosets: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TreeLevels:
    forward_levels: tuple[tuple[int, ...], ...]
    backward_levels: tuple[tuple[int, ...], ...]


def compute_reach_probabilities(
    tree: PublicTree,
    store: InfosetStore,
    *,
    root_player0_reach: float = 1.0,
    root_player1_reach: float = 1.0,
) -> ForwardPassResult:
    player0_reach = np.zeros(tree.node_count, dtype=np.float32)
    player1_reach = np.zeros(tree.node_count, dtype=np.float32)
    player0_reach[0] = np.float32(root_player0_reach)
    player1_reach[0] = np.float32(root_player1_reach)

    for node_index in range(tree.node_count):
        node_type = tree.node_types[node_index]
        if node_type in {NodeType.LEAF, NodeType.TERMINAL}:
            continue

        links = tree.child_links(NodeId(node_index))
        if node_type is NodeType.CHANCE:
            for link in links:
                assert link.chance_prob is not None
                child_index = int(link.child)
                chance_prob = np.float32(link.chance_prob)
                player0_reach[child_index] += player0_reach[node_index] * chance_prob
                player1_reach[child_index] += player1_reach[node_index] * chance_prob
            continue

        infoset_id = tree.infoset_ids[node_index]
        assert infoset_id is not None
        strategy = store.current_strategy(int(infoset_id))
        for action_index, link in enumerate(links):
            child_index = int(link.child)
            action_prob = strategy[action_index]
            if node_type is NodeType.PLAYER0:
                player0_reach[child_index] += player0_reach[node_index] * action_prob
                player1_reach[child_index] += player1_reach[node_index]
            else:
                player0_reach[child_index] += player0_reach[node_index]
                player1_reach[child_index] += player1_reach[node_index] * action_prob

    return ForwardPassResult(
        player0_reach=player0_reach,
        player1_reach=player1_reach,
    )


def compute_reach_probabilities_parallel(
    tree: PublicTree,
    store: InfosetStore,
    *,
    root_player0_reach: float = 1.0,
    root_player1_reach: float = 1.0,
    max_workers: int = 1,
) -> ForwardPassResult:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if max_workers == 1 or tree.node_count <= 1:
        return compute_reach_probabilities(
            tree,
            store,
            root_player0_reach=root_player0_reach,
            root_player1_reach=root_player1_reach,
        )

    levels = build_tree_levels(tree)
    player0_reach = np.zeros(tree.node_count, dtype=np.float32)
    player1_reach = np.zeros(tree.node_count, dtype=np.float32)
    player0_reach[0] = np.float32(root_player0_reach)
    player1_reach[0] = np.float32(root_player1_reach)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for level in levels.forward_levels:
            results = tuple(
                executor.map(
                    lambda node_index: _forward_node_update(
                        tree,
                        store,
                        node_index,
                        np.float32(player0_reach[node_index]),
                        np.float32(player1_reach[node_index]),
                    ),
                    level,
                )
            )
            for updates_p0, updates_p1 in results:
                for child_index, reach_value in updates_p0:
                    player0_reach[child_index] += reach_value
                for child_index, reach_value in updates_p1:
                    player1_reach[child_index] += reach_value

    return ForwardPassResult(
        player0_reach=player0_reach,
        player1_reach=player1_reach,
    )


def compute_counterfactual_values(
    tree: PublicTree,
    store: InfosetStore,
    *,
    leaf_values_player0: NDArray[np.float32] | None = None,
    leaf_values_player1: NDArray[np.float32] | None = None,
    terminal_values_player0: NDArray[np.float32] | None = None,
) -> BackwardPassResult:
    node_values_player0 = np.zeros(tree.node_count, dtype=np.float32)
    node_values_player1 = np.zeros(tree.node_count, dtype=np.float32)
    infoset_action_values: dict[int, NDArray[np.float32]] = {}

    terminal_p0 = (
        np.zeros(tree.node_count, dtype=np.float32)
        if terminal_values_player0 is None
        else np.asarray(terminal_values_player0, dtype=np.float32)
    )
    leaf_p0 = (
        np.zeros(tree.node_count, dtype=np.float32)
        if leaf_values_player0 is None
        else np.asarray(leaf_values_player0, dtype=np.float32)
    )
    leaf_p1 = (
        -leaf_p0
        if leaf_values_player1 is None
        else np.asarray(leaf_values_player1, dtype=np.float32)
    )

    for node_index in range(tree.node_count - 1, -1, -1):
        node_type = tree.node_types[node_index]
        if node_type is NodeType.TERMINAL:
            node_values_player0[node_index] = terminal_p0[node_index]
            node_values_player1[node_index] = -terminal_p0[node_index]
            continue
        if node_type is NodeType.LEAF:
            node_values_player0[node_index] = leaf_p0[node_index]
            node_values_player1[node_index] = leaf_p1[node_index]
            continue

        links = tree.child_links(NodeId(node_index))
        child_values_p0 = np.array(
            [node_values_player0[int(link.child)] for link in links],
            dtype=np.float32,
        )
        child_values_p1 = np.array(
            [node_values_player1[int(link.child)] for link in links],
            dtype=np.float32,
        )

        if node_type is NodeType.CHANCE:
            chance_probs: list[float] = []
            for link in links:
                assert link.chance_prob is not None
                chance_probs.append(link.chance_prob)
            probs = np.array(
                chance_probs,
                dtype=np.float32,
            )
            node_values_player0[node_index] = np.float32(
                np.sum(probs * child_values_p0, dtype=np.float64)
            )
            node_values_player1[node_index] = np.float32(
                np.sum(probs * child_values_p1, dtype=np.float64)
            )
            continue

        infoset_id = tree.infoset_ids[node_index]
        assert infoset_id is not None
        strategy = store.current_strategy(int(infoset_id))
        if node_type is NodeType.PLAYER0:
            infoset_action_values[int(infoset_id)] = child_values_p0
            node_values_player0[node_index] = np.float32(
                np.sum(strategy * child_values_p0, dtype=np.float64)
            )
            node_values_player1[node_index] = np.float32(
                np.sum(strategy * child_values_p1, dtype=np.float64)
            )
        else:
            infoset_action_values[int(infoset_id)] = child_values_p1
            node_values_player0[node_index] = np.float32(
                np.sum(strategy * child_values_p0, dtype=np.float64)
            )
            node_values_player1[node_index] = np.float32(
                np.sum(strategy * child_values_p1, dtype=np.float64)
            )

    return BackwardPassResult(
        node_values_player0=node_values_player0,
        node_values_player1=node_values_player1,
        infoset_action_values=infoset_action_values,
    )


def compute_counterfactual_values_parallel(
    tree: PublicTree,
    store: InfosetStore,
    *,
    leaf_values_player0: NDArray[np.float32] | None = None,
    leaf_values_player1: NDArray[np.float32] | None = None,
    terminal_values_player0: NDArray[np.float32] | None = None,
    max_workers: int = 1,
) -> BackwardPassResult:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if max_workers == 1 or tree.node_count <= 1:
        return compute_counterfactual_values(
            tree,
            store,
            leaf_values_player0=leaf_values_player0,
            leaf_values_player1=leaf_values_player1,
            terminal_values_player0=terminal_values_player0,
        )

    levels = build_tree_levels(tree)
    node_values_player0 = np.zeros(tree.node_count, dtype=np.float32)
    node_values_player1 = np.zeros(tree.node_count, dtype=np.float32)
    infoset_action_values: dict[int, NDArray[np.float32]] = {}

    terminal_p0 = (
        np.zeros(tree.node_count, dtype=np.float32)
        if terminal_values_player0 is None
        else np.asarray(terminal_values_player0, dtype=np.float32)
    )
    leaf_p0 = (
        np.zeros(tree.node_count, dtype=np.float32)
        if leaf_values_player0 is None
        else np.asarray(leaf_values_player0, dtype=np.float32)
    )
    leaf_p1 = (
        -leaf_p0
        if leaf_values_player1 is None
        else np.asarray(leaf_values_player1, dtype=np.float32)
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for level in levels.backward_levels:
            results = tuple(
                executor.map(
                    lambda node_index: _backward_node_update(
                        tree,
                        store,
                        node_index,
                        node_values_player0,
                        node_values_player1,
                        terminal_p0,
                        leaf_p0,
                        leaf_p1,
                    ),
                    level,
                )
            )
            for (
                node_index,
                value_p0,
                value_p1,
                infoset_index,
                action_values,
            ) in results:
                node_values_player0[node_index] = value_p0
                node_values_player1[node_index] = value_p1
                if infoset_index is not None and action_values is not None:
                    infoset_action_values[infoset_index] = action_values

    return BackwardPassResult(
        node_values_player0=node_values_player0,
        node_values_player1=node_values_player1,
        infoset_action_values=infoset_action_values,
    )


def update_regrets_from_traversal(
    tree: PublicTree,
    store: InfosetStore,
    backward_pass: BackwardPassResult,
    *,
    active_player: int,
    strategy_weight: float = 1.0,
) -> RegretUpdateResult:
    if active_player not in {0, 1}:
        raise ValueError("active_player must be 0 or 1")

    infoset_values = np.zeros(store.layout.infoset_count, dtype=np.float32)
    updated_infosets: list[int] = []
    for node_index, infoset_id in enumerate(tree.infoset_ids):
        if infoset_id is None:
            continue
        infoset_index = int(infoset_id)
        node_type = tree.node_types[node_index]
        if active_player == 0 and node_type is not NodeType.PLAYER0:
            continue
        if active_player == 1 and node_type is not NodeType.PLAYER1:
            continue

        action_values = backward_pass.infoset_action_values[infoset_index]
        strategy = store.current_strategy(infoset_index)
        infoset_value = np.float32(np.sum(strategy * action_values, dtype=np.float64))
        store.regrets_for_infoset(infoset_index)[:] += action_values - infoset_value
        store.strategy_sums_for_infoset(infoset_index)[:] += strategy * np.float32(
            strategy_weight
        )
        infoset_values[infoset_index] = infoset_value
        updated_infosets.append(infoset_index)

    return RegretUpdateResult(
        infoset_values=infoset_values,
        updated_infosets=tuple(updated_infosets),
    )


def update_regrets_from_traversal_parallel(
    tree: PublicTree,
    store: InfosetStore,
    backward_pass: BackwardPassResult,
    *,
    active_player: int,
    strategy_weight: float = 1.0,
    max_workers: int = 1,
) -> RegretUpdateResult:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")

    infoset_indices = _active_infoset_indices(tree, active_player)
    if max_workers == 1 or len(infoset_indices) <= 1:
        return update_regrets_from_traversal(
            tree,
            store,
            backward_pass,
            active_player=active_player,
            strategy_weight=strategy_weight,
        )

    chunk_size = max(1, (len(infoset_indices) + max_workers - 1) // max_workers)
    chunks = tuple(
        infoset_indices[start : start + chunk_size]
        for start in range(0, len(infoset_indices), chunk_size)
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        chunk_results = tuple(
            executor.map(
                lambda chunk: _compute_chunk_updates(
                    store,
                    backward_pass,
                    chunk,
                    strategy_weight,
                ),
                chunks,
            )
        )

    infoset_values = np.zeros(store.layout.infoset_count, dtype=np.float32)
    updated_infosets: list[int] = []
    for chunk_infoset_values, regret_updates, strategy_updates, chunk_infosets in (
        chunk_results
    ):
        infoset_values += chunk_infoset_values
        for infoset_index in chunk_infosets:
            store.regrets_for_infoset(infoset_index)[:] += regret_updates[infoset_index]
            store.strategy_sums_for_infoset(infoset_index)[:] += strategy_updates[
                infoset_index
            ]
            updated_infosets.append(infoset_index)

    return RegretUpdateResult(
        infoset_values=infoset_values,
        updated_infosets=tuple(updated_infosets),
    )


def _active_infoset_indices(tree: PublicTree, active_player: int) -> tuple[int, ...]:
    if active_player not in {0, 1}:
        raise ValueError("active_player must be 0 or 1")

    infoset_indices: list[int] = []
    for node_index, infoset_id in enumerate(tree.infoset_ids):
        if infoset_id is None:
            continue
        node_type = tree.node_types[node_index]
        if active_player == 0 and node_type is not NodeType.PLAYER0:
            continue
        if active_player == 1 and node_type is not NodeType.PLAYER1:
            continue
        infoset_indices.append(int(infoset_id))
    return tuple(infoset_indices)


def _compute_chunk_updates(
    store: InfosetStore,
    backward_pass: BackwardPassResult,
    infoset_indices: tuple[int, ...],
    strategy_weight: float,
) -> tuple[
    NDArray[np.float32],
    dict[int, NDArray[np.float32]],
    dict[int, NDArray[np.float32]],
    tuple[int, ...],
]:
    infoset_values = np.zeros(store.layout.infoset_count, dtype=np.float32)
    regret_updates: dict[int, NDArray[np.float32]] = {}
    strategy_updates: dict[int, NDArray[np.float32]] = {}

    for infoset_index in infoset_indices:
        action_values = backward_pass.infoset_action_values[infoset_index]
        strategy = store.current_strategy(infoset_index)
        infoset_value = np.float32(np.sum(strategy * action_values, dtype=np.float64))
        regret_updates[infoset_index] = action_values - infoset_value
        strategy_updates[infoset_index] = strategy * np.float32(strategy_weight)
        infoset_values[infoset_index] = infoset_value

    return infoset_values, regret_updates, strategy_updates, infoset_indices


def build_tree_levels(tree: PublicTree) -> TreeLevels:
    depth_by_node = [-1] * tree.node_count
    depth_by_node[0] = 0
    queue = [0]
    while queue:
        node_index = queue.pop(0)
        for link in tree.child_links(NodeId(node_index)):
            child_index = int(link.child)
            if depth_by_node[child_index] != -1:
                continue
            depth_by_node[child_index] = depth_by_node[node_index] + 1
            queue.append(child_index)

    max_depth = max(depth_by_node)
    forward_levels = tuple(
        tuple(
            node_index
            for node_index, depth in enumerate(depth_by_node)
            if depth == current_depth
        )
        for current_depth in range(max_depth + 1)
    )
    backward_levels = tuple(reversed(forward_levels))
    return TreeLevels(
        forward_levels=forward_levels,
        backward_levels=backward_levels,
    )


def _forward_node_update(
    tree: PublicTree,
    store: InfosetStore,
    node_index: int,
    node_player0_reach: np.float32,
    node_player1_reach: np.float32,
) -> tuple[tuple[tuple[int, np.float32], ...], tuple[tuple[int, np.float32], ...]]:
    node_type = tree.node_types[node_index]
    if node_type in {NodeType.LEAF, NodeType.TERMINAL}:
        return (), ()

    updates_p0: list[tuple[int, np.float32]] = []
    updates_p1: list[tuple[int, np.float32]] = []
    links = tree.child_links(NodeId(node_index))
    if node_type is NodeType.CHANCE:
        for link in links:
            assert link.chance_prob is not None
            child_index = int(link.child)
            chance_prob = np.float32(link.chance_prob)
            updates_p0.append((child_index, node_player0_reach * chance_prob))
            updates_p1.append((child_index, node_player1_reach * chance_prob))
        return tuple(updates_p0), tuple(updates_p1)

    infoset_id = tree.infoset_ids[node_index]
    assert infoset_id is not None
    strategy = store.current_strategy(int(infoset_id))
    for action_index, link in enumerate(links):
        child_index = int(link.child)
        action_prob = strategy[action_index]
        if node_type is NodeType.PLAYER0:
            updates_p0.append((child_index, node_player0_reach * action_prob))
            updates_p1.append((child_index, node_player1_reach))
        else:
            updates_p0.append((child_index, node_player0_reach))
            updates_p1.append((child_index, node_player1_reach * action_prob))
    return tuple(updates_p0), tuple(updates_p1)


def _backward_node_update(
    tree: PublicTree,
    store: InfosetStore,
    node_index: int,
    node_values_player0: NDArray[np.float32],
    node_values_player1: NDArray[np.float32],
    terminal_p0: NDArray[np.float32],
    leaf_p0: NDArray[np.float32],
    leaf_p1: NDArray[np.float32],
) -> tuple[int, np.float32, np.float32, int | None, NDArray[np.float32] | None]:
    node_type = tree.node_types[node_index]
    if node_type is NodeType.TERMINAL:
        value_p0 = terminal_p0[node_index]
        return node_index, value_p0, np.float32(-value_p0), None, None
    if node_type is NodeType.LEAF:
        return node_index, leaf_p0[node_index], leaf_p1[node_index], None, None

    links = tree.child_links(NodeId(node_index))
    child_values_p0 = np.array(
        [node_values_player0[int(link.child)] for link in links],
        dtype=np.float32,
    )
    child_values_p1 = np.array(
        [node_values_player1[int(link.child)] for link in links],
        dtype=np.float32,
    )

    if node_type is NodeType.CHANCE:
        chance_probs = np.array(
            [float(link.chance_prob) for link in links if link.chance_prob is not None],
            dtype=np.float32,
        )
        value_p0 = np.float32(np.sum(chance_probs * child_values_p0, dtype=np.float64))
        value_p1 = np.float32(np.sum(chance_probs * child_values_p1, dtype=np.float64))
        return node_index, value_p0, value_p1, None, None

    infoset_id = tree.infoset_ids[node_index]
    assert infoset_id is not None
    infoset_index = int(infoset_id)
    strategy = store.current_strategy(infoset_index)
    value_p0 = np.float32(np.sum(strategy * child_values_p0, dtype=np.float64))
    value_p1 = np.float32(np.sum(strategy * child_values_p1, dtype=np.float64))
    action_values = (
        child_values_p0 if node_type is NodeType.PLAYER0 else child_values_p1
    )
    return node_index, value_p0, value_p1, infoset_index, action_values

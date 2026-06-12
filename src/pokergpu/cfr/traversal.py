from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field

import numpy as np
from numpy.typing import NDArray

from pokergpu.core.payouts import total_pot
from pokergpu.core.state import GameState
from pokergpu.eval import LeafEvaluator, LeafFeatureBatch, LeafValueBatch
from pokergpu.tree import NodeId, NodeType, PublicTree

from .infosets import InfosetStore


def _is_player_node(node_type: NodeType) -> bool:
    return node_type in {NodeType.PLAYER0, NodeType.PLAYER1, NodeType.PLAYER2}


def _player_index_for_node_type(node_type: NodeType) -> int | None:
    if node_type is NodeType.PLAYER0:
        return 0
    if node_type is NodeType.PLAYER1:
        return 1
    if node_type is NodeType.PLAYER2:
        return 2
    return None


@dataclass(frozen=True, slots=True)
class ForwardPassResult:
    player0_reach: NDArray[np.float32]
    player1_reach: NDArray[np.float32]
    player2_reach: NDArray[np.float32] | None = None


@dataclass(frozen=True, slots=True)
class BackwardPassResult:
    node_values_player0: NDArray[np.float32]
    node_values_player1: NDArray[np.float32]
    node_values_player2: NDArray[np.float32] | None = None
    infoset_action_values: dict[int, NDArray[np.float32]] = field(default_factory=dict)

    @property
    def node_values(self) -> NDArray[np.float32]:
        return np.stack(
            (
                self.node_values_player0,
                self.node_values_player1,
                self.node_values_player2
                if self.node_values_player2 is not None
                else np.zeros_like(self.node_values_player0),
            ),
            axis=0,
        )


@dataclass(frozen=True, slots=True)
class RegretUpdateResult:
    infoset_values: NDArray[np.float32]
    updated_infosets: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TreeLevels:
    forward_levels: tuple[tuple[int, ...], ...]
    backward_levels: tuple[tuple[int, ...], ...]


def build_leaf_feature_batch(
    tree: PublicTree,
    node_indices: tuple[int, ...],
    *,
    node_states: tuple[GameState, ...] | None = None,
    player_to_act: int | None = None,
    reach_p0: NDArray[np.float32] | None = None,
    reach_p1: NDArray[np.float32] | None = None,
    reach_p2: NDArray[np.float32] | None = None,
) -> LeafFeatureBatch:
    node_indices = tuple(node_indices)
    size = len(node_indices)
    player_to_act_arr = np.zeros(size, dtype=np.int32)
    street = np.zeros(size, dtype=np.int32)
    pot = np.zeros(size, dtype=np.float32)
    stack_p0 = np.zeros(size, dtype=np.float32)
    stack_p1 = np.zeros(size, dtype=np.float32)
    board_size = np.zeros(size, dtype=np.int32)
    reach_p0_arr = np.zeros(size, dtype=np.float32)
    reach_p1_arr = np.zeros(size, dtype=np.float32)
    reach_p2_arr = np.zeros(size, dtype=np.float32)
    is_terminal = np.zeros(size, dtype=np.bool_)
    is_frontier = np.zeros(size, dtype=np.bool_)
    infoset_id = np.full(size, -1, dtype=np.int32)

    for batch_index, node_index in enumerate(node_indices):
        node_type = tree.node_types[node_index]
        if node_states is not None:
            node_state = node_states[node_index]
            player_to_act_arr[batch_index] = int(node_state.betting_round.to_act)
            street[batch_index] = _street_to_index(node_state.current_street)
            pot[batch_index] = np.float32(total_pot(node_state))
            stack_p0[batch_index] = np.float32(_stack_for_player(node_state, 0))
            stack_p1[batch_index] = np.float32(_stack_for_player(node_state, 1))
            board_size[batch_index] = len(node_state.board.cards)
        else:
            player_to_act_arr[batch_index] = (
                int(player_to_act) if player_to_act is not None else 0
            )
            street[batch_index] = 0
            board_size[batch_index] = 0
        is_terminal[batch_index] = node_type is NodeType.TERMINAL
        is_frontier[batch_index] = tree.is_frontier[node_index]
        infoset = tree.infoset_ids[node_index]
        if infoset is not None:
            infoset_id[batch_index] = int(infoset)
        if reach_p0 is not None:
            reach_p0_arr[batch_index] = np.float32(reach_p0[node_index])
        if reach_p1 is not None:
            reach_p1_arr[batch_index] = np.float32(reach_p1[node_index])
        if reach_p2 is not None:
            reach_p2_arr[batch_index] = np.float32(reach_p2[node_index])

    return LeafFeatureBatch(
        node_indices=node_indices,
        player_to_act=player_to_act_arr,
        street=street,
        pot=pot,
        stack_p0=stack_p0,
        stack_p1=stack_p1,
        board_size=board_size,
        reach_p0=reach_p0_arr,
        reach_p1=reach_p1_arr,
        reach_p2=reach_p2_arr,
        is_terminal=is_terminal,
        is_frontier=is_frontier,
        infoset_id=infoset_id,
    )


def scatter_leaf_values(
    node_indices: tuple[int, ...],
    values: LeafValueBatch,
    node_values_player0: NDArray[np.float32],
    node_values_player1: NDArray[np.float32],
    node_values_player2: NDArray[np.float32] | None = None,
) -> None:
    for batch_index, node_index in enumerate(node_indices):
        node_values_player0[node_index] = values.ev_player0[batch_index]
        node_values_player1[node_index] = values.ev_player1[batch_index]
        if node_values_player2 is not None:
            if values.ev_player2 is not None:
                node_values_player2[node_index] = values.ev_player2[batch_index]
            else:
                node_values_player2[node_index] = -(
                    values.ev_player0[batch_index] + values.ev_player1[batch_index]
                )


def evaluate_frontier_nodes(
    tree: PublicTree,
    evaluator: LeafEvaluator,
    *,
    node_states: tuple[GameState, ...] | None = None,
    reach_p0: NDArray[np.float32] | None = None,
    reach_p1: NDArray[np.float32] | None = None,
) -> tuple[tuple[int, ...], LeafValueBatch]:
    frontier_nodes = tuple(
        node_index
        for node_index in range(tree.node_count)
        if tree.is_frontier[node_index]
        and tree.node_types[node_index] is not NodeType.TERMINAL
    )
    batch = build_leaf_feature_batch(
        tree,
        frontier_nodes,
        node_states=node_states,
        reach_p0=reach_p0,
        reach_p1=reach_p1,
    )
    return frontier_nodes, evaluator.evaluate(batch)


def compute_reach_probabilities(
    tree: PublicTree,
    store: InfosetStore,
    *,
    root_player0_reach: float = 1.0,
    root_player1_reach: float = 1.0,
    root_player2_reach: float = 1.0,
    min_reach_prob: float = 0.0,
) -> ForwardPassResult:
    player0_reach = np.zeros(tree.node_count, dtype=np.float32)
    player1_reach = np.zeros(tree.node_count, dtype=np.float32)
    player2_reach = np.zeros(tree.node_count, dtype=np.float32)
    player0_reach[0] = np.float32(root_player0_reach)
    player1_reach[0] = np.float32(root_player1_reach)
    player2_reach[0] = np.float32(root_player2_reach)

    for node_index in range(tree.node_count):
        node_type = tree.node_types[node_index]
        if (node_type in {NodeType.LEAF, NodeType.TERMINAL} 
            or tree.is_frontier[node_index]):
            continue
        if min_reach_prob > 0.0:
            reach_sum = float(player0_reach[node_index] + player1_reach[node_index])
            if reach_sum < min_reach_prob:
                continue

        links = tree.child_links(NodeId(node_index))
        if node_type is NodeType.CHANCE:
            for link in links:
                assert link.chance_prob is not None
                child_index = int(link.child)
                chance_prob = np.float32(link.chance_prob)
                player0_reach[child_index] += player0_reach[node_index] * chance_prob
                player1_reach[child_index] += player1_reach[node_index] * chance_prob
                player2_reach[child_index] += player2_reach[node_index] * chance_prob
            continue

        infoset_id = tree.infoset_ids[node_index]
        assert infoset_id is not None
        strategy = store.current_strategy(int(infoset_id))
        if strategy.shape[0] != len(links):
            strategy = strategy[: len(links)]
        for action_index, link in enumerate(links):
            child_index = int(link.child)
            action_prob = strategy[action_index]
            if node_type is NodeType.PLAYER0:
                player0_reach[child_index] += player0_reach[node_index] * action_prob
                player1_reach[child_index] += player1_reach[node_index]
                player2_reach[child_index] += player2_reach[node_index]
            elif node_type is NodeType.PLAYER1:
                player0_reach[child_index] += player0_reach[node_index]
                player1_reach[child_index] += player1_reach[node_index] * action_prob
                player2_reach[child_index] += player2_reach[node_index]
            elif node_type is NodeType.PLAYER2:
                player0_reach[child_index] += player0_reach[node_index]
                player1_reach[child_index] += player1_reach[node_index]
                player2_reach[child_index] += player2_reach[node_index] * action_prob
            else:
                player0_reach[child_index] += player0_reach[node_index]
                player1_reach[child_index] += player1_reach[node_index]
                player2_reach[child_index] += player2_reach[node_index]

    return ForwardPassResult(
        player0_reach=player0_reach,
        player1_reach=player1_reach,
        player2_reach=player2_reach,
    )


def compute_reach_probabilities_parallel(
    tree: PublicTree,
    store: InfosetStore,
    *,
    root_player0_reach: float = 1.0,
    root_player1_reach: float = 1.0,
    root_player2_reach: float = 1.0,
    min_reach_prob: float = 0.0,
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
            root_player2_reach=root_player2_reach,
            min_reach_prob=min_reach_prob,
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
                        np.float32(min_reach_prob),
                    ),
                    level,
                )
            )
            _reduce_child_reach_updates(
                player0_reach,
                tuple(result[0] for result in results),
            )
            _reduce_child_reach_updates(
                player1_reach,
                tuple(result[1] for result in results),
            )

    return ForwardPassResult(
        player0_reach=player0_reach,
        player1_reach=player1_reach,
    )


def compute_counterfactual_values(
    tree: PublicTree,
    store: InfosetStore,
    *,
    node_states: tuple[GameState, ...] | None = None,
    reach_p0: NDArray[np.float32] | None = None,
    reach_p1: NDArray[np.float32] | None = None,
    leaf_values_player0: NDArray[np.float32] | None = None,
    leaf_values_player1: NDArray[np.float32] | None = None,
    terminal_values_player0: NDArray[np.float32] | None = None,
    evaluator: LeafEvaluator | None = None,
) -> BackwardPassResult:
    node_values_player0 = np.zeros(tree.node_count, dtype=np.float32)
    node_values_player1 = np.zeros(tree.node_count, dtype=np.float32)
    node_values_player2 = np.zeros(tree.node_count, dtype=np.float32)
    infoset_action_values: dict[int, NDArray[np.float32]] = {}

    terminal_p0 = (
        np.zeros(tree.node_count, dtype=np.float32)
        if terminal_values_player0 is None
        else np.asarray(terminal_values_player0, dtype=np.float32)
    )
    leaf_p0 = np.zeros(tree.node_count, dtype=np.float32)
    leaf_p1 = np.zeros(tree.node_count, dtype=np.float32)
    leaf_p2 = np.zeros(tree.node_count, dtype=np.float32)
    if leaf_values_player0 is not None:
        leaf_p0[:] = np.asarray(leaf_values_player0, dtype=np.float32)
    if leaf_values_player1 is not None:
        leaf_p1[:] = np.asarray(leaf_values_player1, dtype=np.float32)
    leaf_p2[:] = -(leaf_p0 + leaf_p1)
    if leaf_values_player1 is None:
        leaf_p1[:] = -leaf_p0

    if evaluator is not None:
        frontier_nodes, leaf_values = evaluate_frontier_nodes(
            tree,
            evaluator,
            node_states=node_states,
            reach_p0=reach_p0,
            reach_p1=reach_p1,
        )
        if frontier_nodes:
            scatter_leaf_values(
                frontier_nodes,
                leaf_values,
                leaf_p0,
                leaf_p1,
                node_values_player2,
            )

    for node_index in range(tree.node_count - 1, -1, -1):
        node_type = tree.node_types[node_index]
        if node_type is NodeType.TERMINAL:
            node_values_player0[node_index] = terminal_p0[node_index]
            node_values_player1[node_index] = -terminal_p0[node_index]
            node_values_player2[node_index] = 0.0
            continue
        if node_type is NodeType.LEAF or tree.is_frontier[node_index]:
            node_values_player0[node_index] = leaf_p0[node_index]
            node_values_player1[node_index] = leaf_p1[node_index]
            node_values_player2[node_index] = leaf_p2[node_index]
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
        child_values_p2 = np.array(
            [node_values_player2[int(link.child)] for link in links],
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
            node_values_player2[node_index] = np.float32(
                np.sum(probs * child_values_p2, dtype=np.float64)
            )
            continue

        infoset_id = tree.infoset_ids[node_index]
        assert infoset_id is not None
        strategy = store.current_strategy(int(infoset_id))
        if strategy.shape[0] != len(links):
            strategy = strategy[: len(links)]
        if node_type is NodeType.PLAYER0:
            infoset_action_values[int(infoset_id)] = child_values_p0[: strategy.shape[0]]
            node_values_player0[node_index] = np.float32(
                np.sum(strategy * child_values_p0[: strategy.shape[0]], dtype=np.float64)
            )
            node_values_player1[node_index] = np.float32(
                np.sum(strategy * child_values_p1[: strategy.shape[0]], dtype=np.float64)
            )
            node_values_player2[node_index] = np.float32(
                np.sum(strategy * child_values_p2[: strategy.shape[0]], dtype=np.float64)
            )
        elif node_type is NodeType.PLAYER1:
            infoset_action_values[int(infoset_id)] = child_values_p1[: strategy.shape[0]]
            node_values_player0[node_index] = np.float32(
                np.sum(strategy * child_values_p0[: strategy.shape[0]], dtype=np.float64)
            )
            node_values_player1[node_index] = np.float32(
                np.sum(strategy * child_values_p1[: strategy.shape[0]], dtype=np.float64)
            )
            node_values_player2[node_index] = np.float32(
                np.sum(strategy * child_values_p2[: strategy.shape[0]], dtype=np.float64)
            )
        else:
            infoset_action_values[int(infoset_id)] = child_values_p2[: strategy.shape[0]]
            node_values_player0[node_index] = np.float32(
                np.sum(strategy * child_values_p0[: strategy.shape[0]], dtype=np.float64)
            )
            node_values_player1[node_index] = np.float32(
                np.sum(strategy * child_values_p1[: strategy.shape[0]], dtype=np.float64)
            )
            node_values_player2[node_index] = np.float32(
                np.sum(strategy * child_values_p2[: strategy.shape[0]], dtype=np.float64)
            )

    return BackwardPassResult(
        node_values_player0=node_values_player0,
        node_values_player1=node_values_player1,
        node_values_player2=node_values_player2,
        infoset_action_values=infoset_action_values,
    )


def compute_counterfactual_values_parallel(
    tree: PublicTree,
    store: InfosetStore,
    *,
    node_states: tuple[GameState, ...] | None = None,
    reach_p0: NDArray[np.float32] | None = None,
    reach_p1: NDArray[np.float32] | None = None,
    leaf_values_player0: NDArray[np.float32] | None = None,
    leaf_values_player1: NDArray[np.float32] | None = None,
    terminal_values_player0: NDArray[np.float32] | None = None,
    evaluator: LeafEvaluator | None = None,
    max_workers: int = 1,
) -> BackwardPassResult:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if max_workers == 1 or tree.node_count <= 1:
        return compute_counterfactual_values(
            tree,
            store,
            node_states=node_states,
            reach_p0=reach_p0,
            reach_p1=reach_p1,
            leaf_values_player0=leaf_values_player0,
            leaf_values_player1=leaf_values_player1,
            terminal_values_player0=terminal_values_player0,
            evaluator=evaluator,
        )

    levels = build_tree_levels(tree)
    node_values_player0 = np.zeros(tree.node_count, dtype=np.float32)
    node_values_player1 = np.zeros(tree.node_count, dtype=np.float32)
    node_values_player2 = np.zeros(tree.node_count, dtype=np.float32)
    infoset_action_values: dict[int, NDArray[np.float32]] = {}

    terminal_p0 = (
        np.zeros(tree.node_count, dtype=np.float32)
        if terminal_values_player0 is None
        else np.asarray(terminal_values_player0, dtype=np.float32)
    )
    leaf_p0 = np.zeros(tree.node_count, dtype=np.float32)
    leaf_p1 = np.zeros(tree.node_count, dtype=np.float32)
    if leaf_values_player0 is not None:
        leaf_p0[:] = np.asarray(leaf_values_player0, dtype=np.float32)
    if leaf_values_player1 is not None:
        leaf_p1[:] = np.asarray(leaf_values_player1, dtype=np.float32)
    if leaf_values_player1 is None:
        leaf_p1[:] = -leaf_p0

    if evaluator is not None:
        frontier_nodes, leaf_values = evaluate_frontier_nodes(
            tree,
            evaluator,
            node_states=node_states,
            reach_p0=reach_p0,
            reach_p1=reach_p1,
        )
        if frontier_nodes:
            scatter_leaf_values(
                frontier_nodes,
                leaf_values,
                leaf_p0,
                leaf_p1,
                node_values_player2,
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
                        node_values_player2,
                        terminal_p0,
                        leaf_p0,
                        leaf_p1,
                    ),
                    level,
                )
            )
            _reduce_backward_level_results(
                level_results=results,
                node_values_player0=node_values_player0,
                node_values_player1=node_values_player1,
                infoset_action_values=infoset_action_values,
            )

    return BackwardPassResult(
        node_values_player0=node_values_player0,
        node_values_player1=node_values_player1,
        node_values_player2=node_values_player2,
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
    if active_player not in {0, 1, 2}:
        raise ValueError("active_player must be 0, 1, or 2")

    infoset_values = np.zeros(store.layout.infoset_count, dtype=np.float32)
    updated_infosets: list[int] = []
    for node_index, infoset_id in enumerate(tree.infoset_ids):
        if infoset_id is None:
            continue
        infoset_index = int(infoset_id)
        node_type = tree.node_types[node_index]
        if _player_index_for_node_type(node_type) != active_player:
            continue

        action_values = backward_pass.infoset_action_values[infoset_index]
        if action_values.size == 0:
            continue
        strategy = store.current_strategy(infoset_index)
        limit = min(strategy.shape[0], action_values.shape[0])
        if limit == 0:
            continue
        strategy = strategy[:limit]
        action_values = action_values[:limit]
        infoset_value = np.float32(np.sum(strategy * action_values, dtype=np.float64))
        store.regrets_for_infoset(infoset_index)[:limit] += action_values - infoset_value
        store.strategy_sums_for_infoset(infoset_index)[:limit] += strategy * np.float32(
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
    infoset_values = reduce_float32_arrays(
        tuple(chunk_infoset_values for chunk_infoset_values, _, _, _ in chunk_results),
        store.layout.infoset_count,
    )
    merged_regret_updates = reduce_infoset_vector_maps(
        tuple(regret_updates for _, regret_updates, _, _ in chunk_results)
    )
    merged_strategy_updates = reduce_infoset_vector_maps(
        tuple(strategy_updates for _, _, strategy_updates, _ in chunk_results)
    )
    ordered_infosets = tuple(
        infoset_index
        for _, _, _, chunk_infosets in chunk_results
        for infoset_index in chunk_infosets
    )
    for infoset_index in ordered_infosets:
        store.regrets_for_infoset(infoset_index)[:] += merged_regret_updates[
            infoset_index
        ]
        store.strategy_sums_for_infoset(infoset_index)[:] += merged_strategy_updates[
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
        if active_player == 2 and node_type is not NodeType.PLAYER2:
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
        limit = min(strategy.shape[0], action_values.shape[0])
        if limit == 0:
            continue
        strategy = strategy[:limit]
        action_values = action_values[:limit]
        infoset_value = np.float32(np.sum(strategy * action_values, dtype=np.float64))
        regret_updates[infoset_index] = action_values - infoset_value
        strategy_updates[infoset_index] = strategy * np.float32(strategy_weight)
        infoset_values[infoset_index] = infoset_value

    return infoset_values, regret_updates, strategy_updates, infoset_indices


def _reduce_child_reach_updates(
    target: NDArray[np.float32],
    worker_updates: tuple[tuple[tuple[int, np.float32], ...], ...],
) -> None:
    for child_index, reach_value in _flatten_and_sort_updates(worker_updates):
        target[child_index] += reach_value


def _reduce_backward_level_results(
    level_results: tuple[
        tuple[int, np.float32, np.float32, int | None, NDArray[np.float32] | None],
        ...,
    ],
    node_values_player0: NDArray[np.float32],
    node_values_player1: NDArray[np.float32],
    infoset_action_values: dict[int, NDArray[np.float32]],
) -> None:
    for node_index, value_p0, value_p1, infoset_index, action_values in sorted(
        level_results,
        key=lambda item: item[0],
    ):
        node_values_player0[node_index] = value_p0
        node_values_player1[node_index] = value_p1
        if infoset_index is not None and action_values is not None:
            infoset_action_values[infoset_index] = action_values


def reduce_float32_arrays(
    arrays: tuple[NDArray[np.float32], ...],
    size: int,
) -> NDArray[np.float32]:
    reduced = np.zeros(size, dtype=np.float32)
    for array in arrays:
        reduced += array
    return reduced


def reduce_infoset_vector_maps(
    partial_maps: tuple[dict[int, NDArray[np.float32]], ...],
) -> dict[int, NDArray[np.float32]]:
    reduced: dict[int, NDArray[np.float32]] = {}
    for partial_map in partial_maps:
        for infoset_index in sorted(partial_map):
            if infoset_index not in reduced:
                reduced[infoset_index] = np.array(
                    partial_map[infoset_index],
                    dtype=np.float32,
                    copy=True,
                )
            else:
                reduced[infoset_index] += partial_map[infoset_index]
    return reduced


def _flatten_and_sort_updates(
    worker_updates: tuple[tuple[tuple[int, np.float32], ...], ...],
) -> tuple[tuple[int, np.float32], ...]:
    return tuple(
        sorted(
            (update for updates in worker_updates for update in updates),
            key=lambda item: item[0],
        )
    )


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
    min_reach_prob: np.float32,
) -> tuple[tuple[tuple[int, np.float32], ...], tuple[tuple[int, np.float32], ...]]:
    node_type = tree.node_types[node_index]
    if node_type in {NodeType.LEAF, NodeType.TERMINAL}:
        return (), ()
    if tree.is_frontier[node_index]:
        return (), ()
    if min_reach_prob > 0.0 and float(node_player0_reach + node_player1_reach) < float(
        min_reach_prob
    ):
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
    node_values_player2: NDArray[np.float32],
    terminal_p0: NDArray[np.float32],
    leaf_p0: NDArray[np.float32],
    leaf_p1: NDArray[np.float32],
) -> tuple[int, np.float32, np.float32, int | None, NDArray[np.float32] | None]:
    node_type = tree.node_types[node_index]
    if node_type is NodeType.TERMINAL:
        value_p0 = terminal_p0[node_index]
        return node_index, value_p0, np.float32(-value_p0), None, None
    if node_type is NodeType.LEAF or tree.is_frontier[node_index]:
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


def _stack_for_player(state: GameState, player: int) -> int:
    for stack in state.betting_round.stacks:
        if int(stack.player) == player:
            return int(stack.stack)
    return 0


def _street_to_index(street: object) -> int:
    value = getattr(street, "value", "")
    return {
        "preflop": 0,
        "flop": 1,
        "turn": 2,
        "river": 3,
    }.get(value, 0)

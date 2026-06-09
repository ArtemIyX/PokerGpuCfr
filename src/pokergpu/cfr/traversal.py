from __future__ import annotations

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

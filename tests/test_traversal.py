import math

import numpy as np

from pokergpu.cfr import (
    InfosetLayout,
    InfosetStore,
    compute_counterfactual_values,
    compute_reach_probabilities,
    update_regrets_from_traversal,
)
from pokergpu.core.betting import Chips
from pokergpu.tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


def test_forward_pass_propagates_player_strategy_reach() -> None:
    tree = PublicTree(
        node_types=(NodeType.PLAYER0, NodeType.LEAF, NodeType.LEAF),
        first_child=(0, 2, 2),
        child_count=(2, 0, 0),
        children=(ChildLink(NodeId(1)), ChildLink(NodeId(2))),
        infoset_ids=(InfosetId(0), None, None),
        terminal_payoffs=(None, None, None),
    )
    store = InfosetStore.zeros(InfosetLayout.from_action_counts([2]))
    store.regrets[:] = np.array([3.0, 1.0], dtype=np.float32)

    forward = compute_reach_probabilities(tree, store)

    assert math.isclose(float(forward.player0_reach[1]), 0.75, abs_tol=1e-6)
    assert math.isclose(float(forward.player0_reach[2]), 0.25, abs_tol=1e-6)
    assert math.isclose(float(forward.player1_reach[1]), 1.0, abs_tol=1e-6)
    assert math.isclose(float(forward.player1_reach[2]), 1.0, abs_tol=1e-6)


def test_backward_pass_computes_expected_node_values() -> None:
    tree = PublicTree(
        node_types=(NodeType.PLAYER0, NodeType.LEAF, NodeType.LEAF),
        first_child=(0, 2, 2),
        child_count=(2, 0, 0),
        children=(ChildLink(NodeId(1)), ChildLink(NodeId(2))),
        infoset_ids=(InfosetId(0), None, None),
        terminal_payoffs=(None, None, None),
    )
    store = InfosetStore.zeros(InfosetLayout.from_action_counts([2]))
    store.regrets[:] = np.array([3.0, 1.0], dtype=np.float32)
    leaf_values = np.array([0.0, 2.0, -2.0], dtype=np.float32)

    backward = compute_counterfactual_values(
        tree,
        store,
        leaf_values_player0=leaf_values,
    )

    assert np.allclose(
        backward.infoset_action_values[0],
        np.array([2.0, -2.0], dtype=np.float32),
    )
    assert math.isclose(float(backward.node_values_player0[0]), 1.0, abs_tol=1e-6)
    assert math.isclose(float(backward.node_values_player1[0]), -1.0, abs_tol=1e-6)


def test_regret_update_is_separated_from_backward_pass() -> None:
    tree = PublicTree(
        node_types=(NodeType.PLAYER0, NodeType.LEAF, NodeType.LEAF),
        first_child=(0, 2, 2),
        child_count=(2, 0, 0),
        children=(ChildLink(NodeId(1)), ChildLink(NodeId(2))),
        infoset_ids=(InfosetId(0), None, None),
        terminal_payoffs=(None, None, None),
    )
    store = InfosetStore.zeros(InfosetLayout.from_action_counts([2]))
    backward = compute_counterfactual_values(
        tree,
        store,
        leaf_values_player0=np.array([0.0, 1.0, -1.0], dtype=np.float32),
    )

    update = update_regrets_from_traversal(
        tree,
        store,
        backward,
        active_player=0,
    )

    assert update.updated_infosets == (0,)
    assert np.allclose(store.regrets, np.array([1.0, -1.0], dtype=np.float32))
    assert np.allclose(store.strategy_sums, np.array([0.5, 0.5], dtype=np.float32))


def test_chance_nodes_are_handled_in_forward_and_backward_passes() -> None:
    tree = PublicTree(
        node_types=(NodeType.CHANCE, NodeType.TERMINAL, NodeType.TERMINAL),
        first_child=(0, 2, 2),
        child_count=(2, 0, 0),
        children=(
            ChildLink(NodeId(1), chance_prob=0.25),
            ChildLink(NodeId(2), chance_prob=0.75),
        ),
        infoset_ids=(None, None, None),
        terminal_payoffs=(None, Chips(0), Chips(0)),
    )
    store = InfosetStore.zeros(InfosetLayout.from_action_counts([1]))
    forward = compute_reach_probabilities(tree, store)
    backward = compute_counterfactual_values(
        tree,
        store,
        terminal_values_player0=np.array([0.0, 2.0, -2.0], dtype=np.float32),
    )

    assert math.isclose(float(forward.player0_reach[1]), 0.25, abs_tol=1e-6)
    assert math.isclose(float(forward.player0_reach[2]), 0.75, abs_tol=1e-6)
    assert math.isclose(float(backward.node_values_player0[0]), -1.0, abs_tol=1e-6)

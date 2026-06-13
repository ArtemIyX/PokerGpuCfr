import math

import numpy as np
import pytest

from pokergpu.cfr import (
    InfosetLayout,
    InfosetStore,
    compute_counterfactual_values,
    compute_reach_probabilities,
    update_regrets_from_traversal,
)
from pokergpu.cfr.traversal import BackwardPassResult
from pokergpu.core.betting import Chips
from pokergpu.tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


def _make_simple_tree() -> PublicTree:
    return PublicTree(
        node_types=(
            NodeType.PLAYER0,
            NodeType.TERMINAL,
            NodeType.TERMINAL,
        ),
        is_frontier=(False, False, False),
        first_child=(0, 0, 0),
        child_count=(2, 0, 0),
        children=(ChildLink(NodeId(1)), ChildLink(NodeId(2))),
        infoset_ids=(InfosetId(0), None, None),
        terminal_payoffs=(None, Chips(3), Chips(-1)),
    )


def test_forward_reach_propagates_player_strategy() -> None:
    tree = _make_simple_tree()
    store = InfosetStore.zeros(InfosetLayout.from_action_counts([2]))
    store.regrets[:] = np.array([1.0, 3.0], dtype=np.float32)

    forward = compute_reach_probabilities(tree, store)

    assert math.isclose(float(forward.player0_reach[1]), 0.25, abs_tol=1e-6)
    assert math.isclose(float(forward.player0_reach[2]), 0.75, abs_tol=1e-6)
    assert math.isclose(float(forward.player1_reach[1]), 1.0, abs_tol=1e-6)
    assert math.isclose(float(forward.player1_reach[2]), 1.0, abs_tol=1e-6)


def test_backward_values_use_terminal_payoffs() -> None:
    tree = _make_simple_tree()
    store = InfosetStore.zeros(InfosetLayout.from_action_counts([2]))
    store.regrets[:] = np.array([1.0, 3.0], dtype=np.float32)

    terminal_values = np.array([0.0, 3.0, -1.0], dtype=np.float32)
    backward = compute_counterfactual_values(
        tree,
        store,
        terminal_values_player0=terminal_values,
    )

    assert math.isclose(float(backward.node_values_player0[0]), 0.0, abs_tol=1e-6)
    assert math.isclose(float(backward.node_values_player0[1]), 3.0, abs_tol=1e-6)
    assert math.isclose(float(backward.node_values_player0[2]), -1.0, abs_tol=1e-6)
    assert math.isclose(float(backward.node_values_player1[1]), -3.0, abs_tol=1e-6)
    assert math.isclose(float(backward.node_values_player1[2]), 1.0, abs_tol=1e-6)
    assert np.array_equal(
        backward.infoset_action_values[0],
        np.array([3.0, -1.0], dtype=np.float32),
    )


def test_regret_update_writes_flat_store_in_place() -> None:
    tree = _make_simple_tree()
    store = InfosetStore.zeros(InfosetLayout.from_action_counts([2]))
    backward = BackwardPassResult(
        node_values_by_player=np.stack(
            (
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
            ),
            axis=0,
        ),
        node_values_player0=np.zeros(3, dtype=np.float32),
        node_values_player1=np.zeros(3, dtype=np.float32),
        infoset_action_values={0: np.array([3.0, -1.0], dtype=np.float32)},
    )

    result = update_regrets_from_traversal(
        tree,
        store,
        backward,
        active_player=0,
        strategy_weight=2.0,
    )

    assert np.allclose(store.regrets, np.array([2.0, -2.0], dtype=np.float32))
    assert np.allclose(store.strategy_sums, np.array([1.0, 1.0], dtype=np.float32))
    assert result.updated_infosets == (0,)
    assert math.isclose(float(result.infoset_values[0]), 1.0, abs_tol=1e-6)


def test_cfr_plus_clamps_negative_regrets() -> None:
    store = InfosetStore.zeros(InfosetLayout.from_action_counts([2]))

    from pokergpu.cfr import CFRVariant, run_cfr_iteration

    run_cfr_iteration(
        store,
        [np.array([1.0, -3.0], dtype=np.float32)],
        variant=CFRVariant.CFR_PLUS,
    )

    assert np.all(store.regrets >= 0.0)


def test_dcfr_requires_positive_iteration() -> None:
    from pokergpu.cfr import CFRVariant, run_cfr_iteration

    store = InfosetStore.zeros(InfosetLayout.from_action_counts([2]))

    with pytest.raises(ValueError):
        run_cfr_iteration(
            store,
            [np.array([1.0, -1.0], dtype=np.float32)],
            variant=CFRVariant.DCFR,
            iteration=0,
        )

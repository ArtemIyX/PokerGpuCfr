from __future__ import annotations

import pytest

from pokergpu.cfr.solver import (
    DenseCfrState,
    build_dense_infoset_table,
    make_kuhn_public_tree,
    propagate_reach,
)
from pokergpu.tree.public_tree import (
    ChildLink,
    InfosetId,
    NodeId,
    NodeType,
    PublicTree,
)
from pokergpu.core.betting import Chips
from pokergpu.cfr.solver import apply_dense_solver_strategy_update


def test_build_dense_infoset_table_extracts_dense_mappings() -> None:
    tree = make_kuhn_public_tree()

    table = build_dense_infoset_table(tree)

    assert table.infoset_count == 6
    assert table.infoset_to_node[0] == 1
    assert table.action_counts[0] == 2
    assert table.node_to_infoset[1] == 0
    assert table.infoset_nodes[0] == (1,)
    assert table.infoset_order == (0, 1, 2, 3, 4, 5)


def test_build_dense_infoset_table_rejects_missing_player_infoset() -> None:
    with pytest.raises(ValueError, match="player nodes must have infoset ids"):
        PublicTree(
            node_types=(NodeType.PLAYER0, NodeType.TERMINAL),
            first_child=(0, 0),
            child_count=(0, 0),
            children=(),
            infoset_ids=(None, None),
            terminal_payoffs=(None, Chips(1)),
        )


def test_build_dense_infoset_table_groups_repeated_infosets() -> None:
    tree = PublicTree(
        node_types=(
            NodeType.PLAYER0,
            NodeType.PLAYER1,
            NodeType.PLAYER0,
            NodeType.TERMINAL,
            NodeType.TERMINAL,
        ),
        first_child=(0, 2, 4, 4, 4),
        child_count=(2, 2, 2, 0, 0),
        children=(
            ChildLink(child=NodeId(1)),
            ChildLink(child=NodeId(2)),
            ChildLink(child=NodeId(3)),
            ChildLink(child=NodeId(4)),
            ChildLink(child=NodeId(3)),
            ChildLink(child=NodeId(4)),
        ),
        infoset_ids=(InfosetId(0), InfosetId(1), InfosetId(0), None, None),
        terminal_payoffs=(None, None, None, Chips(1), Chips(2)),
    )

    table = build_dense_infoset_table(tree)
    reach = propagate_reach(
        tree,
        infoset_table=table,
        infoset_strategies={InfosetId(0): (0.25, 0.75), InfosetId(1): (1.0, 0.0)},
    )

    assert table.node_to_infoset == (0, 1, 0, -1, -1)
    assert table.infoset_nodes[0] == (0, 2)
    assert table.infoset_order == (0, 1)
    assert reach.infoset_reach[0] == 1.75
    assert reach.cumulative_strategy[0] == (0.4375, 1.3125)


def test_propagate_reach_threaded_matches_serial() -> None:
    tree = make_kuhn_public_tree()
    table = build_dense_infoset_table(tree)
    strategies = {
        InfosetId(0): (0.25, 0.75),
        InfosetId(1): (0.5, 0.5),
        InfosetId(2): (0.6, 0.4),
        InfosetId(3): (0.3, 0.7),
        InfosetId(4): (0.8, 0.2),
        InfosetId(5): (0.1, 0.9),
    }

    serial = propagate_reach(tree, infoset_table=table, infoset_strategies=strategies)
    threaded = propagate_reach(
        tree,
        infoset_table=table,
        infoset_strategies=strategies,
        max_workers=2,
    )

    assert threaded == serial


def test_apply_dense_solver_strategy_update_updates_each_infoset_row() -> None:
    tree = make_kuhn_public_tree()
    table = build_dense_infoset_table(tree)
    state = DenseCfrState(
        regret_sums=((0.0, 0.0),) * table.infoset_count,
        strategy_sums=((0.0, 0.0),) * table.infoset_count,
    )

    next_state = apply_dense_solver_strategy_update(
        state,
        action_values=tuple((1.0, -1.0) for _ in range(table.infoset_count)),
        infoset_table=table,
        reach_weights=tuple(1.0 for _ in range(table.infoset_count)),
    )

    assert next_state.regret_sums[0] == (1.0, -1.0)
    assert next_state.strategy_sums[0] == (0.5, 0.5)


def test_apply_dense_solver_strategy_update_threaded_matches_serial() -> None:
    tree = make_kuhn_public_tree()
    table = build_dense_infoset_table(tree)
    state = DenseCfrState(
        regret_sums=tuple((float(i), float(-i - 1)) for i in range(table.infoset_count)),
        strategy_sums=tuple((0.0, 0.0) for _ in range(table.infoset_count)),
    )
    action_values = tuple((1.0, -1.0) for _ in range(table.infoset_count))
    serial = apply_dense_solver_strategy_update(
        state,
        action_values,
        infoset_table=table,
        reach_weights=tuple(1.0 for _ in range(table.infoset_count)),
    )
    threaded = apply_dense_solver_strategy_update(
        state,
        action_values,
        infoset_table=table,
        reach_weights=tuple(1.0 for _ in range(table.infoset_count)),
        max_workers=2,
    )

    assert threaded == serial

from __future__ import annotations

from benchmarks.benchmark_stage6_backward_cfv import _make_stage6_benchmark_tree
from benchmarks.benchmark_stage6_backward_cfv import _make_state
from pokergpu.cfr.solver import build_dense_infoset_table
from pokergpu.tree.public_tree import NodeType


def test_stage6_benchmark_tree_has_fixed_two_action_player_nodes() -> None:
    tree = _make_stage6_benchmark_tree()
    table = build_dense_infoset_table(tree)
    state = _make_state(table.action_counts)

    assert tree.node_count == 1023
    assert tree.node_types[0] is NodeType.PLAYER0
    assert tree.node_types[1] is NodeType.PLAYER1
    assert tree.node_types[-1] is NodeType.TERMINAL
    assert table.action_counts[:4] == (2, 2, 2, 2)
    assert state.regret_sums[0] == (0.0, 0.0)
    assert len(state.regret_sums) == 511

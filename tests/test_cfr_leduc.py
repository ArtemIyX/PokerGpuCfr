from __future__ import annotations

from pokergpu.cfr.solver import (
    DenseCfrState,
    build_dense_infoset_table,
    make_leduc_public_tree,
    run_leduc_dense_iteration,
    run_leduc_dense_iterations,
    run_leduc_root_iteration,
    SolverState,
)
from pokergpu.tree.public_tree import NodeType


def test_make_leduc_public_tree_has_chance_root() -> None:
    tree = make_leduc_public_tree()

    assert tree.node_types[0] is NodeType.CHANCE
    assert tree.child_count[0] == 6


def test_leduc_root_iteration_runs() -> None:
    state = SolverState(regret_sums=(0.0, 0.0), strategy_sums=(0.0, 0.0))

    result = run_leduc_root_iteration(state)

    assert result.strategy == (0.5, 0.5)


def test_leduc_dense_iteration_threaded_matches_serial() -> None:
    tree = make_leduc_public_tree()
    table = build_dense_infoset_table(tree)
    state = DenseCfrState(
        regret_sums=tuple((float(index), float(-index - 1)) for index in range(table.infoset_count)),
        strategy_sums=tuple((0.0, 0.0) for _ in range(table.infoset_count)),
    )

    serial = run_leduc_dense_iteration(state)
    threaded = run_leduc_dense_iteration(state, max_workers=2)

    assert threaded == serial


def test_leduc_dense_iterations_threaded_matches_serial() -> None:
    tree = make_leduc_public_tree()
    table = build_dense_infoset_table(tree)
    state = DenseCfrState(
        regret_sums=tuple((float(index), float(-index - 1)) for index in range(table.infoset_count)),
        strategy_sums=tuple((0.0, 0.0) for _ in range(table.infoset_count)),
    )

    serial = run_leduc_dense_iterations(state, 3)
    threaded = run_leduc_dense_iterations(state, 3, max_workers=2)

    assert threaded == serial

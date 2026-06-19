from __future__ import annotations

from pokergpu.cfr.solver import CfrVariant
from pokergpu.cfr.solver import GameStateMode
from pokergpu.cfr.solver import GameStateSpec
from pokergpu.cfr.solver import GameVariant
from pokergpu.cfr.solver import SolverStageRequest
from pokergpu.cfr.solver import aggregate_root_action_values
from pokergpu.cfr.solver import make_game_public_tree
from pokergpu.cfr.solver import run_solver_stage
from pokergpu.cfr.solver.infosets import build_dense_infoset_table
from pokergpu.cfr.solver.kuhn import make_kuhn_public_tree
from pokergpu.cfr.solver.leduc import make_leduc_public_tree
from pokergpu.cfr.solver.state import DenseCfrState
from pokergpu.core.board import Board


def test_root_strategy_differs_between_kuhn_and_leduc() -> None:
    kuhn_strategy = _run_and_get_root_strategy(GameVariant.KUHN, iterations=3)
    leduc_strategy = _run_and_get_root_strategy(GameVariant.LEDUC, iterations=3)

    assert kuhn_strategy != leduc_strategy


def test_root_strategy_changes_with_iterations() -> None:
    one_iter = _run_and_get_root_strategy(GameVariant.KUHN, iterations=1)
    three_iter = _run_and_get_root_strategy(GameVariant.KUHN, iterations=3)

    assert one_iter != three_iter


def test_root_strategy_differs_across_seeds_for_same_game() -> None:
    seed_1 = _run_and_get_root_strategy(GameVariant.KUHN, iterations=3, seed=1)
    seed_2 = _run_and_get_root_strategy(GameVariant.KUHN, iterations=3, seed=95)

    assert seed_1 != seed_2


def test_root_strategy_differs_between_random_and_exact_state() -> None:
    random_state = _run_and_get_root_strategy(
        GameVariant.KUHN,
        iterations=3,
        seed=11,
        game_state=GameStateSpec(mode=GameStateMode.RANDOM, seed=11),
    )
    exact_state = _run_and_get_root_strategy(
        GameVariant.KUHN,
        iterations=3,
        seed=11,
        game_state=GameStateSpec(mode=GameStateMode.EXACT, seed=11, encoded_state=b"state"),
    )

    assert random_state != exact_state


def test_dense_solver_iteration_uses_game_specific_tree_shape() -> None:
    kuhn_tree = make_kuhn_public_tree()
    leduc_tree = make_leduc_public_tree()

    assert kuhn_tree.node_count != 0
    assert leduc_tree.node_count != 0
    assert kuhn_tree.node_count != leduc_tree.node_count


def test_root_action_values_differ_between_kuhn_and_leduc() -> None:
    kuhn_tree = make_kuhn_public_tree()
    leduc_tree = make_leduc_public_tree()

    kuhn_values = aggregate_root_action_values(kuhn_tree)
    leduc_values = aggregate_root_action_values(leduc_tree)

    assert kuhn_values != leduc_values
    assert kuhn_values[0] != leduc_values[0] or kuhn_values[1] != leduc_values[1]


def test_root_infoset_shapes_differ_between_kuhn_and_leduc() -> None:
    kuhn_table = build_dense_infoset_table(make_kuhn_public_tree())
    leduc_table = build_dense_infoset_table(make_leduc_public_tree())

    assert kuhn_table.infoset_count == 6
    assert leduc_table.infoset_count != 6 or leduc_table.infoset_nodes != kuhn_table.infoset_nodes
    assert kuhn_table.infoset_nodes != leduc_table.infoset_nodes


def _run_and_get_root_strategy(
    game: GameVariant,
    *,
    iterations: int,
    seed: int | None = None,
    game_state: GameStateSpec | None = None,
) -> tuple[float, ...]:
    tree = make_game_public_tree(game)
    table = build_dense_infoset_table(tree)
    dense_state = DenseCfrState(
        regret_sums=tuple((0.0, 0.0) for _ in range(table.infoset_count)),
        strategy_sums=tuple((0.0, 0.0) for _ in range(table.infoset_count)),
    )
    request = SolverStageRequest(
        game=game,
        cfr_variant=CfrVariant.CFR,
        depth_limit=2,
        iterations=iterations,
        seed=seed,
        state=game_state,
    )
    board = Board(cards=())
    result = None
    current = dense_state
    for _ in range(iterations):
        result = run_solver_stage(request, tree=tree, dense_state=current, board=board)
        assert isinstance(result.final_state, DenseCfrState)
        current = result.final_state
    assert result is not None
    root_row = current.strategy_sums[table.infoset_order[0]]
    total = sum(max(0.0, value) for value in root_row)
    if total <= 0.0:
        return tuple(1.0 / len(root_row) for _ in root_row)
    return tuple(max(0.0, value) / total for value in root_row)

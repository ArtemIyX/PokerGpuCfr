from __future__ import annotations

import pytest

from pokergpu.cfr.solver import (
    DenseCfrState,
    GameVariant,
    build_dense_infoset_table,
    make_game_public_tree,
    make_kuhn_public_tree,
    propagate_reach,
)
from pokergpu.cfr.solver.chunking import chunk_indices
from pokergpu.cfr.solver.tree import make_holdem_hu_public_tree
from pokergpu.core.board import Board
from pokergpu.tree.public_tree import (
    ChildLink,
    InfosetId,
    NodeId,
    NodeType,
    PublicTree,
)
from pokergpu.core.betting import Chips
from pokergpu.cfr.solver import apply_dense_solver_strategy_update
from pokergpu.cfr.solver.state import DenseCfrState as SolverDenseCfrState


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


def test_holdem_hu_infosets_distinguish_different_board_states() -> None:
    empty_tree = make_holdem_hu_public_tree(state=_make_boardless_holdem_state(Board(cards=())))
    flop_tree = make_holdem_hu_public_tree(state=_make_boardless_holdem_state(Board.from_str("AhKdTc")))

    empty_table = build_dense_infoset_table(empty_tree)
    flop_table = build_dense_infoset_table(flop_tree)

    assert empty_table.infoset_count > 0
    assert flop_table.infoset_count > 0
    assert empty_tree.infoset_ids != flop_tree.infoset_ids
    assert empty_table.infoset_order == tuple(range(empty_table.infoset_count))
    assert flop_table.infoset_order == tuple(range(flop_table.infoset_count))
    assert empty_table.action_counts != flop_table.action_counts or empty_table.action_labels != flop_table.action_labels


def _make_boardless_holdem_state(board: Board):
    from pokergpu.core.betting import BettingRoundState
    from pokergpu.core.betting import BlindStructure
    from pokergpu.core.betting import PlayerBet
    from pokergpu.core.betting import PlayerIndex
    from pokergpu.core.betting import PlayerStack
    from pokergpu.core.betting import Pot
    from pokergpu.core.betting import chips
    from pokergpu.core.state import GameState
    from pokergpu.core.state import PlayerState

    return GameState(
        board=board,
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(3)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(1000)),
                PlayerStack(player=PlayerIndex(1), stack=chips(1000)),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(1)),
                PlayerBet(player=PlayerIndex(1), committed=chips(2)),
            ),
            blinds=BlindStructure(small_blind=chips(1), big_blind=chips(2)),
            to_act=PlayerIndex(0),
        ),
        dealer=PlayerIndex(0),
    )


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


def test_dense_cfr_state_allows_empty_tables() -> None:
    state = SolverDenseCfrState(regret_sums=(), strategy_sums=())

    assert state.regret_sums == ()
    assert state.strategy_sums == ()


def test_chunk_indices_splits_deterministically() -> None:
    assert chunk_indices([0, 1, 2, 3, 4], 2) == ((0, 1, 2), (3, 4))
    assert chunk_indices([0, 1, 2, 3, 4], 10) == ((0,), (1,), (2,), (3,), (4,))
    assert chunk_indices([], 4) == ()

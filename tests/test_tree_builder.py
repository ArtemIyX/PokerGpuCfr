from pokergpu.abstraction.actions import BaselineActionAbstraction, make_compact_profile
from pokergpu.core.betting import (
    BettingRoundState,
    BlindStructure,
    PlayerBet,
    PlayerIndex,
    PlayerStack,
    Pot,
    chips,
)
from pokergpu.core.board import Board
from pokergpu.core.state import GameState, HandPhase, PlayerState
from pokergpu.tree.builder import (
    TreeBuildConfig,
    build_public_tree,
    build_shallow_public_tree,
)


def test_build_shallow_public_tree_creates_root_and_children() -> None:
    state = GameState(
        board=Board(cards=()),
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(150)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(900)),
                PlayerStack(player=PlayerIndex(1), stack=chips(800)),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(0)),
                PlayerBet(player=PlayerIndex(1), committed=chips(0)),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        ),
        dealer=PlayerIndex(0),
    )

    built = build_shallow_public_tree(
        state,
        abstraction=BaselineActionAbstraction(profile=make_compact_profile()),
    )

    assert built.tree.node_count == 3
    assert built.tree.child_count[0] == 2
    assert len(built.actions_by_node[0]) == 2
    assert built.node_states[1].phase is HandPhase.IN_PROGRESS


def test_build_shallow_public_tree_marks_terminal_fold_child() -> None:
    state = GameState(
        board=Board(cards=()),
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(150)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(900)),
                PlayerStack(player=PlayerIndex(1), stack=chips(800)),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(300)),
                PlayerBet(player=PlayerIndex(1), committed=chips(100)),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(1),
        ),
        dealer=PlayerIndex(0),
    )

    built = build_shallow_public_tree(
        state,
        abstraction=BaselineActionAbstraction(profile=make_compact_profile()),
    )

    assert any(
        node_state.phase is HandPhase.TERMINAL
        for node_state in built.node_states[1:]
    )


def test_build_public_tree_can_expand_beyond_one_ply() -> None:
    state = GameState(
        board=Board(cards=()),
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(150)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(900)),
                PlayerStack(player=PlayerIndex(1), stack=chips(800)),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(0)),
                PlayerBet(player=PlayerIndex(1), committed=chips(0)),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        ),
        dealer=PlayerIndex(0),
    )

    built = build_public_tree(state, config=TreeBuildConfig(max_depth=2, max_nodes=16))

    assert built.tree.node_count > 3
    assert any(actions for actions in built.actions_by_node[1:])
    assert any(
        built.tree.is_frontier[node_index]
        for node_index in range(3, built.tree.node_count)
    )


def test_build_public_tree_respects_max_nodes() -> None:
    state = GameState(
        board=Board(cards=()),
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(150)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(900)),
                PlayerStack(player=PlayerIndex(1), stack=chips(800)),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(300)),
                PlayerBet(player=PlayerIndex(1), committed=chips(100)),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(1),
        ),
        dealer=PlayerIndex(0),
    )

    built = build_public_tree(state, config=TreeBuildConfig(max_depth=3, max_nodes=4))

    assert built.tree.node_count <= 4


def test_build_public_tree_marks_depth_frontier_nodes() -> None:
    state = GameState(
        board=Board(cards=()),
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(150)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(900)),
                PlayerStack(player=PlayerIndex(1), stack=chips(800)),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(0)),
                PlayerBet(player=PlayerIndex(1), committed=chips(0)),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        ),
        dealer=PlayerIndex(0),
    )

    built = build_public_tree(state, config=TreeBuildConfig(max_depth=1, max_nodes=16))

    assert built.tree.is_frontier[1]
    assert built.tree.is_frontier[2]


def test_build_public_tree_records_action_abstraction_id() -> None:
    state = GameState(
        board=Board(cards=()),
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(150)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(900)),
                PlayerStack(player=PlayerIndex(1), stack=chips(800)),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(0)),
                PlayerBet(player=PlayerIndex(1), committed=chips(0)),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        ),
        dealer=PlayerIndex(0),
    )

    built = build_public_tree(
        state,
        abstraction=BaselineActionAbstraction(profile=make_compact_profile()),
    )

    assert built.action_abstraction_id == "compact:v2|preflop|late"

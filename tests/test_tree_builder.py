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
    ChanceOutcome,
    TreeBuildConfig,
    build_public_tree,
    build_shallow_public_tree,
)
from pokergpu.tree.public_tree import NodeType
from pokergpu.cfr.solver.tree import make_holdem_hu_public_tree


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

    assert built.tree.node_count == 4
    assert built.tree.child_count[0] == 3
    assert len(built.actions_by_node[0]) == 3
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


def test_build_public_tree_can_emit_chance_root() -> None:
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

    def expand_chance(current_state: GameState) -> tuple[ChanceOutcome, ...] | None:
        if current_state.board.cards:
            return None
        return (
            ChanceOutcome(state=current_state, probability=0.25),
            ChanceOutcome(state=current_state, probability=0.75),
        )

    built = build_public_tree(
        state,
        config=TreeBuildConfig(max_depth=1, max_nodes=8),
        expand_chance=expand_chance,
    )

    assert built.tree.node_types[0] is not NodeType.PLAYER0
    assert built.tree.node_types[0] is NodeType.CHANCE
    assert built.tree.child_count[0] == 2
    assert built.tree.children[0].chance_prob == 0.25


def test_holdem_hu_public_tree_starts_with_chance_root() -> None:
    tree = make_holdem_hu_public_tree(compact=False)

    assert tree.node_types[0] is NodeType.CHANCE
    assert tree.child_count[0] == 4
    assert any(node_type is NodeType.PLAYER0 or node_type is NodeType.PLAYER1 for node_type in tree.node_types)

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
    assert built.canonical_board_key == ""


def test_build_public_tree_uses_canonical_board_key() -> None:
    state_a = GameState(
        board=Board.from_str("AhKhQd"),
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
    state_b = GameState(
        board=Board.from_str("AcKcQd"),
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

    built_a = build_public_tree(state_a, 
                                config=TreeBuildConfig(max_depth=1, max_nodes=8))
    built_b = build_public_tree(state_b, 
                                config=TreeBuildConfig(max_depth=1, max_nodes=8))

    assert built_a.canonical_board_key == built_b.canonical_board_key


def test_build_public_tree_handles_heads_up_postflop_state() -> None:
    state = GameState(
        board=Board.from_str("AhKdTc"),
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(300)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(1700)),
                PlayerStack(player=PlayerIndex(1), stack=chips(1700)),
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
        config=TreeBuildConfig(max_depth=2, max_nodes=32),
    )

    assert built.tree.node_count > 0
    assert built.tree.node_types[0].value in {"player0", "player1"}
    assert built.tree.is_frontier[0] is False
    assert built.tree.first_child[0] == 0 or built.tree.child_count[0] > 0
    assert any(built.tree.is_frontier)
    assert any(
        node_state.current_street.value == "flop"
        for node_state in built.node_states
    )
    assert all(
        built.tree.child_count[index] == 0
        for index, is_frontier in enumerate(built.tree.is_frontier)
        if is_frontier
    )


def test_build_public_tree_records_player_metadata() -> None:
    state = GameState(
        board=Board.from_str("AhKdTc"),
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1)),
            PlayerState(player=PlayerIndex(2)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(300)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(1700)),
                PlayerStack(player=PlayerIndex(1), stack=chips(1700)),
                PlayerStack(player=PlayerIndex(2), stack=chips(1700)),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(0)),
                PlayerBet(player=PlayerIndex(1), committed=chips(0)),
                PlayerBet(player=PlayerIndex(2), committed=chips(0)),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        ),
        dealer=PlayerIndex(0),
    )

    built = build_public_tree(
        state,
        config=TreeBuildConfig(max_depth=1, max_nodes=8),
    )

    assert built.player_count == 3
    assert built.active_players == (0, 1, 2)


def test_build_public_tree_exposes_flat_level_layout() -> None:
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

    built_a = build_public_tree(state, config=TreeBuildConfig(max_depth=2, max_nodes=16))
    built_b = build_public_tree(state, config=TreeBuildConfig(max_depth=2, max_nodes=16))

    assert built_a.flat_view.node_depth == built_a.template.depth
    assert built_a.flat_view.first_child == built_a.tree.first_child
    assert built_a.flat_view.child_count == built_a.tree.child_count
    assert built_a.level_schedule == built_a.template.level_schedule
    assert built_a.template.level_schedule == built_b.template.level_schedule

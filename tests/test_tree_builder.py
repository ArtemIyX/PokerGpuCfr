import pytest

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
from pokergpu.tree.public_tree import (
    ChildLink,
    InfosetId,
    NodeId,
    NodeType,
    PublicTree,
    PublicTreeFlatView,
    PublicTreeLevelSchedule,
    PublicTreeTemplate,
)
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
    assert built.template.node_count == built.tree.node_count
    assert built.tree.first_child == built.template.first_child
    assert built.tree.child_count == built.template.child_count


def test_build_public_tree_max_nodes_keeps_tree_valid() -> None:
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

    built = build_public_tree(state, config=TreeBuildConfig(max_depth=4, max_nodes=3))

    assert built.tree.node_count == 3
    assert built.tree.first_child[0] + built.tree.child_count[0] <= len(built.tree.children)
    assert all(
        built.tree.first_child[i] + built.tree.child_count[i] <= len(built.tree.children)
        for i in range(built.tree.node_count)
    )
    assert built.template.depth == built.flat_view.node_depth
    assert built.template.level_schedule.forward_levels[0] == (0,)


def test_build_public_tree_max_nodes_clamps_partial_parent_children() -> None:
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

    built = build_public_tree(state, config=TreeBuildConfig(max_depth=4, max_nodes=2))

    assert built.tree.node_count == 2
    assert built.tree.first_child[0] == 0
    assert built.tree.child_count[0] > 0
    assert built.tree.child_count[1] == 0
    assert built.tree.first_child[0] + built.tree.child_count[0] <= len(built.tree.children)
    assert built.template.depth == built.flat_view.node_depth
    assert built.template.level_schedule.forward_levels[0] == (0,)


def test_public_tree_rejects_invalid_child_range() -> None:
    with pytest.raises(ValueError):
        PublicTree(
            node_types=(NodeType.PLAYER0, NodeType.TERMINAL),
            is_frontier=(False, True),
            first_child=(0, 1),
            child_count=(2, 0),
            children=(ChildLink(child=NodeId(1)),),
            infoset_ids=(InfosetId(0), None),
            terminal_payoffs=(None, chips(0)),
        )


def test_public_tree_template_rebuild_keeps_child_ranges_valid() -> None:
    tree = PublicTree(
        node_types=(NodeType.PLAYER0, NodeType.TERMINAL),
        is_frontier=(False, True),
        first_child=(0, 1),
        child_count=(1, 0),
        children=(ChildLink(child=NodeId(1)),),
        infoset_ids=(InfosetId(0), None),
        terminal_payoffs=(None, chips(0)),
    )
    template = PublicTreeTemplate(
        node_types=tree.node_types,
        is_frontier=tree.is_frontier,
        first_child=tree.first_child,
        child_count=tree.child_count,
        infoset_ids=tree.infoset_ids,
        terminal_payoffs=tree.terminal_payoffs,
        depth=(0, 1),
        street=("flop", "flop"),
        level_schedule=PublicTreeLevelSchedule(
            forward_levels=((0,), (1,)),
            backward_levels=((1,), (0,)),
            level_nodes=((0,), (1,)),
        ),
        flat_view=PublicTreeFlatView(
            node_type=tree.node_types,
            node_depth=(0, 1),
            node_level=(0, 1),
            street=(1, 1),
            infoset_id=(0, -1),
            first_child=tree.first_child,
            child_count=tree.child_count,
            is_frontier=tree.is_frontier,
            terminal_payoff=(0.0, 0.0),
            edge_parent=(0,),
            edge_child=(1,),
            edge_action_slot=(0,),
            edge_chance_prob=(0.0,),
            edge_infoset_id=(0,),
            edge_player=(0,),
        ),
        canonical_board_key="",
        action_abstraction_id="test",
        tree_key="test",
    )

    rebuilt = template.as_public_tree(tree.children)
    assert rebuilt.first_child == tree.first_child
    assert rebuilt.child_count == tree.child_count


def test_build_public_tree_template_round_trip_after_truncation() -> None:
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

    built = build_public_tree(state, config=TreeBuildConfig(max_depth=4, max_nodes=2))
    rebuilt = built.template.as_public_tree(built.tree.children)

    assert rebuilt.node_count == built.tree.node_count
    assert rebuilt.first_child == built.tree.first_child
    assert rebuilt.child_count == built.tree.child_count


def test_build_public_tree_small_cap_postflop_round_trip() -> None:
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
        config=TreeBuildConfig(max_depth=2, max_nodes=2),
    )

    rebuilt = built.template.as_public_tree(built.tree.children)

    assert rebuilt.first_child == built.tree.first_child
    assert rebuilt.child_count == built.tree.child_count
    assert rebuilt.children == built.tree.children


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

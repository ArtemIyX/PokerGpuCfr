from pokergpu.abstraction.actions import BaselineActionAbstraction, make_compact_profile
from pokergpu.abstraction.hands import RangeVector
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
from pokergpu.core.state import GameState, PlayerState
from pokergpu.runtime.gpu_postflop import _rebuilt_tree_from_template
from pokergpu.runtime.postflop import PostflopResolveSpec
from pokergpu.runtime.caching import PublicStateFingerprint
from pokergpu.tree.builder import TreeBuildConfig, build_public_tree


def _make_state(board: str = "") -> GameState:
    return GameState(
        board=Board.from_str(board) if board else Board(cards=()),
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


def test_public_tree_template_contains_flat_arrays() -> None:
    built = build_public_tree(
        _make_state(),
        abstraction=BaselineActionAbstraction(profile=make_compact_profile()),
        config=TreeBuildConfig(max_depth=1, max_nodes=8),
    )

    template = built.template
    assert template.node_count == built.tree.node_count
    assert template.node_types == built.tree.node_types
    assert template.first_child == built.tree.first_child
    assert template.child_count == built.tree.child_count
    assert template.infoset_ids == built.tree.infoset_ids
    assert template.terminal_payoffs == built.tree.terminal_payoffs
    assert len(template.depth) == built.tree.node_count
    assert len(template.street) == built.tree.node_count
    assert template.level_schedule.forward_levels[0] == (0,)
    assert template.level_schedule.backward_levels[-1] == (0,)
    assert template.flat_view.first_child == built.tree.first_child
    assert template.flat_view.child_count == built.tree.child_count


def test_public_tree_template_is_deterministic() -> None:
    state = _make_state("AhKhQd")
    built_a = build_public_tree(
        state,
        abstraction=BaselineActionAbstraction(profile=make_compact_profile()),
        config=TreeBuildConfig(max_depth=2, max_nodes=16),
    )
    built_b = build_public_tree(
        state,
        abstraction=BaselineActionAbstraction(profile=make_compact_profile()),
        config=TreeBuildConfig(max_depth=2, max_nodes=16),
    )

    assert built_a.template.tree_key == built_b.template.tree_key
    assert built_a.template.canonical_board_key == built_b.template.canonical_board_key
    assert built_a.template.action_abstraction_id == built_b.template.action_abstraction_id
    assert built_a.template.flat_view.edge_child == built_b.template.flat_view.edge_child
    assert built_a.template.level_schedule == built_b.template.level_schedule


def test_public_tree_template_canonical_board_key_matches_isomorphic_boards() -> None:
    built_a = build_public_tree(
        _make_state("AhKhQd"),
        abstraction=BaselineActionAbstraction(profile=make_compact_profile()),
        config=TreeBuildConfig(max_depth=1, max_nodes=8),
    )
    built_b = build_public_tree(
        _make_state("AcKcQd"),
        abstraction=BaselineActionAbstraction(profile=make_compact_profile()),
        config=TreeBuildConfig(max_depth=1, max_nodes=8),
    )

    assert built_a.template.canonical_board_key == built_b.template.canonical_board_key


def test_public_tree_template_round_trip_with_truncation() -> None:
    built = build_public_tree(
        _make_state(),
        abstraction=BaselineActionAbstraction(profile=make_compact_profile()),
        config=TreeBuildConfig(max_depth=4, max_nodes=2),
    )

    rebuilt = built.template.as_public_tree(built.tree.children)

    assert rebuilt.node_count == built.tree.node_count
    assert rebuilt.first_child == built.tree.first_child
    assert rebuilt.child_count == built.tree.child_count
    assert rebuilt.children == built.tree.children


def test_public_tree_template_round_trip_various_node_caps() -> None:
    for max_nodes in (2, 3, 4, 5, 8):
        built = build_public_tree(
            _make_state(),
            abstraction=BaselineActionAbstraction(profile=make_compact_profile()),
            config=TreeBuildConfig(max_depth=4, max_nodes=max_nodes),
        )

        rebuilt = built.template.as_public_tree(built.tree.children)

        assert rebuilt.node_count == built.tree.node_count
        assert rebuilt.first_child == built.tree.first_child
        assert rebuilt.child_count == built.tree.child_count
        assert rebuilt.children == built.tree.children


def test_rebuilt_tree_from_template_matches_original_tree() -> None:
    state = _make_state("AhKhQd")
    built = build_public_tree(
        state,
        abstraction=BaselineActionAbstraction(profile=make_compact_profile()),
        config=TreeBuildConfig(max_depth=4, max_nodes=2),
    )
    spec = PostflopResolveSpec(
        state=state,
        range_p0=RangeVector.uniform(),
        range_p1=RangeVector.uniform(),
        time_budget_sec=0.0,
        max_depth=4,
        max_nodes=2,
    )

    rebuilt = _rebuilt_tree_from_template(spec, built.template)

    assert rebuilt.tree.node_count == built.tree.node_count
    assert rebuilt.tree.first_child == built.tree.first_child
    assert rebuilt.tree.child_count == built.tree.child_count
    assert rebuilt.tree.children == built.tree.children
    assert rebuilt.template.depth == built.template.depth


def test_public_state_fingerprint_tree_template_key_uses_required_fields() -> None:
    fingerprint = PublicStateFingerprint(
        variant="nlhe",
        street="flop",
        acting_player=1,
        pot=300,
        stacks=(1700, 1700),
        blinds=(50, 100),
        antes=(0, 0),
        board=("Ah", "Kh", "Qd"),
        action_history=(),
        action_abstraction_id="postflop_mvp:v1|flop|oop",
        range_abstraction_id="private_hand_v1",
        subtree_depth_limit=2,
        evaluator_id="Dummy",
        solver_version="mvp-postflop-v1",
        player_count=2,
        active_players=(0, 1),
        canonical_board="AhKhQd",
        card_removal_version="1",
    )

    key = fingerprint.tree_template_key()

    assert key.street == "flop"
    assert key.canonical_board == "AhKhQd"
    assert key.pot == 300
    assert key.stacks == (1700, 1700)
    assert key.to_act == 1
    assert key.action_abstraction_id == "postflop_mvp:v1|flop|oop"
    assert key.digest() == "flop|AhKhQd|300|1700,1700|1|postflop_mvp:v1|flop|oop"

from __future__ import annotations

import pytest

from pokergpu.cfr.solver import GameStateMode
from pokergpu.cfr.solver import GameStateSpec
from pokergpu.cfr.solver import GameVariant
from pokergpu.cfr.solver import make_game_public_tree
from pokergpu.cfr.solver import resolve_game_state_spec


def test_make_game_public_tree_selects_kuhn_and_leduc() -> None:
    kuhn = make_game_public_tree(GameVariant.KUHN)
    leduc = make_game_public_tree(GameVariant.LEDUC)
    holdem = make_game_public_tree(GameVariant.HOLDEM_HU)

    assert kuhn.node_count > 0
    assert leduc.node_count > 0
    assert holdem.node_count > 0
    assert kuhn != leduc


def test_make_game_public_tree_rejects_placeholder_variants() -> None:
    with pytest.raises(NotImplementedError, match="unsupported game variant 'holdem_6max'"):
        make_game_public_tree(GameVariant.HOLDEM_6MAX)


def test_make_game_public_tree_builds_holdem_street_scaffold() -> None:
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)

    assert tree.node_count == 25
    assert tree.node_types[0].value == "player0"
    assert all(node_type.value == "player1" for node_type in tree.node_types[1:7])
    assert all(node_type.value == "player0" for node_type in tree.node_types[7:13])
    assert all(node_type.value == "player1" for node_type in tree.node_types[13:19])
    assert all(node_type.value == "terminal" for node_type in tree.node_types[19:24])
    assert tree.node_types[-1].value == "leaf"
    assert tree.child_count[0] == 6
    assert all(tree.child_count[index] == 1 for index in range(1, 19))
    assert all(tree.child_count[index] == 0 for index in range(19, tree.node_count))
    assert tree.child_count[tree.node_count - 1] == 0
    root_labels = tree.action_labels[0]
    assert root_labels is not None
    assert root_labels == ("check", "bet:25pct", "bet:50pct", "bet:75pct", "bet:100pct", "bet:150pct")


def test_make_game_public_tree_builds_multistreet_holdem_shape() -> None:
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)

    assert tree.node_count >= 25
    assert sum(1 for node_type in tree.node_types if node_type.value == "player0") + sum(
        1 for node_type in tree.node_types if node_type.value == "player1"
    ) >= 8
    assert any(node_type.value == "leaf" for node_type in tree.node_types)
    assert any(node_type.value == "terminal" for node_type in tree.node_types)
    assert len({count for count in tree.child_count if count > 0}) >= 2
    root_labels = tree.action_labels[0]
    assert root_labels is not None
    assert root_labels[0] == "check"


def test_make_game_public_tree_is_deterministic_for_holdem() -> None:
    first = make_game_public_tree(GameVariant.HOLDEM_HU)
    second = make_game_public_tree(GameVariant.HOLDEM_HU)

    assert first == second


def test_make_game_public_tree_can_build_compact_holdem_tree() -> None:
    full_tree = make_game_public_tree(GameVariant.HOLDEM_HU)
    compact_tree = make_game_public_tree(GameVariant.HOLDEM_HU, compact=True)

    assert compact_tree.node_count < full_tree.node_count
    assert compact_tree.child_count[0] < full_tree.child_count[0]
    assert compact_tree.node_types[-1].value == "leaf"


def test_make_game_public_tree_uses_depth_limit_for_holdem_shape() -> None:
    compact_tree = make_game_public_tree(GameVariant.HOLDEM_HU, depth_limit=1)
    full_tree = make_game_public_tree(GameVariant.HOLDEM_HU, depth_limit=3)

    assert compact_tree.node_count <= full_tree.node_count
    assert compact_tree.child_count[0] <= full_tree.child_count[0]


def test_resolve_game_state_spec_preserves_exact_and_random_modes() -> None:
    exact = GameStateSpec(mode=GameStateMode.EXACT, encoded_state=b"state")
    random = GameStateSpec(mode=GameStateMode.RANDOM, seed=7)

    assert resolve_game_state_spec(exact) == exact
    assert resolve_game_state_spec(random) == random
    assert resolve_game_state_spec(None) is None


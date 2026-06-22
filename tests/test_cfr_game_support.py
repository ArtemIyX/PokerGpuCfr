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

    assert tree.node_count == 10
    assert tree.node_types[0].value == "player0"
    assert tree.node_types[1].value == "player1"
    assert tree.node_types[2].value == "player0"
    assert tree.node_types[3].value == "player1"
    assert all(node_type.value == "terminal" for node_type in tree.node_types[4:])
    assert tree.child_count[0] == 6
    assert tree.child_count[1] == 6
    assert tree.child_count[2] == 5
    assert tree.child_count[3] == 6
    assert any(node_type.value == "terminal" for node_type in tree.node_types)
    assert tree.action_labels[0] == ("check", "bet:25pct", "bet:50pct", "bet:75pct", "bet:100pct", "bet:150pct")


def test_make_game_public_tree_is_deterministic_for_holdem() -> None:
    first = make_game_public_tree(GameVariant.HOLDEM_HU)
    second = make_game_public_tree(GameVariant.HOLDEM_HU)

    assert first == second


def test_resolve_game_state_spec_preserves_exact_and_random_modes() -> None:
    exact = GameStateSpec(mode=GameStateMode.EXACT, encoded_state=b"state")
    random = GameStateSpec(mode=GameStateMode.RANDOM, seed=7)

    assert resolve_game_state_spec(exact) == exact
    assert resolve_game_state_spec(random) == random
    assert resolve_game_state_spec(None) is None


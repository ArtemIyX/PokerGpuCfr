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

    assert kuhn.node_count > 0
    assert leduc.node_count > 0
    assert kuhn != leduc


def test_make_game_public_tree_rejects_placeholder_variants() -> None:
    with pytest.raises(NotImplementedError, match="public tree factory not implemented"):
        make_game_public_tree(GameVariant.HOLDEM_HU)

    with pytest.raises(NotImplementedError, match="public tree factory not implemented"):
        make_game_public_tree(GameVariant.HOLDEM_6MAX)


def test_resolve_game_state_spec_preserves_exact_and_random_modes() -> None:
    exact = GameStateSpec(mode=GameStateMode.EXACT, encoded_state=b"state")
    random = GameStateSpec(mode=GameStateMode.RANDOM, seed=7)

    assert resolve_game_state_spec(exact) == exact
    assert resolve_game_state_spec(random) == random
    assert resolve_game_state_spec(None) is None


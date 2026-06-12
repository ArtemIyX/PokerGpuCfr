from __future__ import annotations

import numpy as np
import pytest

from pokergpu.abstraction.hands import RangeVector, private_hand_index
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
from pokergpu.core.cards import card_from_str
from pokergpu.core.state import GameState, PlayerState
from pokergpu.abstraction.actions import BaselineActionAbstraction, make_postflop_mvp_profile
from pokergpu.runtime import PostflopResolveSpec, SolveCacheState, resolve_postflop_hu
from pokergpu.runtime.gpu_postflop import resolve_postflop_gpu
from pokergpu.tree.builder import TreeBuildConfig, build_public_tree


def _make_state(board_text: str) -> GameState:
    return GameState(
        board=Board.from_str(board_text),
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


@pytest.mark.parametrize("board_text", ["AhKdTc", "2c7dJh", "AsKsQd"])
def test_postflop_solver_runs_on_several_boards(board_text: str) -> None:
    result = resolve_postflop_hu(
        PostflopResolveSpec(
            state=_make_state(board_text),
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            time_budget_sec=0.0,
            max_depth=1,
            max_nodes=16,
        )
    )

    assert result.root_infoset_id == 0
    assert result.root_actions
    assert result.root_strategy.shape == result.root_action_ev_player0.shape
    assert np.isclose(float(result.root_strategy.sum()), 1.0)
    assert np.isclose(result.root_ev_player1, -result.root_ev_player0)


def test_blocked_cards_are_removed_before_solve() -> None:
    state = _make_state("AhKdTc")
    blocked_index = int(private_hand_index(card_from_str("Ah"), card_from_str("Qs")))
    live_index = int(private_hand_index(card_from_str("2c"), card_from_str("3d")))
    values_p0 = np.zeros(1326, dtype=np.float32)
    values_p1 = np.zeros(1326, dtype=np.float32)
    values_p0[blocked_index] = 1.0
    values_p0[live_index] = 2.0
    values_p1[live_index] = 1.0
    range_p0 = RangeVector.from_values(values_p0)
    range_p1 = RangeVector.from_values(values_p1)

    result = resolve_postflop_hu(
        PostflopResolveSpec(
            state=state,
            range_p0=range_p0,
            range_p1=range_p1,
            time_budget_sec=0.0,
            max_depth=1,
            max_nodes=16,
        )
    )

    assert np.isclose(float(result.root_strategy.sum()), 1.0)


def test_root_actions_match_legal_action_generation() -> None:
    state = _make_state("AhKdTc")
    built = build_public_tree(
        state,
        abstraction=BaselineActionAbstraction(profile=make_postflop_mvp_profile()),
        config=TreeBuildConfig(max_depth=1, max_nodes=16),
    )
    result = resolve_postflop_hu(
        PostflopResolveSpec(
            state=state,
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            time_budget_sec=0.0,
            max_depth=1,
            max_nodes=16,
        )
    )

    assert len(result.root_actions) == len(built.actions_by_node[0])
    assert result.root_actions[0] == "check"
    assert all(
        action.startswith("bet") for action in result.root_actions[1:]
    )


def test_terminal_ev_calculation_is_zero_sum() -> None:
    state = GameState(
        board=Board.from_str("AhKdTc9s2d"),
        players=(
            PlayerState(player=PlayerIndex(0), hole_cards=(card_from_str("As"), card_from_str("Ks"))),
            PlayerState(player=PlayerIndex(1), hole_cards=(card_from_str("Qh"), card_from_str("Jh"))),
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
    result = resolve_postflop_hu(
        PostflopResolveSpec(
            state=state,
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            time_budget_sec=0.0,
            max_depth=1,
            max_nodes=16,
        )
    )

    assert np.isclose(result.root_ev_player0 + result.root_ev_player1, 0.0)


def test_postflop_solver_is_deterministic_for_same_input() -> None:
    spec = PostflopResolveSpec(
        state=_make_state("AhKdTc"),
        range_p0=RangeVector.uniform(),
        range_p1=RangeVector.uniform(),
        time_budget_sec=0.0,
        seed=123,
        max_depth=1,
        max_nodes=16,
    )

    result_a = resolve_postflop_hu(spec)
    result_b = resolve_postflop_hu(spec)

    assert np.allclose(result_a.root_strategy, result_b.root_strategy)
    assert np.allclose(result_a.root_action_ev_player0, result_b.root_action_ev_player0)
    assert np.allclose(result_a.root_action_ev_player1, result_b.root_action_ev_player1)
    assert result_a.root_infoset_id == result_b.root_infoset_id


def test_warm_start_reuse_keeps_cache_hot() -> None:
    cache = SolveCacheState()
    spec = PostflopResolveSpec(
        state=_make_state("AhKdTc"),
        range_p0=RangeVector.uniform(),
        range_p1=RangeVector.uniform(),
        time_budget_sec=0.0,
        max_depth=1,
        max_nodes=16,
        cache_state=cache,
    )

    resolve_postflop_hu(spec)
    resolve_postflop_hu(spec)

    assert cache.bundle.warm_start.stats()["entries"] > 0


def test_cuda_solver_is_available_when_torch_has_cuda() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test")

    result = resolve_postflop_gpu(
        PostflopResolveSpec(
            state=_make_state("AhKdTc"),
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            time_budget_sec=0.0,
            max_depth=1,
            max_nodes=16,
        )
    )

    assert result.root_actions
    assert np.isclose(float(result.root_strategy.sum()), 1.0)

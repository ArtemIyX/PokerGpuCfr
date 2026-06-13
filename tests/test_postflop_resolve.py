import numpy as np
import pytest

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
from pokergpu.runtime import (
    POSTFLOP_SOLVER_DEFAULT_SEED,
    POSTFLOP_SOLVER_VERSION,
    PostflopResolveSpec,
    PublicStateFingerprint,
    SolveCacheState,
    WarmStartState,
    resolve_postflop_hu,
    resolve_postflop_multi,
    resolve_postflop_threeway,
)


def test_postflop_resolver_returns_root_strategy() -> None:
    state = GameState(
        board=Board.from_str("AhKdTc"),
        players=(PlayerState(player=PlayerIndex(0)), 
                 PlayerState(player=PlayerIndex(1))),
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

    assert result.root_infoset_id == 0
    assert np.isclose(float(result.root_strategy.sum()), 1.0)
    assert result.root_action_ev_player0.shape == result.root_strategy.shape
    assert result.root_action_ev_player1.shape == result.root_strategy.shape
    assert result.iterations == 1
    assert result.elapsed_seconds >= 0.0
    assert result.node_count > 0
    assert result.leaf_count > 0


def test_postflop_resolver_root_ev_matches_action_ev_weighting() -> None:
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

    expected = float(
        np.sum(
            result.root_strategy[: result.root_action_ev_player0.shape[0]]
            * result.root_action_ev_player0,
            dtype=np.float64,
        )
    )
    assert np.isclose(result.root_ev_player0, expected / 100.0)
    assert np.isclose(result.root_ev_player1, -result.root_ev_player0)


def test_postflop_resolve_spec_has_stable_defaults() -> None:
    assert POSTFLOP_SOLVER_DEFAULT_SEED == 0
    assert POSTFLOP_SOLVER_VERSION == "mvp-postflop-v1"


def test_postflop_resolver_uses_cached_warm_start() -> None:
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
    cache = SolveCacheState()
    fingerprint = PublicStateFingerprint(
        variant="nlhe",
        street="flop",
        acting_player=0,
        pot=300,
        stacks=(1700, 1700),
        blinds=(50, 100),
        antes=(0, 0),
        board=("Ah", "Kd", "Tc"),
        action_history=(),
        action_abstraction_id="compact:v2|flop|oop",
        range_abstraction_id="private_hand_v1",
        subtree_depth_limit=1,
        evaluator_id="CpuStubLeafEvaluator",
        solver_version="1",
        player_count=2,
        active_players=(0, 1),
        canonical_board="AhKdTc",
        card_removal_version="1",
    )
    cache.store_warm_start(
        fingerprint.digest(),
        WarmStartState(
            regret=(1.0, -1.0),
            strategy_sum=(2.0, 2.0),
            source_key="seed",
            blend_alpha=1.0,
        ),
    )

    result = resolve_postflop_hu(
        PostflopResolveSpec(
            state=state,
            range_p0=RangeVector.uniform(),
        range_p1=RangeVector.uniform(),
        time_budget_sec=0.0,
        solver_version=POSTFLOP_SOLVER_VERSION,
        max_depth=1,
        max_nodes=16,
        cache_state=cache,
    )
    )

    assert result.iterations == 1
    assert cache.bundle.warm_start.stats()["entries"] == 2


def test_postflop_resolver_is_deterministic_for_same_seed() -> None:
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
    spec = PostflopResolveSpec(
        state=state,
        range_p0=RangeVector.uniform(),
        range_p1=RangeVector.uniform(),
        time_budget_sec=0.0,
        seed=123,
        solver_version=POSTFLOP_SOLVER_VERSION,
        max_depth=1,
        max_nodes=16,
    )

    result_a = resolve_postflop_hu(spec)
    result_b = resolve_postflop_hu(spec)

    assert np.allclose(result_a.root_strategy, result_b.root_strategy)
    assert np.allclose(result_a.root_action_ev_player0, result_b.root_action_ev_player0)
    assert np.allclose(result_a.root_action_ev_player1, result_b.root_action_ev_player1)
    assert result_a.root_infoset_id == result_b.root_infoset_id


def test_postflop_multiway_masks_each_player_range_independently() -> None:
    from pokergpu.runtime.postflop import _apply_root_ranges

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
    ranges = (
        RangeVector.uniform(),
        RangeVector.uniform(),
        RangeVector.uniform(),
    )

    masked = _apply_root_ranges(state, ranges)

    assert len(masked) == 3
    assert all(np.isclose(float(r.total_weight()), 1.0) for r in masked)


def test_multiway_postflop_rejects_preflop() -> None:
    state = GameState(
        board=Board(cards=()),
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
    spec = PostflopResolveSpec(
        state=state,
        range_p0=RangeVector.uniform(),
        range_p1=RangeVector.uniform(),
        range_p2=RangeVector.uniform(),
        time_budget_sec=0.0,
    )

    with pytest.raises(ValueError):
        resolve_postflop_threeway(spec)


def test_multiway_postflop_rejects_two_players_when_3way_path_requested() -> None:
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
    spec = PostflopResolveSpec(
        state=state,
        range_p0=RangeVector.uniform(),
        range_p1=RangeVector.uniform(),
        time_budget_sec=0.0,
    )

    with pytest.raises(ValueError):
        resolve_postflop_threeway(spec)


def test_multiway_postflop_builds_3_player_tree() -> None:
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
    result = resolve_postflop_multi(
        PostflopResolveSpec(
            state=state,
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            range_p2=RangeVector.uniform(),
            time_budget_sec=0.0,
            max_depth=1,
            max_nodes=16,
        ),
        max_player_count=3,
    )

    assert result.node_count > 0


def test_multiway_postflop_root_strategy_is_normalized() -> None:
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
    result = resolve_postflop_threeway(
        PostflopResolveSpec(
            state=state,
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            range_p2=RangeVector.uniform(),
            time_budget_sec=0.0,
            max_depth=1,
            max_nodes=16,
        )
    )

    assert np.isclose(float(result.root_strategy.sum()), 1.0)


def test_multiway_postflop_returns_three_root_evs() -> None:
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
    result = resolve_postflop_threeway(
        PostflopResolveSpec(
            state=state,
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            range_p2=RangeVector.uniform(),
            time_budget_sec=0.0,
            max_depth=1,
            max_nodes=16,
        )
    )

    assert result.root_ev.shape == (3,)


def test_multiway_postflop_respects_time_budget() -> None:
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
    result = resolve_postflop_threeway(
        PostflopResolveSpec(
            state=state,
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            range_p2=RangeVector.uniform(),
            time_budget_sec=0.0,
            max_depth=1,
            max_nodes=16,
        )
    )

    assert result.iterations == 1


def test_resolve_postflop_multi_returns_distribution_for_three_players() -> None:
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
    result = resolve_postflop_threeway(
        PostflopResolveSpec(
            state=state,
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            range_p2=RangeVector.uniform(),
            time_budget_sec=0.0,
            max_depth=1,
            max_nodes=16,
        )
    )

    assert np.isclose(float(result.root_strategy.sum()), 1.0)
    assert result.root_ev.shape == (3,)


def test_resolve_postflop_multi_falls_back_when_budget_is_zero() -> None:
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
    result = resolve_postflop_threeway(
        PostflopResolveSpec(
            state=state,
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            range_p2=RangeVector.uniform(),
            time_budget_sec=0.0,
            max_depth=1,
            max_nodes=16,
        )
    )

    assert result.iterations == 1


def test_resolve_postflop_multi_uses_warm_start_if_available() -> None:
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
    cache = SolveCacheState()
    resolve_postflop_threeway(
        PostflopResolveSpec(
            state=state,
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            range_p2=RangeVector.uniform(),
            time_budget_sec=0.0,
            max_depth=1,
            max_nodes=16,
            cache_state=cache,
        )
    )

    assert cache.stats()["warm_start"]["max_entries"] == 128


def test_postflop_threeway_resolver_returns_three_player_result() -> None:
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
    result = resolve_postflop_threeway(
        PostflopResolveSpec(
            state=state,
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            range_p2=RangeVector.uniform(),
            time_budget_sec=0.0,
            max_depth=1,
            max_nodes=16,
        )
    )

    assert result.root_ev.shape == (3,)
    assert np.isclose(float(result.root_ev.sum()), float(result.root_ev[0] + result.root_ev[1] + result.root_ev[2]))

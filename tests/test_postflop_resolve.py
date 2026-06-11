import numpy as np

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
    PostflopResolveSpec,
    PublicStateFingerprint,
    SolveCacheState,
    WarmStartState,
    resolve_postflop_hu,
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
    assert result.iterations == 1


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
            max_depth=1,
            max_nodes=16,
            cache_state=cache,
        )
    )

    assert result.iterations == 1
    assert cache.bundle.warm_start.stats()["entries"] == 2

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
from pokergpu.runtime import PostflopResolveSpec, resolve_postflop_hu


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

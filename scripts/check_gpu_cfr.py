from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
from pokergpu.eval import EvalDeviceConfig
from pokergpu.eval.gpu_stub import GpuStubLeafEvaluator
from pokergpu.runtime import PostflopResolveSpec, resolve_postflop_gpu


def make_state() -> GameState:
    return GameState(
        board=Board.from_str("KhQs5hJs"),
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(200)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(2000)),
                PlayerStack(player=PlayerIndex(1), stack=chips(2000)),
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


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return 1

    result = resolve_postflop_gpu(
        PostflopResolveSpec(
            state=make_state(),
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            time_budget_sec=0.0,
            max_depth=6,
            max_nodes=2048,
        ),
        evaluator=GpuStubLeafEvaluator(EvalDeviceConfig(mode="cuda")),
    )

    print(f"iterations={result.iterations}")
    print(f"node_count={result.node_count}")
    print(f"leaf_count={result.leaf_count}")
    print(f"root_ev_p0={result.root_ev_player0:.6f}")
    print(f"root_ev_p1={result.root_ev_player1:.6f}")
    print(f"root_strategy={np.array2string(result.root_strategy, precision=4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

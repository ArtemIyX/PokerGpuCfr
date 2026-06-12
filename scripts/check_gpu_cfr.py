from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pokergpu.cfr import train_toy_mccfr, toy_expected_value
from pokergpu.eval import EvalDeviceConfig, LeafFeatureBatch
from pokergpu.eval.gpu_stub import GpuStubLeafEvaluator


def main() -> int:
    result = train_toy_mccfr(iterations=128, seed=7)
    print(f"expected_ev={toy_expected_value():.6f}")
    print(f"learned_ev={result.expected_value_p0:.6f}")
    print(f"regrets={np.array2string(result.store.regrets, precision=4)}")
    print(f"strategy_sums={np.array2string(result.store.strategy_sums, precision=4)}")
    if torch.cuda.is_available():
        evaluator = GpuStubLeafEvaluator(EvalDeviceConfig(mode="cuda"))
        batch = LeafFeatureBatch(
            node_indices=(0,),
            player_to_act=np.array([0], dtype=np.int32),
            street=np.array([3], dtype=np.int32),
            pot=np.array([200.0], dtype=np.float32),
            stack_p0=np.array([1000.0], dtype=np.float32),
            stack_p1=np.array([1000.0], dtype=np.float32),
            board_size=np.array([4], dtype=np.int32),
            reach_p0=np.array([1.0], dtype=np.float32),
            reach_p1=np.array([1.0], dtype=np.float32),
            reach_p2=np.array([0.0], dtype=np.float32),
            is_terminal=np.array([False], dtype=np.bool_),
            is_frontier=np.array([True], dtype=np.bool_),
            infoset_id=np.array([0], dtype=np.int32),
        )
        leaf = evaluator.evaluate(batch)
        print(f"gpu_stub_ev={float(leaf.ev_player0[0]):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import cProfile
import pstats
import sys
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pokergpu.cfr.stage1 import propagate_forward
from pokergpu.cfr.stage2 import aggregate_prob_sum
from pokergpu.cfr.stage3 import compute_opponent_reach
from pokergpu.cfr.solver.tree import make_toy_pipeline_tree


def main() -> None:
    tree = make_toy_pipeline_tree()
    forward = propagate_forward(tree)
    aggregate = aggregate_prob_sum(tree, forward)

    profiler = cProfile.Profile()
    profiler.enable()
    opponent = compute_opponent_reach(tree, aggregate, max_workers=16)
    profiler.disable()

    stream = StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(80)
    print(stream.getvalue())
    print("Stage 3 output summary")
    print(f"  infoset_opponent_reach: {opponent.infoset_opponent_reach}")
    print(f"  node_opponent_share: {opponent.node_opponent_share}")


if __name__ == "__main__":
    main()

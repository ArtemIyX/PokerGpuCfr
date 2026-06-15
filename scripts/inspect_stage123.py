from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pokergpu.cfr.stage1 import propagate_forward
from pokergpu.cfr.stage2 import aggregate_prob_sum
from pokergpu.cfr.stage3 import compute_opponent_reach
from pokergpu.cfr.solver import make_toy_pipeline_tree


def main() -> None:
    tree = make_toy_pipeline_tree()
    forward = propagate_forward(tree)
    aggregate = aggregate_prob_sum(tree, forward)
    opponent = compute_opponent_reach(tree, aggregate)

    print("Stage 1: forward reach")
    print(f"  node_reach: {forward.node_reach}")
    print(f"  infoset_reach: {forward.infoset_reach}")
    print(f"  action_reach: {forward.action_reach}")
    print()

    print("Stage 2: aggregate probabilities")
    print(f"  node_aggregate.reach: {aggregate.node_aggregate.reach}")
    print(f"  leaf_node_ids: {aggregate.leaf_node_ids}")
    print(f"  leaf_reach_sum: {aggregate.leaf_reach_sum}")
    print()

    print("Stage 3: opponent reach")
    print(f"  infoset_opponent_reach: {opponent.infoset_opponent_reach}")
    print(f"  infoset_card_opponent_reach[0][:8]: {opponent.infoset_card_opponent_reach[0][:8]}")
    print(f"  node_opponent_reach: {opponent.node_opponent_reach}")
    print(f"  node_opponent_share: {opponent.node_opponent_share}")


if __name__ == "__main__":
    main()

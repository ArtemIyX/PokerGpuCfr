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
    tree = _make_heavy_tree(replicas=512)
    forward = propagate_forward(tree)
    aggregate = aggregate_prob_sum(tree, forward)

    # Warm up Numba so the profile reflects steady-state execution only.
    compute_opponent_reach(tree, aggregate, max_workers=16)

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


def _make_heavy_tree(*, replicas: int):
    base = make_toy_pipeline_tree()
    if replicas <= 0:
        raise ValueError("replicas must be positive")

    node_types = []
    first_child = []
    child_count = []
    children = []
    infoset_ids = []
    terminal_payoffs = []
    base_node_count = base.node_count
    base_infoset_count = max((int(infoset_id) for infoset_id in base.infoset_ids if infoset_id is not None), default=-1) + 1

    for replica_index in range(replicas):
        node_offset = replica_index * base_node_count
        infoset_offset = replica_index * base_infoset_count
        child_offset = len(children)
        for node_index in range(base_node_count):
            node_types.append(base.node_types[node_index])
            first_child.append(child_offset + base.first_child[node_index])
            child_count.append(base.child_count[node_index])
            infoset_id = base.infoset_ids[node_index]
            infoset_ids.append(None if infoset_id is None else type(infoset_id)(int(infoset_id) + infoset_offset))
            terminal_payoffs.append(base.terminal_payoffs[node_index])
        for link in base.children:
            children.append(type(link)(child=type(link.child)(int(link.child) + node_offset), chance_prob=link.chance_prob))

    return type(base)(
        node_types=tuple(node_types),
        first_child=tuple(first_child),
        child_count=tuple(child_count),
        children=tuple(children),
        infoset_ids=tuple(infoset_ids),
        terminal_payoffs=tuple(terminal_payoffs),
    )


if __name__ == "__main__":
    main()

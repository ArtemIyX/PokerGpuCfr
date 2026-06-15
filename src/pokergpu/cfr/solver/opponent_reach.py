from __future__ import annotations

from pokergpu.cfr.stage2 import AggregateProbSumResult
from pokergpu.cfr.stage3 import OpponentReachResult, compute_opponent_reach
from pokergpu.tree.public_tree import PublicTree


def propagate_opponent_reach(
    tree: PublicTree,
    aggregate: AggregateProbSumResult,
    *,
    max_workers: int | None = None,
) -> OpponentReachResult:
    assert tree.node_count > 0, "public tree cannot be empty"
    return compute_opponent_reach(tree, aggregate, max_workers=max_workers)

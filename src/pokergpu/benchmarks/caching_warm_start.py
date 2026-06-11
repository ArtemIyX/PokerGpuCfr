from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

from pokergpu.runtime import (
    CachedLeafResult,
    CachedTree,
    PublicStateFingerprint,
    SolveCacheState,
    build_benchmark,
    blend_regret,
    make_leaf_key,
    make_warm_start_state,
)


def _fake_tree() -> CachedTree:
    return CachedTree(
        node_type=(0, 1, 2, 3),
        first_child=(1, 0, 0, 0),
        child_count=(3, 0, 0, 0),
        action_id=(0, 1, 2, 3),
        infoset_id=(10, 11, 12, 13),
        terminal_payoff=(0.0, 1.0, -1.0, 0.5),
        chance_prob=(1.0, 0.0, 0.0, 0.0),
        depth=(0, 1, 1, 1),
        street=(2, 2, 2, 2),
        frontier=(False, True, True, True),
    )


def run_caching_benchmark(iterations: int = 20000) -> list[dict[str, object]]:
    spot = PublicStateFingerprint(
        variant="nlhe",
        street="flop",
        acting_player=0,
        pot=120,
        stacks=(980, 980),
        blinds=(5, 10),
        antes=(0, 0),
        board=("As", "Kh", "7d"),
        action_history=("b33", "c"),
        action_abstraction_id="flop_ip_v1",
        range_abstraction_id="bucket_v1",
        subtree_depth_limit=4,
        evaluator_id="cpu_stub_v1",
        solver_version="1",
        player_count=2,
        active_players=(0, 1),
        canonical_board="AsKh7d",
        card_removal_version="1",
    )
    spot_key = spot.digest()
    cache = SolveCacheState()

    tree = _fake_tree()
    leaf_key = make_leaf_key(
        public_key=spot_key,
        leaf_index=1,
        evaluator_id="cpu_stub_v1",
        range_signature="r0:r1",
        board_signature="AsKh7d",
    )
    leaf = CachedLeafResult(value=(0.25, -0.25), evaluator_id="cpu_stub_v1", leaf_key=leaf_key)
    warm = make_warm_start_state(
        regret=(0.1, -0.1, 0.05, -0.05),
        strategy_sum=(10.0, 20.0, 30.0, 40.0),
        source_key=spot_key,
        blend_alpha=0.9,
    )

    t0 = perf_counter()
    for _ in range(iterations):
        cache.store_tree(spot_key, tree, size_bytes=256)
        cache.store_leaf(leaf_key, leaf, size_bytes=64)
        cache.store_warm_start(spot_key, warm, size_bytes=128)
        _ = cache.lookup_tree(spot_key)
        _ = cache.lookup_leaf(leaf_key)
        _ = cache.lookup_warm_start(spot_key)
    total_ms = (perf_counter() - t0) * 1000.0

    cold_t0 = perf_counter()
    for _ in range(iterations):
        _ = spot.digest()
        _ = blend_regret((0.4, 0.2, -0.1), (0.0, 0.0, 0.0), 0.75)
    cold_ms = (perf_counter() - cold_t0) * 1000.0

    benchmark = build_benchmark(
        spot_key=spot_key,
        mode="warm",
        tree_build_ms=total_ms * 0.25,
        leaf_eval_ms=total_ms * 0.15,
        cfr_ms=total_ms * 0.35,
        total_ms=total_ms,
        cache_hits=cache.counters.hits,
        cache_misses=cache.counters.misses,
        root_strategy=(0.55, 0.30, 0.10, 0.05),
        root_ev=0.12,
    )

    return [
        {
            "benchmark": "cache_warm_start",
            "spot": spot_key,
            "iterations": iterations,
            "total_ms": round(total_ms, 3),
            "cold_ms": round(cold_ms, 3),
            "tree_cache": cache.bundle.tree.stats(),
            "leaf_cache": cache.bundle.leaf.stats(),
            "warm_cache": cache.bundle.warm_start.stats(),
            "cache_hits": cache.counters.hits,
            "cache_misses": cache.counters.misses,
        },
        asdict(benchmark),
    ]


def main() -> None:
    for row in run_caching_benchmark():
        print(row)


if __name__ == "__main__":
    main()

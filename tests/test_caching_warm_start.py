from pokergpu.runtime import (
    CachedLeafResult,
    CachedTree,
    LruCache,
    PublicStateFingerprint,
    SolveCacheState,
    blend_regret,
    build_benchmark,
    make_leaf_key,
    make_warm_start_state,
    normalize_sequence,
    stable_hash,
)


def test_public_state_key_is_deterministic() -> None:
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
    assert spot.digest() == spot.digest()


def test_cache_round_trip() -> None:
    cache = SolveCacheState()
    spot_key = "spot"
    tree = CachedTree(
        node_type=(0,),
        first_child=(0,),
        child_count=(0,),
        action_id=(0,),
        infoset_id=(1,),
        terminal_payoff=(0.0,),
        chance_prob=(1.0,),
        depth=(0,),
        street=(2,),
        frontier=(False,),
    )
    cache.store_tree(spot_key, tree)
    assert cache.lookup_tree(spot_key) == tree


def test_leaf_key_and_warm_state() -> None:
    leaf_key = make_leaf_key(
        public_key="spot",
        leaf_index=1,
        evaluator_id="cpu_stub_v1",
        range_signature="r0:r1",
        board_signature="AsKh7d",
    )
    leaf = CachedLeafResult(value=(0.25, -0.25), evaluator_id="cpu_stub_v1", leaf_key=leaf_key)
    warm = make_warm_start_state(
        regret=(0.1, 0.2),
        strategy_sum=(1.0, 2.0),
        source_key="spot",
        blend_alpha=0.9,
    )
    assert leaf.leaf_key == leaf_key
    assert warm.source_key == "spot"


def test_blend_regret_and_normalize() -> None:
    blended = blend_regret((1.0, 0.0), (0.0, 1.0), 0.5)
    assert blended == (0.5, 0.5)
    assert normalize_sequence((2.0, 2.0)) == (0.5, 0.5)


def test_benchmark_shape() -> None:
    benchmark = build_benchmark(
        spot_key="spot",
        mode="cold",
        tree_build_ms=1.0,
        leaf_eval_ms=2.0,
        cfr_ms=3.0,
        total_ms=6.0,
        cache_hits=1,
        cache_misses=2,
        root_strategy=(0.5, 0.5),
        root_ev=0.1,
    )
    data = benchmark.as_dict()
    assert data["spot"] == "spot"
    assert data["mode"] == "cold"
    assert data["cache_hits"] == 1
    assert data["cache_misses"] == 2
    assert data["root_ev"] == 0.1


def test_lru_cache_replaces_old_entries() -> None:
    cache = LruCache(max_entries=1)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") is None
    assert cache.get("b") == 2

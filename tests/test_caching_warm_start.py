import pytest

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
)
from pokergpu.cfr import InfosetLayout
from pokergpu.tree import NodeType
from pokergpu.runtime import gpu_postflop
from pokergpu.runtime.cache import PackedGpuSubtree


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


def test_public_state_key_uses_canonical_board_identity() -> None:
    spot_a = PublicStateFingerprint(
        variant="nlhe",
        street="flop",
        acting_player=0,
        pot=120,
        stacks=(980, 980),
        blinds=(5, 10),
        antes=(0, 0),
        board=("Ah", "Kh", "Qd"),
        action_history=("b33", "c"),
        action_abstraction_id="flop_ip_v1",
        range_abstraction_id="bucket_v1",
        subtree_depth_limit=4,
        evaluator_id="cpu_stub_v1",
        solver_version="1",
        player_count=2,
        active_players=(0, 1),
        canonical_board="AcKcQd",
        card_removal_version="1",
    )
    spot_b = PublicStateFingerprint(
        variant="nlhe",
        street="flop",
        acting_player=0,
        pot=120,
        stacks=(980, 980),
        blinds=(5, 10),
        antes=(0, 0),
        board=("Ac", "Kc", "Qd"),
        action_history=("b33", "c"),
        action_abstraction_id="flop_ip_v1",
        range_abstraction_id="bucket_v1",
        subtree_depth_limit=4,
        evaluator_id="cpu_stub_v1",
        solver_version="1",
        player_count=2,
        active_players=(0, 1),
        canonical_board="AcKcQd",
        card_removal_version="1",
    )

    assert spot_a.digest() == spot_b.digest()


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
    leaf = CachedLeafResult(value=(0.25, -0.25), 
                            evaluator_id="cpu_stub_v1", leaf_key=leaf_key)
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
    cache: LruCache[int] = LruCache(max_entries=1)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") is None
    assert cache.get("b") == 2


def test_packed_subtree_cache_reuses_compiled_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"build": 0, "compile": 0}

    class DummyTree:
        def __init__(self) -> None:
            from types import SimpleNamespace

            self.tree = SimpleNamespace(
                node_count=1,
                node_types=(NodeType.PLAYER0,),
                is_frontier=(False,),
                first_child=(0,),
                child_count=(0,),
                children=(),
                infoset_ids=(0,),
                terminal_payoffs=(None,),
            )
            self.actions_by_node = ((object(),),)
            self.node_states = ()

    def fake_build_public_tree(*args: object, **kwargs: object) -> object:
        calls["build"] += 1
        return DummyTree()

    def fake_build_tree_levels(tree: object) -> object:
        return type("L", (), {"forward_levels": ((),), "backward_levels": ((),)})()

    class DummyLayout:
        action_counts = (1,)
        offsets = (0,)
        total_actions = 1
        infoset_count = 1

        @classmethod
        def from_action_counts(cls, counts: tuple[int, ...]) -> "DummyLayout":
            return cls()

    def fake_plan(*args: object, **kwargs: object) -> object:
        return type(
            "P",
            (),
            {
                "forward_levels": (),
                "backward_levels": (),
                "edge_parent": None,
                "edge_child": None,
                "edge_node_type": None,
                "edge_infoset": None,
                "edge_action_slot": None,
                "edge_chance_prob": None,
                "node_infoset": None,
                "node_type": None,
                "frontier_nodes": None,
                "frontier_leaf_batch": None,
                "root_child_nodes": None,
                "root_child_parent_infoset": 0,
                "action_counts": None,
                "action_offsets": None,
            },
        )()

    def fake_compile_packed_subtree(tree: object, *, spot_key: str, device: str) -> PackedGpuSubtree:
        calls["compile"] += 1
        return PackedGpuSubtree(
            spot_key=spot_key,
            node_type=(),
            is_frontier=(),
            first_child=(),
            child_count=(),
            children=(),
            infoset_ids=(),
            terminal_payoffs=(),
            node_depth=(),
            street=(),
            action_slot=(),
            chance_prob=(),
            frontier_nodes=(),
            leaf_feature_batch=type("B", (), {"size": 0})(),
            action_infoset_index=(),
            action_slot_index=(),
            root_node=0,
            root_infoset=0,
            node_count=1,
            edge_count=0,
            infoset_count=1,
            leaf_count=0,
            max_actions=1,
            device=device,
            tree_version="1",
        )

    monkeypatch.setattr(gpu_postflop, "build_public_tree", fake_build_public_tree)
    monkeypatch.setattr(gpu_postflop, "compile_packed_subtree", fake_compile_packed_subtree)
    monkeypatch.setattr(gpu_postflop, "build_tree_levels", fake_build_tree_levels)
    monkeypatch.setattr(InfosetLayout, "from_action_counts", DummyLayout.from_action_counts)
    monkeypatch.setattr(gpu_postflop, "_build_batched_gpu_plan", fake_plan)
    gpu_postflop._GPU_PLAN_CACHE.clear()

    from types import SimpleNamespace

    spec: object = SimpleNamespace(
        state=SimpleNamespace(
            player_count=2,
            current_street=SimpleNamespace(value="flop"),
            betting_round=SimpleNamespace(
                pot=SimpleNamespace(amount=300),
                to_act=0,
                blinds=SimpleNamespace(big_blind=100),
            ),
        ),
        max_depth=1,
        max_nodes=8,
        min_reach_prob=0.0,
        solver_version="test",
    )
    gpu_postflop._prepare_gpu_solve(spec)  # type: ignore[arg-type]
    gpu_postflop._prepare_gpu_solve(spec)  # type: ignore[arg-type]

    assert calls["build"] == 1
    assert calls["compile"] == 1

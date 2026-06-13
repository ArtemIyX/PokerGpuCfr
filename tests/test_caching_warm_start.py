import pytest
import torch
from types import SimpleNamespace
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
from pokergpu.core.state import GameState, PlayerState
from pokergpu.cfr import InfosetLayout
from pokergpu.core.board import Board
from pokergpu.runtime import (
    CachedLeafResult,
    CachedTree,
    LruCache,
    PublicStateFingerprint,
    SolveCacheState,
    blend_regret,
    build_benchmark,
    gpu_postflop,
    make_leaf_key,
    make_warm_start_state,
    normalize_sequence,
    PostflopResolveSpec,
)
from pokergpu.runtime.cache import PackedGpuSubtree
from pokergpu.tree import NodeType


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
    class DummyLeafBatch:
        size = 0

    class DummyTemplate:
        tree_key = "spot"

        def as_public_tree(self, children: tuple[object, ...]) -> object:
            return self

    class DummyTree:
        def __init__(self) -> None:
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
            self.template = DummyTemplate()
            self.actions_by_node = ((object(),),)
            self.node_states = ()
            self.action_abstraction_id = "runtime:v1|flop|oop"
            self.canonical_board_key = "AsKh7d"
            self.player_count = 2
            self.active_players = (0, 1)

    def fake_build_public_tree(*args: object, **kwargs: object) -> DummyTree:
        calls["build"] += 1
        return DummyTree()

    def fake_build_tree_levels(tree: object) -> object:
        return SimpleNamespace(forward_levels=((),), backward_levels=((),))

    def fake_compile_packed_subtree(tree: object, *, spot_key: str, device: str) -> PackedGpuSubtree:
        calls["compile"] += 1
        return PackedGpuSubtree(
            spot_key=spot_key,
            node_type=torch.zeros(1, dtype=torch.int64),
            is_frontier=torch.zeros(1, dtype=torch.bool),
            first_child=torch.zeros(1, dtype=torch.int64),
            child_count=torch.zeros(1, dtype=torch.int64),
            children=torch.zeros(0, dtype=torch.int64),
            infoset_ids=torch.zeros(1, dtype=torch.int64),
            terminal_payoffs=torch.zeros(1, dtype=torch.float32),
            node_depth=torch.zeros(1, dtype=torch.int64),
            street=torch.zeros(1, dtype=torch.int64),
            action_slot=torch.zeros(0, dtype=torch.int64),
            chance_prob=torch.zeros(0, dtype=torch.float32),
            frontier_nodes=torch.zeros(0, dtype=torch.int64),
            leaf_feature_batch=DummyLeafBatch(),
            action_infoset_index=torch.zeros(0, dtype=torch.int64),
            action_slot_index=torch.zeros(0, dtype=torch.int64),
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

    def fake_layout_from_action_counts(counts: tuple[int, ...]) -> object:
        return SimpleNamespace(action_counts=(1,), offsets=(0,), total_actions=1, infoset_count=1)

    def fake_plan(*args: object, **kwargs: object) -> object:
        return SimpleNamespace(
            forward_levels=(),
            backward_levels=(),
            edge_parent=torch.zeros(0, dtype=torch.int64),
            edge_child=torch.zeros(0, dtype=torch.int64),
            edge_node_type=torch.zeros(0, dtype=torch.int64),
            edge_infoset=torch.zeros(0, dtype=torch.int64),
            edge_action_slot=torch.zeros(0, dtype=torch.int64),
            edge_chance_prob=torch.zeros(0, dtype=torch.float32),
            node_infoset=torch.zeros(1, dtype=torch.int64),
            node_type=torch.zeros(1, dtype=torch.int64),
            frontier_nodes=torch.zeros(0, dtype=torch.int64),
            frontier_leaf_batch=DummyLeafBatch(),
            root_child_nodes=torch.zeros(0, dtype=torch.int64),
            root_child_parent_infoset=0,
            action_counts=torch.zeros(1, dtype=torch.int64),
            action_offsets=torch.zeros(1, dtype=torch.int64),
        )

    def fake_run_gpu_solve(packed: object, evaluator: object) -> object:
        return SimpleNamespace(
            packed=packed,
            iterations=1,
            elapsed_seconds=0.0,
            node_count=1,
            leaf_count=0,
            root_strategy=np.asarray([1.0], dtype=np.float32),
            root_action_ev_player0=np.asarray([0.0], dtype=np.float32),
            root_action_ev_player1=np.asarray([0.0], dtype=np.float32),
            root_ev_player0=0.0,
            root_ev_player1=0.0,
            gpu_backward_p0=np.asarray([0.0], dtype=np.float32),
            gpu_backward_p1=np.asarray([0.0], dtype=np.float32),
            cpu_backward_p0=np.asarray([0.0], dtype=np.float32),
            cpu_backward_p1=np.asarray([0.0], dtype=np.float32),
        )

    monkeypatch.setattr(gpu_postflop, "build_public_tree", fake_build_public_tree)
    monkeypatch.setattr(gpu_postflop, "compile_packed_subtree", fake_compile_packed_subtree)
    monkeypatch.setattr(gpu_postflop, "build_tree_levels", fake_build_tree_levels)
    monkeypatch.setattr(InfosetLayout, "from_action_counts", fake_layout_from_action_counts)
    monkeypatch.setattr(gpu_postflop, "_build_batched_gpu_plan", fake_plan)
    monkeypatch.setattr(gpu_postflop, "_run_gpu_solve", fake_run_gpu_solve)
    gpu_postflop._GPU_PLAN_CACHE.clear()

    state = GameState(
        board=Board.from_str("AsKh7d"),
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(300)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(980)),
                PlayerStack(player=PlayerIndex(1), stack=chips(980)),
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
    spec = PostflopResolveSpec(
        state=state,
        range_p0=RangeVector.uniform(),
        range_p1=RangeVector.uniform(),
        time_budget_sec=0.0,
        cache_state=SolveCacheState(),
        max_depth=1,
        max_nodes=8,
    )

    gpu_postflop._prepare_gpu_solve(spec)
    gpu_postflop._prepare_gpu_solve(spec)

    assert calls["build"] == 2
    assert calls["compile"] == 1

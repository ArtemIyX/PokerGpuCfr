from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cache import (
    CacheBundle,
    CachedLeafResult,
    CachedTree,
    PackedGpuSubtree,
    PublicStateKey,
    SolveBenchmark,
    WarmStartState,
    entropy_from_probs,
)


@dataclass(frozen=True, slots=True)
class PublicStateFingerprint:
    variant: str
    street: str
    acting_player: int
    pot: int
    stacks: tuple[int, ...]
    blinds: tuple[int, ...]
    antes: tuple[int, ...]
    board: tuple[str, ...]
    action_history: tuple[str, ...]
    action_abstraction_id: str
    range_abstraction_id: str
    subtree_depth_limit: int
    evaluator_id: str
    solver_version: str
    player_count: int
    active_players: tuple[int, ...]
    canonical_board: str = ""
    card_removal_version: str = ""

    def key(self) -> PublicStateKey:
        return PublicStateKey(
            variant=self.variant,
            street=self.street,
            acting_player=self.acting_player,
            pot=self.pot,
            stacks=self.stacks,
            blinds=self.blinds,
            antes=self.antes,
            board=self.board,
            action_history=self.action_history,
            action_abstraction_id=self.action_abstraction_id,
            range_abstraction_id=self.range_abstraction_id,
            subtree_depth_limit=self.subtree_depth_limit,
            evaluator_id=self.evaluator_id,
            solver_version=self.solver_version,
            player_count=self.player_count,
            active_players=self.active_players,
            canonical_board=self.canonical_board,
            card_removal_version=self.card_removal_version,
        )

    def digest(self) -> str:
        return self.key().digest()


@dataclass(slots=True)
class CacheHitMiss:
    hits: int = 0
    misses: int = 0

    def hit(self) -> None:
        self.hits += 1

    def miss(self) -> None:
        self.misses += 1

    def total(self) -> int:
        return self.hits + self.misses


@dataclass(slots=True)
class SolveCacheState:
    bundle: CacheBundle = field(default_factory=CacheBundle)
    counters: CacheHitMiss = field(default_factory=CacheHitMiss)

    def lookup_tree(self, key: str) -> CachedTree | None:
        tree = self.bundle.tree.get(key)
        if tree is None:
            self.counters.miss()
        else:
            self.counters.hit()
        return tree

    def store_tree(self, key: str, tree: CachedTree, size_bytes: int = 0) -> None:
        self.bundle.tree.put(key, tree, size_bytes=size_bytes)

    def lookup_packed_subtree(self, key: str) -> PackedGpuSubtree | None:
        packed = self.bundle.packed_subtree.get(key)
        if packed is None:
            self.counters.miss()
        else:
            self.counters.hit()
        return packed

    def store_packed_subtree(
        self,
        key: str,
        packed: PackedGpuSubtree,
        size_bytes: int = 0,
    ) -> None:
        self.bundle.packed_subtree.put(key, packed, size_bytes=size_bytes)

    def lookup_leaf(self, key: str) -> CachedLeafResult | None:
        leaf = self.bundle.leaf.get(key)
        if leaf is None:
            self.counters.miss()
        else:
            self.counters.hit()
        return leaf

    def store_leaf(self, key: str, leaf: CachedLeafResult, size_bytes: int = 0) -> None:
        self.bundle.leaf.put(key, leaf, size_bytes=size_bytes)

    def lookup_warm_start(self, key: str) -> WarmStartState | None:
        warm = self.bundle.warm_start.get(key)
        if warm is None:
            self.counters.miss()
        else:
            self.counters.hit()
        return warm

    def store_warm_start(self, 
                         key: str, 
                         warm: WarmStartState, 
                         size_bytes: int = 0) -> None:
        self.bundle.warm_start.put(key, warm, size_bytes=size_bytes)

    def stats(self) -> dict[str, Any]:
        return {
            "tree": self.bundle.tree.stats(),
            "packed_subtree": self.bundle.packed_subtree.stats(),
            "leaf": self.bundle.leaf.stats(),
            "warm_start": self.bundle.warm_start.stats(),
            "hits": self.counters.hits,
            "misses": self.counters.misses,
        }


def normalize_sequence(values: tuple[float, ...]) -> tuple[float, ...]:
    total = sum(values)
    if total <= 0.0:
        return values
    return tuple(v / total for v in values)


def blend_regret(
    cached_regret: tuple[float, ...],
    fresh_prior: tuple[float, ...],
    alpha: float,
) -> tuple[float, ...]:
    if len(cached_regret) != len(fresh_prior):
        raise ValueError("regret vectors must have the same length")
    alpha = max(0.0, min(1.0, alpha))
    beta = 1.0 - alpha
    return tuple((alpha * c) + (beta * f) for c, f in 
                 zip(cached_regret, fresh_prior, strict= False))


def build_benchmark(
    *,
    spot_key: str,
    mode: str,
    tree_build_ms: float,
    leaf_eval_ms: float,
    cfr_ms: float,
    total_ms: float,
    cache_hits: int,
    cache_misses: int,
    root_strategy: tuple[float, ...],
    root_ev: float,
) -> SolveBenchmark:
    return SolveBenchmark(
        spot_key=spot_key,
        mode=mode,
        tree_build_ms=tree_build_ms,
        leaf_eval_ms=leaf_eval_ms,
        cfr_ms=cfr_ms,
        total_ms=total_ms,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        root_ev=root_ev,
        root_action_entropy=entropy_from_probs(root_strategy),
    )


def make_leaf_key(
    *,
    public_key: str,
    leaf_index: int,
    evaluator_id: str,
    range_signature: str = "",
    board_signature: str = "",
) -> str:
    return "|".join(
        (
            public_key,
            str(leaf_index),
            evaluator_id,
            range_signature,
            board_signature,
        )
    )


def make_warm_start_state(
    *,
    regret: tuple[float, ...],
    strategy_sum: tuple[float, ...] = (),
    source_key: str = "",
    blend_alpha: float = 1.0,
) -> WarmStartState:
    return WarmStartState(
        regret=regret,
        strategy_sum=strategy_sum,
        source_key=source_key,
        blend_alpha=blend_alpha,
    )

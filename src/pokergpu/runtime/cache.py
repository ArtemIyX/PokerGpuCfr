from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from hashlib import blake2b
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = Any  # type: ignore[assignment]


def _stable_bytes(value: Any) -> bytes:
    if value is None:
        return b"null"
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, bool):
        return b"1" if value else b"0"
    if isinstance(value, int):
        return str(value).encode("ascii")
    if isinstance(value, float):
        return repr(value).encode("ascii")
    if isinstance(value, (tuple, list)):
        return b"[" + b"|".join(_stable_bytes(item) for item in value) + b"]"
    if isinstance(value, dict):
        parts = []
        for key in sorted(value):
            parts.append(_stable_bytes(key) + b":" + _stable_bytes(value[key]))
        return b"{" + b"|".join(parts) + b"}"
    return repr(value).encode("utf-8")


def stable_hash(*parts: Any, digest_size: int = 16) -> str:
    h = blake2b(digest_size=digest_size)
    for part in parts:
        chunk = _stable_bytes(part)
        h.update(len(chunk).to_bytes(4, "big"))
        h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class PublicStateKey:
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
    schema_version: str = "1"

    def digest(self) -> str:
        board_key = self.canonical_board or self.board
        return stable_hash(
            self.schema_version,
            self.variant,
            self.street,
            self.acting_player,
            self.pot,
            self.stacks,
            self.blinds,
            self.antes,
            board_key,
            self.action_history,
            self.action_abstraction_id,
            self.range_abstraction_id,
            self.subtree_depth_limit,
            self.evaluator_id,
            self.solver_version,
            self.player_count,
            self.active_players,
            self.canonical_board,
            self.card_removal_version,
        )


@dataclass(slots=True)
class CacheEntry[T]:
    key: str
    value: T
    size_bytes: int = 0
    hits: int = 0
    created_tick: int = 0
    last_used_tick: int = 0


class LruCache[T]:
    def __init__(self, max_entries: int = 128, max_bytes: int = 0) -> None:
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._bytes = 0
        self._tick = 0

    def get(self, key: str) -> T | None:
        entry = self._entries.get(key)
        self._tick += 1
        if entry is None:
            return None
        entry.hits += 1
        entry.last_used_tick = self._tick
        self._entries.move_to_end(key)
        return entry.value

    def put(self, key: str, value: T, size_bytes: int = 0) -> None:
        self._tick += 1
        if key in self._entries:
            self._bytes -= self._entries[key].size_bytes
            del self._entries[key]
        entry = CacheEntry(
            key=key,
            value=value,
            size_bytes=size_bytes,
            created_tick=self._tick,
            last_used_tick=self._tick,
        )
        self._entries[key] = entry
        self._bytes += size_bytes
        self._evict()

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._entries),
            "bytes": self._bytes,
            "max_entries": self.max_entries,
            "max_bytes": self.max_bytes,
        }

    def _evict(self) -> None:
        while self._entries and (
            (self.max_entries > 0 and len(self._entries) > self.max_entries)
            or (self.max_bytes > 0 and self._bytes > self.max_bytes)
        ):
            _, entry = self._entries.popitem(last=False)
            self._bytes -= entry.size_bytes


@dataclass(slots=True)
class CachedTree:
    node_type: tuple[int, ...]
    first_child: tuple[int, ...]
    child_count: tuple[int, ...]
    action_id: tuple[int, ...]
    infoset_id: tuple[int, ...]
    terminal_payoff: tuple[float, ...]
    chance_prob: tuple[float, ...]
    depth: tuple[int, ...]
    street: tuple[int, ...]
    frontier: tuple[bool, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class CachedLeafResult:
    value: tuple[float, ...]
    evaluator_id: str
    leaf_key: str


@dataclass(slots=True)
class PackedGpuSubtree:
    spot_key: str
    node_type: Any
    is_frontier: Any
    first_child: Any
    child_count: Any
    children: Any
    infoset_ids: Any
    terminal_payoffs: Any
    node_depth: Any
    street: Any
    action_slot: Any
    chance_prob: Any
    frontier_nodes: Any
    chance_child_nodes: Any
    chance_child_valid_mask: Any
    chance_child_safe_nodes: Any
    leaf_feature_batch: Any
    chance_leaf_feature_batch: Any
    leaf_feature_tensors: Any
    action_infoset_index: Any
    action_slot_index: Any
    legal_action_mask: Any
    card_removal_mask: Any
    root_node: int = 0
    root_infoset: int = -1
    node_count: int = 0
    edge_count: int = 0
    infoset_count: int = 0
    leaf_count: int = 0
    max_actions: int = 0
    device: str = "cpu"
    tree_version: str = "1"

    def __post_init__(self) -> None:
        if self.node_count < 0 or self.edge_count < 0 or self.infoset_count < 0:
            raise ValueError("packed subtree counts must be non-negative")
        if self.leaf_count < 0 or self.max_actions < 0:
            raise ValueError("packed subtree sizes must be non-negative")
        if self.root_node < 0:
            raise ValueError("root node must be non-negative")
        if self.node_count and self.root_node >= self.node_count:
            raise ValueError("root node must be within node range")


@dataclass(slots=True)
class PackedGpuSolveState:
    packed: PackedGpuSubtree
    regrets: Any
    strategy_sums: Any
    strategy_table: Any
    node_range_p0: Any
    node_range_p1: Any
    node_range_p2: Any
    action_infoset_index: Any
    action_slot_index: Any
    action_offsets: Any
    action_counts: Any
    backward_p0: Any
    backward_p1: Any
    frontier_nodes: Any
    frontier_start: int
    frontier_count: int
    node_type: Any
    node_first_child: Any
    node_child_count: Any
    node_parent: Any
    node_infoset: Any
    node_street: Any
    node_depth: Any
    node_terminal_payoff: Any
    node_is_frontier: Any
    node_player: Any
    edge_parent: Any
    edge_child: Any
    edge_node_type: Any
    edge_infoset: Any
    edge_action_slot: Any
    edge_chance_prob: Any
    level_nodes: Any
    level_frontier_mask: Any
    level_player_mask: Any
    level_legal_action_mask: Any
    level_card_removal_mask: Any
    level_edge_start: Any
    level_edge_count: Any
    level_edge_src: Any
    level_edge_dst: Any
    level_edge_infoset: Any
    level_edge_slot: Any
    level_edge_kind: Any
    level_edge_prob: Any
    compact_level_edge_src: Any
    compact_level_edge_dst: Any
    compact_level_edge_infoset: Any
    compact_level_edge_slot: Any
    compact_level_edge_kind: Any
    compact_level_edge_prob: Any
    compact_level_edge_flat: Any
    compact_level_edge_src_chance: Any
    compact_level_edge_dst_chance: Any
    compact_level_edge_prob_chance: Any
    compact_level_edge_src_p0: Any
    compact_level_edge_dst_p0: Any
    compact_level_edge_infoset_p0: Any
    compact_level_edge_slot_p0: Any
    compact_level_edge_flat_p0: Any
    compact_level_edge_prob_p0: Any
    compact_level_edge_src_p1: Any
    compact_level_edge_dst_p1: Any
    compact_level_edge_infoset_p1: Any
    compact_level_edge_slot_p1: Any
    compact_level_edge_flat_p1: Any
    compact_level_edge_prob_p1: Any
    compact_forward_groups: Any
    compact_backward_groups: Any
    compact_backward_edge_src: Any
    compact_backward_edge_dst: Any
    compact_backward_edge_infoset: Any
    compact_backward_edge_slot: Any
    compact_backward_edge_kind: Any
    compact_backward_edge_prob: Any
    compact_backward_edge_flat: Any
    compact_backward_edge_src_chance: Any
    compact_backward_edge_dst_chance: Any
    compact_backward_edge_prob_chance: Any
    compact_backward_edge_src_p0: Any
    compact_backward_edge_dst_p0: Any
    compact_backward_edge_infoset_p0: Any
    compact_backward_edge_slot_p0: Any
    compact_backward_edge_flat_p0: Any
    compact_backward_edge_prob_p0: Any
    compact_backward_edge_src_p1: Any
    compact_backward_edge_dst_p1: Any
    compact_backward_edge_infoset_p1: Any
    compact_backward_edge_slot_p1: Any
    compact_backward_edge_flat_p1: Any
    compact_backward_edge_prob_p1: Any
    compact_forward_levels: Any
    compact_backward_levels: Any
    infoset_blocks: Any
    frontier_leaf_tensors: Any
    frontier_range_nodes: Any
    frontier_range_p0: Any
    frontier_range_p1: Any
    frontier_range_p2: Any
    root_branch_nodes: Any
    root_child_nodes: Any
    root_action_ev_buffer: Any
    root_child_parent_infoset: int = -1
    iteration: int = 0


@dataclass(slots=True)
class WarmStartState:
    regret: tuple[float, ...]
    strategy_sum: tuple[float, ...] = ()
    source_key: str = ""
    blend_alpha: float = 1.0

@dataclass(slots=True)
class CacheBundle:
    tree: LruCache[CachedTree] = field(
        default_factory=lambda: LruCache(max_entries=64)
    )
    packed_subtree: LruCache[PackedGpuSubtree] = field(
        default_factory=lambda: LruCache(max_entries=64)
    )
    leaf: LruCache[CachedLeafResult] = field(
        default_factory=lambda: LruCache(max_entries=4096)
    )
    warm_start: LruCache[WarmStartState] = field(
        default_factory=lambda: LruCache(max_entries=128)
    )
    tree_template: LruCache[Any] = field(
        default_factory=lambda: LruCache(max_entries=128)
    )


@dataclass(slots=True)
class SolveBenchmark:
    spot_key: str
    mode: str
    tree_build_ms: float = 0.0
    leaf_eval_ms: float = 0.0
    cfr_ms: float = 0.0
    total_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    root_ev: float = 0.0
    root_action_entropy: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "spot": self.spot_key,
            "mode": self.mode,
            "tree_build_ms": self.tree_build_ms,
            "leaf_eval_ms": self.leaf_eval_ms,
            "cfr_ms": self.cfr_ms,
            "total_ms": self.total_ms,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "root_ev": self.root_ev,
            "root_action_entropy": self.root_action_entropy,
        }


def entropy_from_probs(probs: tuple[float, ...]) -> float:
    import math

    total = 0.0
    for p in probs:
        if p > 0.0:
            total -= p * math.log(p)
    return total

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = object  # type: ignore[assignment]

from pokergpu.abstraction.hands import RangeVector
from pokergpu.cfr import InfosetLayout
from pokergpu.eval.types import LeafFeatureBatch
from pokergpu.runtime.cache import PackedGpuSolveState, PackedGpuSubtree
from pokergpu.runtime.postflop import PostflopResolveSpec
from pokergpu.tree.builder import BuiltPublicTree
from pokergpu.tree.public_tree import PublicTreeTemplate


@dataclass(frozen=True, slots=True)
class GpuSolveStats:
    iterations: int
    node_count: int
    leaf_count: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class PackedGpuSolve:
    spec: PostflopResolveSpec
    tree: BuiltPublicTree
    plan: "BatchedGpuPlan"
    layout: InfosetLayout
    root_infoset: int
    root_actions: tuple[str, ...]
    packed_subtree: PackedGpuSubtree
    gpu_state: PackedGpuSolveState | None = None
    root_ranges: tuple[RangeVector, ...] = ()


@dataclass(frozen=True, slots=True)
class CompactEdgeGroup:
    src: torch.Tensor
    dst: torch.Tensor
    infoset: torch.Tensor
    slot: torch.Tensor
    kind: torch.Tensor
    prob: torch.Tensor
    flat: torch.Tensor
    chance_src: torch.Tensor
    chance_dst: torch.Tensor
    chance_prob: torch.Tensor
    p0_src: torch.Tensor
    p0_dst: torch.Tensor
    p0_infoset: torch.Tensor
    p0_slot: torch.Tensor
    p0_flat: torch.Tensor
    p0_prob: torch.Tensor
    p1_src: torch.Tensor
    p1_dst: torch.Tensor
    p1_infoset: torch.Tensor
    p1_slot: torch.Tensor
    p1_flat: torch.Tensor
    p1_prob: torch.Tensor


@dataclass(frozen=True, slots=True)
class BatchedGpuSolveInput:
    spec: PostflopResolveSpec
    template: PublicTreeTemplate
    cache_key: str = ""


@dataclass(frozen=True, slots=True)
class BatchedGpuPlan:
    node_type: torch.Tensor
    node_first_child: torch.Tensor
    node_child_count: torch.Tensor
    node_parent: torch.Tensor
    node_infoset: torch.Tensor
    node_street: torch.Tensor
    node_depth: torch.Tensor
    node_terminal_payoff: torch.Tensor
    node_is_frontier: torch.Tensor
    forward_levels: tuple[dict[str, torch.Tensor], ...]
    backward_levels: tuple[dict[str, torch.Tensor], ...]
    level_frontier_mask: tuple[torch.Tensor, ...]
    level_player_mask: tuple[torch.Tensor, ...]
    edge_parent: torch.Tensor
    edge_child: torch.Tensor
    edge_node_type: torch.Tensor
    edge_infoset: torch.Tensor
    edge_action_slot: torch.Tensor
    edge_chance_prob: torch.Tensor
    level_nodes: tuple[torch.Tensor, ...]
    node_player: torch.Tensor
    level_edge_start: tuple[torch.Tensor, ...]
    level_edge_count: tuple[torch.Tensor, ...]
    level_edge_src: tuple[torch.Tensor, ...]
    level_edge_dst: tuple[torch.Tensor, ...]
    level_edge_infoset: tuple[torch.Tensor, ...]
    level_edge_slot: tuple[torch.Tensor, ...]
    level_edge_kind: tuple[torch.Tensor, ...]
    level_edge_prob: tuple[torch.Tensor, ...]
    compact_forward_groups: tuple[CompactEdgeGroup, ...]
    compact_backward_groups: tuple[CompactEdgeGroup, ...]
    compact_level_edge_src: tuple[torch.Tensor, ...]
    compact_level_edge_dst: tuple[torch.Tensor, ...]
    compact_level_edge_infoset: tuple[torch.Tensor, ...]
    compact_level_edge_slot: tuple[torch.Tensor, ...]
    compact_level_edge_kind: tuple[torch.Tensor, ...]
    compact_level_edge_prob: tuple[torch.Tensor, ...]
    compact_level_edge_src_chance: tuple[torch.Tensor, ...]
    compact_level_edge_dst_chance: tuple[torch.Tensor, ...]
    compact_level_edge_prob_chance: tuple[torch.Tensor, ...]
    compact_level_edge_src_p0: tuple[torch.Tensor, ...]
    compact_level_edge_dst_p0: tuple[torch.Tensor, ...]
    compact_level_edge_infoset_p0: tuple[torch.Tensor, ...]
    compact_level_edge_slot_p0: tuple[torch.Tensor, ...]
    compact_level_edge_flat_p0: tuple[torch.Tensor, ...]
    compact_level_edge_prob_p0: tuple[torch.Tensor, ...]
    compact_level_edge_src_p1: tuple[torch.Tensor, ...]
    compact_level_edge_dst_p1: tuple[torch.Tensor, ...]
    compact_level_edge_infoset_p1: tuple[torch.Tensor, ...]
    compact_level_edge_slot_p1: tuple[torch.Tensor, ...]
    compact_level_edge_flat_p1: tuple[torch.Tensor, ...]
    compact_level_edge_prob_p1: tuple[torch.Tensor, ...]
    compact_backward_edge_src: tuple[torch.Tensor, ...]
    compact_backward_edge_dst: tuple[torch.Tensor, ...]
    compact_backward_edge_infoset: tuple[torch.Tensor, ...]
    compact_backward_edge_slot: tuple[torch.Tensor, ...]
    compact_backward_edge_kind: tuple[torch.Tensor, ...]
    compact_backward_edge_prob: tuple[torch.Tensor, ...]
    compact_backward_edge_src_chance: tuple[torch.Tensor, ...]
    compact_backward_edge_dst_chance: tuple[torch.Tensor, ...]
    compact_backward_edge_prob_chance: tuple[torch.Tensor, ...]
    compact_backward_edge_src_p0: tuple[torch.Tensor, ...]
    compact_backward_edge_dst_p0: tuple[torch.Tensor, ...]
    compact_backward_edge_infoset_p0: tuple[torch.Tensor, ...]
    compact_backward_edge_slot_p0: tuple[torch.Tensor, ...]
    compact_backward_edge_flat_p0: tuple[torch.Tensor, ...]
    compact_backward_edge_prob_p0: tuple[torch.Tensor, ...]
    compact_backward_edge_src_p1: tuple[torch.Tensor, ...]
    compact_backward_edge_dst_p1: tuple[torch.Tensor, ...]
    compact_backward_edge_infoset_p1: tuple[torch.Tensor, ...]
    compact_backward_edge_slot_p1: tuple[torch.Tensor, ...]
    compact_backward_edge_flat_p1: tuple[torch.Tensor, ...]
    compact_backward_edge_prob_p1: tuple[torch.Tensor, ...]
    compact_forward_levels: tuple[tuple[int, ...], ...]
    compact_backward_levels: tuple[tuple[int, ...], ...]
    infoset_blocks: tuple[torch.Tensor, ...]
    frontier_nodes: torch.Tensor
    frontier_leaf_batch: LeafFeatureBatch
    root_child_nodes: torch.Tensor
    root_child_parent_infoset: int
    action_counts: torch.Tensor
    action_offsets: torch.Tensor


@dataclass(frozen=True, slots=True)
class GpuSolveTrace:
    packed: PackedGpuSolve
    iterations: int
    elapsed_seconds: float
    phase_seconds: dict[str, float]
    level_node_counts: tuple[int, ...]
    level_edge_counts: tuple[int, ...]
    level_frontier_counts: tuple[int, ...]
    compact_forward_level_sizes: tuple[int, ...]
    compact_backward_level_sizes: tuple[int, ...]
    compact_phase_seconds: dict[str, tuple[float, ...]]
    node_count: int
    leaf_count: int
    root_strategy: np.ndarray
    root_action_ev_player0: np.ndarray
    root_action_ev_player1: np.ndarray
    root_ev_player0: float
    root_ev_player1: float
    gpu_backward_p0: np.ndarray
    gpu_backward_p1: np.ndarray

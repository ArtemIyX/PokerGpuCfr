from __future__ import annotations

from typing import Any

from pokergpu.cfr import InfosetLayout, TreeLevels
from pokergpu.eval.types import LeafFeatureBatch
from pokergpu.tree import NodeType
from pokergpu.tree.builder import BuiltPublicTree

from .gpu_schedule import build_infoset_blocks, compact_level_schedule
from .gpu_types import BatchedGpuPlan


def build_batched_gpu_plan(
    tree: BuiltPublicTree,
    actions_by_node: tuple[tuple[object, ...], ...],
    levels: TreeLevels,
    layout: InfosetLayout,
    *,
    frontier_leaf_batch: LeafFeatureBatch,
    device: Any,
) -> BatchedGpuPlan:
    from .gpu_postflop import _build_level_plan, _node_type_code, _root_child_nodes
    import torch

    forward_levels: list[dict[str, torch.Tensor]] = []
    backward_levels: list[dict[str, torch.Tensor]] = []
    node_infoset: list[int] = []
    node_player: list[int] = []
    node_parent: list[int] = [-1] * tree.tree.node_count
    flat_view = tree.template.flat_view
    frontier_nodes = [
        index for index in range(tree.tree.node_count)
        if flat_view.is_frontier[index] and flat_view.node_type[index] is not NodeType.TERMINAL
    ]
    root_children = _root_child_nodes(tree)
    action_counts = torch.as_tensor(layout.action_counts, dtype=torch.int64, device=device)
    action_offsets = torch.as_tensor(layout.offsets, dtype=torch.int64, device=device)
    node_type_tensor = torch.as_tensor([_node_type_code(nt) for nt in flat_view.node_type], dtype=torch.int64, device=device)
    node_first_child = torch.as_tensor(flat_view.first_child, dtype=torch.int64, device=device)
    node_child_count = torch.as_tensor(flat_view.child_count, dtype=torch.int64, device=device)
    node_street = torch.as_tensor(flat_view.street, dtype=torch.int64, device=device)
    node_depth = torch.as_tensor(flat_view.node_depth, dtype=torch.int64, device=device)
    node_terminal_payoff = torch.as_tensor(flat_view.terminal_payoff, dtype=torch.float32, device=device)
    node_is_frontier = torch.as_tensor(flat_view.is_frontier, dtype=torch.bool, device=device)
    edge_parent = torch.as_tensor(flat_view.edge_parent, dtype=torch.int64, device=device)
    edge_child = torch.as_tensor(flat_view.edge_child, dtype=torch.int64, device=device)
    edge_action_slot = torch.as_tensor(flat_view.edge_action_slot, dtype=torch.int64, device=device)
    edge_chance_prob = torch.as_tensor(flat_view.edge_chance_prob, dtype=torch.float32, device=device)
    edge_infoset = torch.as_tensor(flat_view.edge_infoset_id, dtype=torch.int64, device=device)
    edge_node_type_tensor = torch.as_tensor(flat_view.edge_player, dtype=torch.int64, device=device)
    level_nodes = tuple(torch.as_tensor(level, dtype=torch.int64, device=device) for level in levels.forward_levels)
    level_frontier_mask: list[torch.Tensor] = []
    level_player_mask: list[torch.Tensor] = []
    level_edge_start: list[torch.Tensor] = []
    level_edge_count: list[torch.Tensor] = []
    level_edge_src: list[torch.Tensor] = []
    level_edge_dst: list[torch.Tensor] = []
    level_edge_infoset: list[torch.Tensor] = []
    level_edge_slot: list[torch.Tensor] = []
    level_edge_kind: list[torch.Tensor] = []
    level_edge_prob: list[torch.Tensor] = []
    compact_forward_levels = compact_level_schedule(levels.forward_levels)
    compact_backward_levels = compact_level_schedule(levels.backward_levels)
    infoset_blocks = build_infoset_blocks(layout, device=device)
    for level in levels.forward_levels:
        forward_levels.append(_build_level_plan(tree, level, actions_by_node, device=device))
    for level in levels.backward_levels:
        backward_levels.append(_build_level_plan(tree, level, actions_by_node, device=device))
    for level_tensor in level_nodes:
        level_set = set(int(i) for i in level_tensor.tolist())
        level_frontier_mask.append(torch.as_tensor([flat_view.is_frontier[int(i)] for i in level_tensor.tolist()], dtype=torch.bool, device=device))
        level_player_mask.append(torch.as_tensor([flat_view.node_type[int(i)] in {NodeType.PLAYER0, NodeType.PLAYER1} for i in level_tensor.tolist()], dtype=torch.bool, device=device))
        starts = [flat_view.first_child[int(i)] for i in level_tensor.tolist()]
        counts = [flat_view.child_count[int(i)] for i in level_tensor.tolist()]
        level_edge_start.append(torch.as_tensor(starts, dtype=torch.int64, device=device))
        level_edge_count.append(torch.as_tensor(counts, dtype=torch.int64, device=device))
        edge_indices = [i for i, parent in enumerate(flat_view.edge_parent) if parent in level_set]
        level_edge_src.append(torch.as_tensor([flat_view.edge_parent[i] for i in edge_indices], dtype=torch.int64, device=device))
        level_edge_dst.append(torch.as_tensor([flat_view.edge_child[i] for i in edge_indices], dtype=torch.int64, device=device))
        level_edge_infoset.append(torch.as_tensor([flat_view.edge_infoset_id[i] for i in edge_indices], dtype=torch.int64, device=device))
        level_edge_slot.append(torch.as_tensor([flat_view.edge_action_slot[i] for i in edge_indices], dtype=torch.int64, device=device))
        level_edge_kind.append(torch.as_tensor([flat_view.edge_player[i] for i in edge_indices], dtype=torch.int64, device=device))
        level_edge_prob.append(torch.as_tensor([flat_view.edge_chance_prob[i] for i in edge_indices], dtype=torch.float32, device=device))
    for node_index in range(tree.tree.node_count):
        node_type = tree.tree.node_types[node_index]
        node_player.append(0 if node_type is NodeType.PLAYER0 else 1 if node_type is NodeType.PLAYER1 else -1)
        infoset_id = tree.tree.infoset_ids[node_index]
        node_infoset.append(int(infoset_id) if infoset_id is not None else -1)
    for parent, child in enumerate(flat_view.edge_child):
        node_parent[int(child)] = int(flat_view.edge_parent[parent])
    return BatchedGpuPlan(
        node_type=node_type_tensor,
        node_first_child=node_first_child,
        node_child_count=node_child_count,
        node_parent=torch.as_tensor(node_parent, dtype=torch.int64, device=device),
        node_infoset=torch.as_tensor(node_infoset, dtype=torch.int64, device=device),
        node_street=node_street,
        node_depth=node_depth,
        node_terminal_payoff=node_terminal_payoff,
        node_is_frontier=node_is_frontier,
        forward_levels=tuple(forward_levels),
        backward_levels=tuple(backward_levels),
        level_frontier_mask=tuple(level_frontier_mask),
        level_player_mask=tuple(level_player_mask),
        edge_parent=edge_parent,
        edge_child=edge_child,
        edge_node_type=edge_node_type_tensor,
        edge_infoset=edge_infoset,
        edge_action_slot=edge_action_slot,
        edge_chance_prob=edge_chance_prob,
        level_nodes=level_nodes,
        node_player=torch.as_tensor(node_player, dtype=torch.int64, device=device),
        level_edge_start=tuple(level_edge_start),
        level_edge_count=tuple(level_edge_count),
        level_edge_src=tuple(level_edge_src),
        level_edge_dst=tuple(level_edge_dst),
        level_edge_infoset=tuple(level_edge_infoset),
        level_edge_slot=tuple(level_edge_slot),
        level_edge_kind=tuple(level_edge_kind),
        level_edge_prob=tuple(level_edge_prob),
        compact_forward_levels=compact_forward_levels,
        compact_backward_levels=compact_backward_levels,
        infoset_blocks=infoset_blocks,
        frontier_nodes=torch.as_tensor(frontier_nodes, dtype=torch.int64, device=device),
        frontier_leaf_batch=frontier_leaf_batch,
        root_child_nodes=torch.as_tensor(root_children, dtype=torch.int64, device=device),
        root_child_parent_infoset=int(tree.tree.infoset_ids[0] or 0),
        action_counts=action_counts,
        action_offsets=action_offsets,
    )

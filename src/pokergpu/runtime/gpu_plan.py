from __future__ import annotations

from typing import Any

import torch

from pokergpu.cfr import InfosetLayout, TreeLevels
from pokergpu.eval.types import LeafFeatureBatch
from pokergpu.tree import NodeType
from pokergpu.tree.builder import BuiltPublicTree

from .gpu_schedule import build_infoset_blocks, compact_level_schedule
from .gpu_types import BatchedGpuPlan


def concat_level_edges(
    state: Any,
    level_indices: tuple[int, ...],
) -> tuple[Any, Any, Any, Any, Any, Any]:
    import torch

    if not level_indices:
        empty_i64 = torch.empty(0, dtype=torch.int64, device=state.regrets.device)
        empty_f32 = torch.empty(0, dtype=torch.float32, device=state.regrets.device)
        return empty_i64, empty_i64, empty_i64, empty_i64, empty_i64, empty_f32
    edge_src = torch.cat([state.level_edge_src[index] for index in level_indices], dim=0)
    edge_dst = torch.cat([state.level_edge_dst[index] for index in level_indices], dim=0)
    edge_infoset = torch.cat([state.level_edge_infoset[index] for index in level_indices], dim=0)
    edge_slot = torch.cat([state.level_edge_slot[index] for index in level_indices], dim=0)
    edge_kind = torch.cat([state.level_edge_kind[index] for index in level_indices], dim=0)
    edge_prob = torch.cat([state.level_edge_prob[index] for index in level_indices], dim=0)
    return edge_src, edge_dst, edge_infoset, edge_slot, edge_kind, edge_prob


def concat_compact_level_edges(
    state: Any,
    compact_level_index: int,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    import torch

    if compact_level_index < 0 or compact_level_index >= len(state.compact_level_edge_dst):
        empty_i64 = torch.empty(0, dtype=torch.int64, device=state.regrets.device)
        empty_f32 = torch.empty(0, dtype=torch.float32, device=state.regrets.device)
        return empty_i64, empty_i64, empty_i64, empty_i64, empty_i64, empty_f32
    return (
        state.compact_level_edge_src[compact_level_index],
        state.compact_level_edge_dst[compact_level_index],
        state.compact_level_edge_infoset[compact_level_index],
        state.compact_level_edge_slot[compact_level_index],
        state.compact_level_edge_kind[compact_level_index],
        state.compact_level_edge_prob[compact_level_index],
    )


def init_gpu_ranges(
    root_ranges: tuple[Any, ...],
    *,
    device: Any,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    import numpy as np
    import torch

    tensors = []
    for index in range(6):
        if index < len(root_ranges):
            values = root_ranges[index].values
        else:
            values = np.zeros(1326, dtype=np.float32)
        tensors.append(torch.as_tensor(values, dtype=torch.float32, device=device))
    return tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5]


def root_child_nodes(tree: BuiltPublicTree) -> tuple[int, ...]:
    start = tree.tree.first_child[0]
    count = tree.tree.child_count[0]
    return tuple(int(tree.tree.children[start + index].child) for index in range(count))


def node_type_code(node_type: NodeType) -> int:
    if node_type is NodeType.CHANCE:
        return 0
    if node_type is NodeType.PLAYER0:
        return 1
    if node_type is NodeType.PLAYER1:
        return 2
    if node_type is NodeType.TERMINAL:
        return 3
    return 4


def build_level_plan(
    tree: BuiltPublicTree,
    level: tuple[int, ...],
    actions_by_node: tuple[tuple[object, ...], ...],
    *,
    device: Any,
) -> dict[str, Any]:
    import torch

    parents: list[int] = []
    children: list[int] = []
    node_types: list[int] = []
    infosets: list[int] = []
    action_slots: list[int] = []
    chance_probs: list[float] = []
    for node_index in level:
        if tree.tree.is_frontier[node_index]:
            continue
        node_type = tree.tree.node_types[node_index]
        start = tree.tree.first_child[node_index]
        count = tree.tree.child_count[node_index]
        if count <= 0:
            continue
        links = tree.tree.children[start : start + count]
        infoset_id = tree.tree.infoset_ids[node_index]
        for action_index, link in enumerate(links):
            parents.append(node_index)
            children.append(int(link.child))
            node_types.append(node_type_code(node_type))
            infosets.append(int(infoset_id) if infoset_id is not None else -1)
            action_slots.append(action_index)
            chance_probs.append(float(link.chance_prob or 0.0))
    return {
        "parents": torch.as_tensor(parents, dtype=torch.int64, device=device),
        "children": torch.as_tensor(children, dtype=torch.int64, device=device),
        "node_types": torch.as_tensor(node_types, dtype=torch.int64, device=device),
        "infosets": torch.as_tensor(infosets, dtype=torch.int64, device=device),
        "action_slots": torch.as_tensor(action_slots, dtype=torch.int64, device=device),
        "chance_probs": torch.as_tensor(chance_probs, dtype=torch.float32, device=device),
    }


def build_batched_gpu_plan(
    tree: BuiltPublicTree,
    actions_by_node: tuple[tuple[object, ...], ...],
    levels: TreeLevels,
    layout: InfosetLayout,
    *,
    frontier_leaf_batch: LeafFeatureBatch,
    device: Any,
) -> BatchedGpuPlan:
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
    root_children = root_child_nodes(tree)
    action_counts = torch.as_tensor(layout.action_counts, dtype=torch.int64, device=device)
    action_offsets = torch.as_tensor(layout.offsets, dtype=torch.int64, device=device)
    node_type_tensor = torch.as_tensor([node_type_code(nt) for nt in flat_view.node_type], dtype=torch.int64, device=device)
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
    max_actions = max(layout.action_counts) if layout.action_counts else 1
    for level in levels.forward_levels:
        forward_levels.append(build_level_plan(tree, level, actions_by_node, device=device))
    for level in levels.backward_levels:
        backward_levels.append(build_level_plan(tree, level, actions_by_node, device=device))
    compact_level_edge_src = [_concat_level_tensors(level_edge_src, level_indices, device=device) for level_indices in compact_forward_levels]
    compact_level_edge_dst = [_concat_level_tensors(level_edge_dst, level_indices, device=device) for level_indices in compact_forward_levels]
    compact_level_edge_infoset = [_concat_level_tensors(level_edge_infoset, level_indices, device=device) for level_indices in compact_forward_levels]
    compact_level_edge_slot = [_concat_level_tensors(level_edge_slot, level_indices, device=device) for level_indices in compact_forward_levels]
    compact_level_edge_kind = [_concat_level_tensors(level_edge_kind, level_indices, device=device) for level_indices in compact_forward_levels]
    compact_level_edge_prob = [_concat_level_tensors(level_edge_prob, level_indices, device=device) for level_indices in compact_forward_levels]
    compact_level_edge_src_chance: list[torch.Tensor] = []
    compact_level_edge_dst_chance: list[torch.Tensor] = []
    compact_level_edge_prob_chance: list[torch.Tensor] = []
    compact_level_edge_src_p0: list[torch.Tensor] = []
    compact_level_edge_dst_p0: list[torch.Tensor] = []
    compact_level_edge_infoset_p0: list[torch.Tensor] = []
    compact_level_edge_slot_p0: list[torch.Tensor] = []
    compact_level_edge_flat_p0: list[torch.Tensor] = []
    compact_level_edge_prob_p0: list[torch.Tensor] = []
    compact_level_edge_src_p1: list[torch.Tensor] = []
    compact_level_edge_dst_p1: list[torch.Tensor] = []
    compact_level_edge_infoset_p1: list[torch.Tensor] = []
    compact_level_edge_slot_p1: list[torch.Tensor] = []
    compact_level_edge_flat_p1: list[torch.Tensor] = []
    compact_level_edge_prob_p1: list[torch.Tensor] = []
    backward_edge_src: list[torch.Tensor] = []
    backward_edge_dst: list[torch.Tensor] = []
    backward_edge_infoset: list[torch.Tensor] = []
    backward_edge_slot: list[torch.Tensor] = []
    backward_edge_kind: list[torch.Tensor] = []
    backward_edge_prob: list[torch.Tensor] = []
    backward_edge_src_chance: list[torch.Tensor] = []
    backward_edge_dst_chance: list[torch.Tensor] = []
    backward_edge_prob_chance: list[torch.Tensor] = []
    backward_edge_src_p0: list[torch.Tensor] = []
    backward_edge_dst_p0: list[torch.Tensor] = []
    backward_edge_infoset_p0: list[torch.Tensor] = []
    backward_edge_slot_p0: list[torch.Tensor] = []
    backward_edge_flat_p0: list[torch.Tensor] = []
    backward_edge_prob_p0: list[torch.Tensor] = []
    backward_edge_src_p1: list[torch.Tensor] = []
    backward_edge_dst_p1: list[torch.Tensor] = []
    backward_edge_infoset_p1: list[torch.Tensor] = []
    backward_edge_slot_p1: list[torch.Tensor] = []
    backward_edge_flat_p1: list[torch.Tensor] = []
    backward_edge_prob_p1: list[torch.Tensor] = []
    for block_index, _level_indices in enumerate(compact_forward_levels):
        src = compact_level_edge_src[block_index]
        dst = compact_level_edge_dst[block_index]
        infoset = compact_level_edge_infoset[block_index]
        slot = compact_level_edge_slot[block_index]
        kind = compact_level_edge_kind[block_index]
        prob = compact_level_edge_prob[block_index]
        chance_mask = kind == 0
        p0_mask = kind == 1
        p1_mask = kind == 2
        compact_level_edge_src_chance.append(src[chance_mask])
        compact_level_edge_dst_chance.append(dst[chance_mask])
        compact_level_edge_prob_chance.append(prob[chance_mask])
        compact_level_edge_src_p0.append(src[p0_mask])
        compact_level_edge_dst_p0.append(dst[p0_mask])
        compact_level_edge_infoset_p0.append(infoset[p0_mask])
        compact_level_edge_slot_p0.append(slot[p0_mask])
        compact_level_edge_flat_p0.append(infoset[p0_mask] * max_actions + slot[p0_mask])
        compact_level_edge_prob_p0.append(prob[p0_mask])
        compact_level_edge_src_p1.append(src[p1_mask])
        compact_level_edge_dst_p1.append(dst[p1_mask])
        compact_level_edge_infoset_p1.append(infoset[p1_mask])
        compact_level_edge_slot_p1.append(slot[p1_mask])
        compact_level_edge_flat_p1.append(infoset[p1_mask] * max_actions + slot[p1_mask])
        compact_level_edge_prob_p1.append(prob[p1_mask])
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
        level_edge_kind.append(torch.as_tensor([node_type_code(flat_view.node_type[parent]) for parent in (flat_view.edge_parent[i] for i in edge_indices)], dtype=torch.int64, device=device))
        level_edge_prob.append(torch.as_tensor([flat_view.edge_chance_prob[i] for i in edge_indices], dtype=torch.float32, device=device))
    for level in levels.backward_levels:
        level_set = set(int(i) for i in level)
        edge_indices = [i for i, child in enumerate(flat_view.edge_child) if child in level_set]
        backward_edge_src.append(torch.as_tensor([flat_view.edge_parent[i] for i in edge_indices], dtype=torch.int64, device=device))
        backward_edge_dst.append(torch.as_tensor([flat_view.edge_child[i] for i in edge_indices], dtype=torch.int64, device=device))
        backward_edge_infoset.append(torch.as_tensor([flat_view.edge_infoset_id[i] for i in edge_indices], dtype=torch.int64, device=device))
        backward_edge_slot.append(torch.as_tensor([flat_view.edge_action_slot[i] for i in edge_indices], dtype=torch.int64, device=device))
        backward_edge_kind.append(torch.as_tensor([node_type_code(flat_view.node_type[flat_view.edge_parent[i]]) for i in edge_indices], dtype=torch.int64, device=device))
        backward_edge_prob.append(torch.as_tensor([flat_view.edge_chance_prob[i] for i in edge_indices], dtype=torch.float32, device=device))
        backward_chance_idx = [i for i in edge_indices if flat_view.node_type[flat_view.edge_parent[i]] is NodeType.CHANCE]
        backward_p0_idx = [i for i in edge_indices if flat_view.node_type[flat_view.edge_parent[i]] is NodeType.PLAYER0]
        backward_p1_idx = [i for i in edge_indices if flat_view.node_type[flat_view.edge_parent[i]] is NodeType.PLAYER1]
        backward_edge_src_chance.append(torch.as_tensor([flat_view.edge_parent[i] for i in backward_chance_idx], dtype=torch.int64, device=device))
        backward_edge_dst_chance.append(torch.as_tensor([flat_view.edge_child[i] for i in backward_chance_idx], dtype=torch.int64, device=device))
        backward_edge_prob_chance.append(torch.as_tensor([flat_view.edge_chance_prob[i] for i in backward_chance_idx], dtype=torch.float32, device=device))
        backward_edge_src_p0.append(torch.as_tensor([flat_view.edge_parent[i] for i in backward_p0_idx], dtype=torch.int64, device=device))
        backward_edge_dst_p0.append(torch.as_tensor([flat_view.edge_child[i] for i in backward_p0_idx], dtype=torch.int64, device=device))
        backward_edge_infoset_p0.append(torch.as_tensor([flat_view.edge_infoset_id[i] for i in backward_p0_idx], dtype=torch.int64, device=device))
        backward_edge_slot_p0.append(torch.as_tensor([flat_view.edge_action_slot[i] for i in backward_p0_idx], dtype=torch.int64, device=device))
        backward_edge_flat_p0.append(torch.as_tensor([flat_view.edge_infoset_id[i] * max_actions + flat_view.edge_action_slot[i] for i in backward_p0_idx], dtype=torch.int64, device=device))
        backward_edge_prob_p0.append(torch.as_tensor([flat_view.edge_chance_prob[i] for i in backward_p0_idx], dtype=torch.float32, device=device))
        backward_edge_src_p1.append(torch.as_tensor([flat_view.edge_parent[i] for i in backward_p1_idx], dtype=torch.int64, device=device))
        backward_edge_dst_p1.append(torch.as_tensor([flat_view.edge_child[i] for i in backward_p1_idx], dtype=torch.int64, device=device))
        backward_edge_infoset_p1.append(torch.as_tensor([flat_view.edge_infoset_id[i] for i in backward_p1_idx], dtype=torch.int64, device=device))
        backward_edge_slot_p1.append(torch.as_tensor([flat_view.edge_action_slot[i] for i in backward_p1_idx], dtype=torch.int64, device=device))
        backward_edge_flat_p1.append(torch.as_tensor([flat_view.edge_infoset_id[i] * max_actions + flat_view.edge_action_slot[i] for i in backward_p1_idx], dtype=torch.int64, device=device))
        backward_edge_prob_p1.append(torch.as_tensor([flat_view.edge_chance_prob[i] for i in backward_p1_idx], dtype=torch.float32, device=device))
    def _filter_backward_triplet(
        src_list: list[torch.Tensor],
        dst_list: list[torch.Tensor],
        infoset_list: list[torch.Tensor],
        slot_list: list[torch.Tensor],
        flat_list: list[torch.Tensor],
        prob_list: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        filtered_src: list[torch.Tensor] = []
        filtered_dst: list[torch.Tensor] = []
        filtered_infoset: list[torch.Tensor] = []
        filtered_slot: list[torch.Tensor] = []
        filtered_flat: list[torch.Tensor] = []
        filtered_prob: list[torch.Tensor] = []
        action_limit = layout.total_actions
        for index in range(len(src_list)):
            src = src_list[index]
            dst = dst_list[index]
            infoset = infoset_list[index]
            slot = slot_list[index]
            flat = flat_list[index]
            prob = prob_list[index]
            valid = (src >= 0) & (src < tree.tree.node_count) & (dst >= 0) & (dst < tree.tree.node_count)
            valid = valid & (infoset >= 0) & (infoset < layout.infoset_count) if hasattr(layout, "infoset_count") else valid
            valid = valid & (slot >= 0) & (slot < max_actions)
            valid = valid & (flat >= 0) & (flat < action_limit)
            filtered_src.append(src[valid])
            filtered_dst.append(dst[valid])
            filtered_infoset.append(infoset[valid])
            filtered_slot.append(slot[valid])
            filtered_flat.append(flat[valid])
            filtered_prob.append(prob[valid])
        return filtered_src, filtered_dst, filtered_infoset, filtered_slot, filtered_flat, filtered_prob
    compact_backward_edge_src_chance, compact_backward_edge_dst_chance, _, _, _, compact_backward_edge_prob_chance = _filter_backward_triplet(
        backward_edge_src_chance, backward_edge_dst_chance, backward_edge_src_chance, backward_edge_dst_chance, backward_edge_src_chance, backward_edge_prob_chance
    )
    compact_backward_edge_src_p0, compact_backward_edge_dst_p0, compact_backward_edge_infoset_p0, compact_backward_edge_slot_p0, compact_backward_edge_flat_p0, compact_backward_edge_prob_p0 = _filter_backward_triplet(
        backward_edge_src_p0, backward_edge_dst_p0, backward_edge_infoset_p0, backward_edge_slot_p0, backward_edge_flat_p0, backward_edge_prob_p0
    )
    compact_backward_edge_src_p1, compact_backward_edge_dst_p1, compact_backward_edge_infoset_p1, compact_backward_edge_slot_p1, compact_backward_edge_flat_p1, compact_backward_edge_prob_p1 = _filter_backward_triplet(
        backward_edge_src_p1, backward_edge_dst_p1, backward_edge_infoset_p1, backward_edge_slot_p1, backward_edge_flat_p1, backward_edge_prob_p1
    )
    compact_backward_edge_src = compact_backward_edge_src_p0
    compact_backward_edge_dst = compact_backward_edge_dst_p0
    compact_backward_edge_infoset = compact_backward_edge_infoset_p0
    compact_backward_edge_slot = compact_backward_edge_slot_p0
    compact_backward_edge_prob = compact_backward_edge_prob_p0
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
        compact_level_edge_src=tuple(compact_level_edge_src),
        compact_level_edge_dst=tuple(compact_level_edge_dst),
        compact_level_edge_infoset=tuple(compact_level_edge_infoset),
        compact_level_edge_slot=tuple(compact_level_edge_slot),
        compact_level_edge_kind=tuple(compact_level_edge_kind),
        compact_level_edge_prob=tuple(compact_level_edge_prob),
        compact_level_edge_src_chance=tuple(compact_level_edge_src_chance),
        compact_level_edge_dst_chance=tuple(compact_level_edge_dst_chance),
        compact_level_edge_prob_chance=tuple(compact_level_edge_prob_chance),
        compact_level_edge_src_p0=tuple(compact_level_edge_src_p0),
        compact_level_edge_dst_p0=tuple(compact_level_edge_dst_p0),
        compact_level_edge_infoset_p0=tuple(compact_level_edge_infoset_p0),
        compact_level_edge_slot_p0=tuple(compact_level_edge_slot_p0),
        compact_level_edge_flat_p0=tuple(compact_level_edge_flat_p0),
        compact_level_edge_prob_p0=tuple(compact_level_edge_prob_p0),
        compact_level_edge_src_p1=tuple(compact_level_edge_src_p1),
        compact_level_edge_dst_p1=tuple(compact_level_edge_dst_p1),
        compact_level_edge_infoset_p1=tuple(compact_level_edge_infoset_p1),
        compact_level_edge_slot_p1=tuple(compact_level_edge_slot_p1),
        compact_level_edge_flat_p1=tuple(compact_level_edge_flat_p1),
        compact_level_edge_prob_p1=tuple(compact_level_edge_prob_p1),
        compact_backward_edge_src=tuple(backward_edge_src),
        compact_backward_edge_dst=tuple(backward_edge_dst),
        compact_backward_edge_infoset=tuple(backward_edge_infoset),
        compact_backward_edge_slot=tuple(backward_edge_slot),
        compact_backward_edge_kind=tuple(backward_edge_kind),
        compact_backward_edge_prob=tuple(backward_edge_prob),
        compact_backward_edge_src_chance=tuple(backward_edge_src_chance),
        compact_backward_edge_dst_chance=tuple(backward_edge_dst_chance),
        compact_backward_edge_prob_chance=tuple(backward_edge_prob_chance),
        compact_backward_edge_src_p0=tuple(backward_edge_src_p0),
        compact_backward_edge_dst_p0=tuple(backward_edge_dst_p0),
        compact_backward_edge_infoset_p0=tuple(backward_edge_infoset_p0),
        compact_backward_edge_slot_p0=tuple(backward_edge_slot_p0),
        compact_backward_edge_flat_p0=tuple(backward_edge_flat_p0),
        compact_backward_edge_prob_p0=tuple(backward_edge_prob_p0),
        compact_backward_edge_src_p1=tuple(backward_edge_src_p1),
        compact_backward_edge_dst_p1=tuple(backward_edge_dst_p1),
        compact_backward_edge_infoset_p1=tuple(backward_edge_infoset_p1),
        compact_backward_edge_slot_p1=tuple(backward_edge_slot_p1),
        compact_backward_edge_flat_p1=tuple(backward_edge_flat_p1),
        compact_backward_edge_prob_p1=tuple(backward_edge_prob_p1),
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


def _concat_level_tensors(
    tensors: list[torch.Tensor],
    indices: tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    import torch

    selected = [tensors[index] for index in indices if 0 <= index < len(tensors)]
    if not selected:
        if not tensors:
            return torch.empty(0, dtype=torch.int64, device=device)
        return torch.empty(0, dtype=tensors[0].dtype, device=device)
    return torch.cat(selected, dim=0)

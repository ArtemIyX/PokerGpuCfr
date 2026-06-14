from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"torch is required for GPU subtree compilation: {exc}") from exc

from pokergpu.cfr.traversal import build_leaf_feature_batch
from pokergpu.abstraction.hands import private_hand_mask
from pokergpu.eval.tensor_builder import build_gpu_leaf_tensors
from pokergpu.tree import NodeType
from pokergpu.tree.builder import BuiltPublicTree

from .cache import PackedGpuSubtree


@dataclass(frozen=True, slots=True)
class PackedGpuCompileResult:
    subtree: PackedGpuSubtree
    device: str


def compile_packed_subtree(
    tree: BuiltPublicTree,
    *,
    spot_key: str,
    device: str = "cuda",
) -> PackedGpuSubtree:
    flat = tree.flat_view
    frontier_node_list = [
        index
        for level in tree.level_schedule.forward_levels
        for index in level
        if flat.is_frontier[index] and flat.node_type[index] is not NodeType.TERMINAL
    ]
    infoset_remap: dict[int, int] = {}
    compact_infoset_ids: list[int] = []
    for infoset in flat.infoset_id:
        if infoset < 0:
            compact_infoset_ids.append(-1)
            continue
        infoset_int = int(infoset)
        compact_index = infoset_remap.setdefault(infoset_int, len(infoset_remap))
        compact_infoset_ids.append(compact_index)
    node_type = torch.as_tensor(
        [_node_type_code(flat.node_type[old_index]) for old_index in range(tree.tree.node_count)],
        dtype=torch.int64,
        device=device,
    )
    is_frontier = torch.as_tensor([flat.is_frontier[old_index] for old_index in range(tree.tree.node_count)], dtype=torch.bool, device=device)
    first_child = torch.as_tensor([flat.first_child[old_index] for old_index in range(tree.tree.node_count)], dtype=torch.int64, device=device)
    child_count = torch.as_tensor([flat.child_count[old_index] for old_index in range(tree.tree.node_count)], dtype=torch.int64, device=device)
    infoset_ids = torch.as_tensor([compact_infoset_ids[old_index] for old_index in range(tree.tree.node_count)], dtype=torch.int64, device=device)
    terminal_payoffs = torch.as_tensor([flat.terminal_payoff[old_index] for old_index in range(tree.tree.node_count)], dtype=torch.float32, device=device)
    node_depth = torch.as_tensor([flat.node_depth[old_index] for old_index in range(tree.tree.node_count)], dtype=torch.int64, device=device)
    street = torch.as_tensor([flat.street[old_index] for old_index in range(tree.tree.node_count)], dtype=torch.int64, device=device)
    action_slot = torch.as_tensor([flat.edge_action_slot[i] for i in range(len(flat.edge_action_slot))], dtype=torch.int64, device=device)
    chance_prob = torch.as_tensor([flat.edge_chance_prob[i] for i in range(len(flat.edge_chance_prob))], dtype=torch.float32, device=device)
    children = torch.as_tensor([int(flat.edge_child[i]) for i in range(len(flat.edge_child))], dtype=torch.int64, device=device)
    frontier_nodes = torch.as_tensor(tuple(frontier_node_list), dtype=torch.int64, device=device)
    frontier_node_indices = tuple(int(index) for index in frontier_nodes.tolist())
    leaf_feature_batch = build_leaf_feature_batch(
        tree.tree,
        frontier_node_indices,
        node_states=tree.node_states,
    )
    leaf_feature_tensors = build_gpu_leaf_tensors(leaf_feature_batch, torch.device(device))
    action_infoset_index, action_slot_index = _action_maps(flat, infoset_remap, device=device)
    if action_infoset_index.numel() != action_slot_index.numel():
        raise ValueError("packed action maps must have matching lengths")
    legal_action_mask = torch.ones_like(action_infoset_index, dtype=torch.bool, device=device)
    card_removal_mask = _build_card_removal_mask(tree, device=device)
    root_infoset = int(infoset_ids[0].item()) if infoset_ids.numel() else -1
    max_actions = max((len(actions) for actions in tree.actions_by_node), default=0)
    infoset_count = len(infoset_remap)
    return PackedGpuSubtree(
        spot_key=spot_key,
        node_type=node_type,
        is_frontier=is_frontier,
        first_child=first_child,
        child_count=child_count,
        children=children,
        infoset_ids=infoset_ids,
        terminal_payoffs=terminal_payoffs,
        node_depth=node_depth,
        street=street,
        action_slot=action_slot,
        chance_prob=chance_prob,
        frontier_nodes=frontier_nodes,
        leaf_feature_batch=leaf_feature_batch,
        leaf_feature_tensors=leaf_feature_tensors,
        action_infoset_index=action_infoset_index,
        action_slot_index=action_slot_index,
        legal_action_mask=legal_action_mask,
        card_removal_mask=card_removal_mask,
        root_node=0,
        root_infoset=root_infoset,
        node_count=tree.tree.node_count,
        edge_count=int(children.numel()),
        infoset_count=infoset_count,
        leaf_count=int(frontier_nodes.numel()),
        max_actions=max_actions,
        device=device,
        tree_version="1",
    )


def _node_type_code(node_type: NodeType) -> int:
    if node_type is NodeType.CHANCE:
        return 0
    if node_type is NodeType.PLAYER0:
        return 1
    if node_type is NodeType.PLAYER1:
        return 2
    if node_type is NodeType.PLAYER2:
        return 3
    return 4


def _action_maps(
    flat: object,
    infoset_remap: dict[int, int],
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    infoset_index: list[int] = []
    action_slot: list[int] = []
    max_by_infoset: dict[int, int] = {}
    edge_infoset_ids = getattr(flat, "edge_infoset_id")
    edge_action_slots = getattr(flat, "edge_action_slot")
    for infoset, slot in zip(edge_infoset_ids, edge_action_slots, strict=True):
        if infoset < 0:
            continue
        compact_index = infoset_remap[int(infoset)]
        slot_index = int(slot)
        max_by_infoset[compact_index] = max(max_by_infoset.get(compact_index, 0), slot_index + 1)
    for compact_index in sorted(max_by_infoset):
        for slot in range(max_by_infoset[compact_index]):
            infoset_index.append(compact_index)
            action_slot.append(slot)
    return (
        torch.as_tensor(infoset_index, dtype=torch.int64, device=device),
        torch.as_tensor(action_slot, dtype=torch.int64, device=device),
    )


def _build_card_removal_mask(tree: BuiltPublicTree, *, device: str) -> torch.Tensor:
    flat = tree.flat_view
    hand_count = 1326
    masks: list[torch.Tensor] = []
    for node_state in tree.node_states:
        board_cards = tuple(node_state.board.cards)
        masks.append(torch.as_tensor(private_hand_mask(board_cards), dtype=torch.bool, device=device))
    if not masks:
        return torch.zeros((0, hand_count), dtype=torch.bool, device=device)
    return torch.stack(masks, dim=0)

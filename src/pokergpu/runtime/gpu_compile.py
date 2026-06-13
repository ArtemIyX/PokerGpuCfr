from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"torch is required for GPU subtree compilation: {exc}") from exc

from pokergpu.cfr.traversal import build_leaf_feature_batch
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
        [_node_type_code(node_type) for node_type in flat.node_type],
        dtype=torch.int64,
        device=device,
    )
    is_frontier = torch.as_tensor(flat.is_frontier, dtype=torch.bool, device=device)
    first_child = torch.as_tensor(flat.first_child, dtype=torch.int64, device=device)
    child_count = torch.as_tensor(flat.child_count, dtype=torch.int64, device=device)
    infoset_ids = torch.as_tensor(compact_infoset_ids, dtype=torch.int64, device=device)
    terminal_payoffs = torch.as_tensor(flat.terminal_payoff, dtype=torch.float32, device=device)
    node_depth = torch.as_tensor(flat.node_depth, dtype=torch.int64, device=device)
    street = torch.as_tensor(flat.street, dtype=torch.int64, device=device)
    action_slot = torch.as_tensor(
        [slot if slot >= 0 else -1 for slot in flat.edge_action_slot],
        dtype=torch.int64,
        device=device,
    )
    chance_prob = torch.as_tensor(flat.edge_chance_prob, dtype=torch.float32, device=device)
    children = torch.as_tensor(flat.edge_child, dtype=torch.int64, device=device)
    frontier_nodes = torch.as_tensor(
        [index for index, is_front in enumerate(flat.is_frontier) if is_front and flat.node_type[index] is not NodeType.TERMINAL],
        dtype=torch.int64,
        device=device,
    )
    leaf_feature_batch = build_leaf_feature_batch(tree.tree, tuple(int(index) for index in frontier_nodes.tolist()), node_states=tree.node_states)
    action_infoset_index, action_slot_index = _action_maps(tree, infoset_remap, device=device)
    if action_infoset_index.numel() != action_slot_index.numel():
        raise ValueError("packed action maps must have matching lengths")
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
        action_infoset_index=action_infoset_index,
        action_slot_index=action_slot_index,
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
    tree: BuiltPublicTree,
    infoset_remap: dict[int, int],
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    infoset_index: list[int] = []
    action_slot: list[int] = []
    max_by_infoset: dict[int, int] = {}
    for node_index, infoset in enumerate(tree.tree.infoset_ids):
        if infoset is None:
            continue
        actions = tree.actions_by_node[node_index]
        infoset_int = int(infoset)
        compact_index = infoset_remap[infoset_int]
        max_by_infoset[compact_index] = max(max_by_infoset.get(compact_index, 0), len(actions))
    for infoset_int in sorted(max_by_infoset):
        for slot in range(max_by_infoset[infoset_int]):
            infoset_index.append(infoset_int)
            action_slot.append(slot)
    return (
        torch.as_tensor(infoset_index, dtype=torch.int64, device=device),
        torch.as_tensor(action_slot, dtype=torch.int64, device=device),
    )

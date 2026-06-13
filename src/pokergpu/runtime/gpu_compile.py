from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"torch is required for GPU subtree compilation: {exc}") from exc

from pokergpu.tree import NodeType
from pokergpu.tree.builder import BuiltPublicTree
from pokergpu.cfr.traversal import build_leaf_feature_batch

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
    node_count = tree.tree.node_count
    node_type = torch.as_tensor(
        [_node_type_code(node_type) for node_type in tree.tree.node_types],
        dtype=torch.int64,
        device=device,
    )
    is_frontier = torch.as_tensor(tree.tree.is_frontier, dtype=torch.bool, device=device)
    first_child = torch.as_tensor(tree.tree.first_child, dtype=torch.int64, device=device)
    child_count = torch.as_tensor(tree.tree.child_count, dtype=torch.int64, device=device)
    infoset_ids = torch.as_tensor(
        [int(infoset) if infoset is not None else -1 for infoset in tree.tree.infoset_ids],
        dtype=torch.int64,
        device=device,
    )
    terminal_payoffs = torch.as_tensor(
        [float(payoff or 0.0) for payoff in tree.tree.terminal_payoffs],
        dtype=torch.float32,
        device=device,
    )
    node_depth = torch.as_tensor(_node_depths(tree), dtype=torch.int64, device=device)
    street = torch.as_tensor(_street_ids(tree), dtype=torch.int64, device=device)
    action_slot = torch.as_tensor(_action_slots(tree), dtype=torch.int64, device=device)
    chance_prob = torch.as_tensor(_chance_probs(tree), dtype=torch.float32, device=device)
    children = torch.as_tensor(
        [int(link.child) for link in tree.tree.children],
        dtype=torch.int64,
        device=device,
    )
    frontier_nodes = torch.as_tensor(
        [index for index, is_front in enumerate(tree.tree.is_frontier) if is_front and tree.tree.node_types[index] is not NodeType.TERMINAL],
        dtype=torch.int64,
        device=device,
    )
    leaf_feature_batch = build_leaf_feature_batch(tree.tree, tuple(int(index) for index in frontier_nodes.tolist()), node_states=tree.node_states)
    action_infoset_index, action_slot_index = _action_maps(tree, device=device)
    root_infoset = int(tree.tree.infoset_ids[0] or -1)
    max_actions = max((len(actions) for actions in tree.actions_by_node), default=0)
    infoset_count = max((int(infoset) for infoset in tree.tree.infoset_ids if infoset is not None), default=-1) + 1
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
        node_count=node_count,
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


def _node_depths(tree: BuiltPublicTree) -> list[int]:
    depth = [-1] * tree.tree.node_count
    depth[0] = 0
    for node_index in range(tree.tree.node_count):
        start = tree.tree.first_child[node_index]
        count = tree.tree.child_count[node_index]
        for link_index in range(start, start + count):
            child = int(tree.tree.children[link_index].child)
            depth[child] = depth[node_index] + 1
    return depth


def _street_ids(tree: BuiltPublicTree) -> list[int]:
    values: list[int] = []
    for state in tree.node_states:
        street_value = getattr(state.current_street, "value", "")
        values.append(
            {
                "preflop": 0,
                "flop": 1,
                "turn": 2,
                "river": 3,
            }.get(street_value, 0)
        )
    return values


def _action_slots(tree: BuiltPublicTree) -> list[int]:
    slots: list[int] = []
    for node_index, actions in enumerate(tree.actions_by_node):
        count = len(actions)
        if count == 0:
            slots.append(-1)
            continue
        slots.extend(range(count))
    if len(slots) < tree.tree.node_count:
        slots.extend([-1] * (tree.tree.node_count - len(slots)))
    return slots[: tree.tree.node_count]


def _chance_probs(tree: BuiltPublicTree) -> list[float]:
    probs: list[float] = []
    for link in tree.tree.children:
        probs.append(float(link.chance_prob or 0.0))
    return probs


def _action_maps(tree: BuiltPublicTree, *, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    infoset_index: list[int] = []
    action_slot: list[int] = []
    for node_index, infoset in enumerate(tree.tree.infoset_ids):
        if infoset is None:
            continue
        actions = tree.actions_by_node[node_index]
        for slot in range(len(actions)):
            infoset_index.append(int(infoset))
            action_slot.append(slot)
    return (
        torch.as_tensor(infoset_index, dtype=torch.int64, device=device),
        torch.as_tensor(action_slot, dtype=torch.int64, device=device),
    )

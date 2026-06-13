from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType

from pokergpu.core.betting import Chips

NodeId = NewType("NodeId", int)
InfosetId = NewType("InfosetId", int)


class NodeType(StrEnum):
    PLAYER0 = "player0"
    PLAYER1 = "player1"
    PLAYER2 = "player2"
    PLAYER3 = "player3"
    PLAYER4 = "player4"
    PLAYER5 = "player5"
    CHANCE = "chance"
    TERMINAL = "terminal"
    LEAF = "leaf"


@dataclass(slots=True, frozen=True)
class ChildLink:
    child: NodeId
    chance_prob: float | None = None


@dataclass(slots=True, frozen=True)
class PublicTreeFlatView:
    node_type: tuple[NodeType, ...]
    node_depth: tuple[int, ...]
    node_level: tuple[int, ...]
    street: tuple[int, ...]
    infoset_id: tuple[int, ...]
    first_child: tuple[int, ...]
    child_count: tuple[int, ...]
    is_frontier: tuple[bool, ...]
    terminal_payoff: tuple[float, ...]
    edge_parent: tuple[int, ...]
    edge_child: tuple[int, ...]
    edge_action_slot: tuple[int, ...]
    edge_chance_prob: tuple[float, ...]
    edge_infoset_id: tuple[int, ...]
    edge_player: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class PublicTreeLevelSchedule:
    forward_levels: tuple[tuple[int, ...], ...]
    backward_levels: tuple[tuple[int, ...], ...]
    level_nodes: tuple[tuple[int, ...], ...]


@dataclass(slots=True, frozen=True)
class PublicTree:
    node_types: tuple[NodeType, ...]
    is_frontier: tuple[bool, ...]
    first_child: tuple[int, ...]
    child_count: tuple[int, ...]
    children: tuple[ChildLink, ...]
    infoset_ids: tuple[InfosetId | None, ...]
    terminal_payoffs: tuple[Chips | None, ...]

    def __post_init__(self) -> None:
        node_count = len(self.node_types)
        aligned_lengths = (
            len(self.is_frontier),
            len(self.first_child),
            len(self.child_count),
            len(self.infoset_ids),
            len(self.terminal_payoffs),
        )
        if any(length != node_count for length in aligned_lengths):
            raise ValueError("all per-node arrays must have the same length")

        for node_index in range(node_count):
            node_type = self.node_types[node_index]
            start = self.first_child[node_index]
            count = self.child_count[node_index]
            end = start + count
            frontier = self.is_frontier[node_index]

            if start < 0 or count < 0 or end > len(self.children):
                raise ValueError("child ranges must stay within children array bounds")

            infoset_id = self.infoset_ids[node_index]
            payoff = self.terminal_payoffs[node_index]

            if node_type in {
                NodeType.PLAYER0,
                NodeType.PLAYER1,
                NodeType.PLAYER2,
                NodeType.PLAYER3,
                NodeType.PLAYER4,
                NodeType.PLAYER5,
            }:
                if infoset_id is None:
                    raise ValueError("player nodes must have infoset ids")
                if payoff is not None:
                    raise ValueError("player nodes cannot have terminal payoffs")
            elif infoset_id is not None:
                raise ValueError("only player nodes may have infoset ids")

            if node_type is NodeType.TERMINAL:
                if payoff is None:
                    raise ValueError("terminal nodes must have payoffs")
                if count != 0:
                    raise ValueError("terminal nodes cannot have children")
            elif payoff is not None:
                raise ValueError("only terminal nodes may have payoffs")

            if node_type in {NodeType.LEAF, NodeType.TERMINAL} and count != 0:
                raise ValueError("leaf and terminal nodes cannot have children")
            if frontier and node_type is NodeType.TERMINAL and count != 0:
                raise ValueError("frontier terminal nodes cannot have children")

            child_slice = self.children[start:end]
            if node_type is NodeType.CHANCE:
                if count == 0:
                    raise ValueError("chance nodes must have children")
                total_prob = 0.0
                for link in child_slice:
                    if link.chance_prob is None:
                        raise ValueError("chance children must define probabilities")
                    if link.chance_prob < 0.0 or link.chance_prob > 1.0:
                        raise ValueError("chance probabilities must be in [0, 1]")
                    total_prob += link.chance_prob
                    if link.child < 0 or link.child >= node_count:
                        raise ValueError("child node ids must reference valid nodes")
                if abs(total_prob - 1.0) > 1e-6:
                    raise ValueError("chance child probabilities must sum to 1")
            else:
                for link in child_slice:
                    if link.chance_prob is not None:
                        raise ValueError(
                            "non-chance child links cannot carry chance probabilities"
                        )
                    if link.child < 0 or link.child >= node_count:
                        raise ValueError("child node ids must reference valid nodes")

    @property
    def node_count(self) -> int:
        return len(self.node_types)

    def child_links(self, node: NodeId) -> tuple[ChildLink, ...]:
        start = self.first_child[node]
        count = self.child_count[node]
        return self.children[start : start + count]


@dataclass(slots=True, frozen=True)
class PublicTreeTemplate:
    node_types: tuple[NodeType, ...]
    is_frontier: tuple[bool, ...]
    first_child: tuple[int, ...]
    child_count: tuple[int, ...]
    infoset_ids: tuple[InfosetId | None, ...]
    terminal_payoffs: tuple[Chips | None, ...]
    depth: tuple[int, ...]
    street: tuple[str, ...]
    edge_parent: tuple[int, ...]
    edge_child: tuple[int, ...]
    edge_action_slot: tuple[int, ...]
    edge_chance_prob: tuple[float, ...]
    edge_infoset_id: tuple[int, ...]
    edge_player: tuple[int, ...]
    level_schedule: PublicTreeLevelSchedule
    flat_view: PublicTreeFlatView
    canonical_board_key: str
    action_abstraction_id: str
    tree_key: str

    @property
    def node_count(self) -> int:
        return len(self.node_types)

    def as_public_tree(self, children: tuple[ChildLink, ...]) -> PublicTree:
        return PublicTree(
            node_types=self.node_types,
            is_frontier=self.is_frontier,
            first_child=self.first_child,
            child_count=self.child_count,
            children=children,
            infoset_ids=self.infoset_ids,
            terminal_payoffs=self.terminal_payoffs,
        )

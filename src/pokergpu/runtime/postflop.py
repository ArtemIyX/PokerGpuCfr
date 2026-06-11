from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pokergpu.abstraction.actions import BaselineActionAbstraction, make_compact_profile
from pokergpu.abstraction.hands import RangeVector
from pokergpu.cfr import (
    InfosetLayout,
    InfosetStore,
    compute_counterfactual_values,
    compute_reach_probabilities,
    update_regrets_from_traversal,
)
from pokergpu.core.board import Street
from pokergpu.core.cards import Card
from pokergpu.core.state import GameState
from pokergpu.eval import CpuStubLeafEvaluator, LeafEvaluator
from pokergpu.tree.builder import TreeBuildConfig, build_public_tree


@dataclass(frozen=True, slots=True)
class PostflopResolveSpec:
    state: GameState
    range_p0: RangeVector
    range_p1: RangeVector
    time_budget_sec: float
    max_depth: int = 2
    max_nodes: int = 256
    min_reach_prob: float = 0.0


@dataclass(frozen=True, slots=True)
class PostflopResolveResult:
    root_infoset_id: int
    root_actions: tuple[str, ...]
    root_strategy: NDArray[np.float32]
    iterations: int
    elapsed_seconds: float
    node_count: int
    leaf_count: int


def resolve_postflop_hu(
    spec: PostflopResolveSpec,
    *,
    evaluator: LeafEvaluator | None = None,
) -> PostflopResolveResult:
    if spec.state.player_count != 2:
        raise ValueError("postflop resolver supports heads-up only")
    if spec.state.current_street is Street.PREFLOP:
        raise ValueError("postflop resolver requires a postflop state")

    evaluator_impl = evaluator or CpuStubLeafEvaluator()
    started_at = time.monotonic()
    tree = build_public_tree(
        spec.state,
        abstraction=BaselineActionAbstraction(profile=make_compact_profile()),
        config=TreeBuildConfig(
            max_depth=spec.max_depth,
            max_nodes=spec.max_nodes,
            min_reach_prob=spec.min_reach_prob,
        ),
    )
    root_infoset = tree.tree.infoset_ids[0]
    if root_infoset is None:
        raise ValueError("root node must be a player infoset")
    root_infoset_id = int(root_infoset)
    root_actions = tuple(str(action) for action in tree.actions_by_node[0])
    action_counts = tuple(
        len(tree.actions_by_node[node_index]) if (tree.tree.infoset_ids[node_index] 
                                                  is not None) else 1
        for node_index in range(tree.tree.node_count)
        if tree.tree.infoset_ids[node_index] is not None
    )
    if not action_counts:
        raise ValueError("resolver requires at least one player infoset")
    store = InfosetStore.zeros(InfosetLayout.from_action_counts(action_counts))

    _masked_root_ranges = _apply_root_ranges(
        spec.state,
        spec.range_p0,
        spec.range_p1,
    )

    deadline = time.monotonic() + max(0.0, spec.time_budget_sec)
    iterations = 0
    leaf_count = int(np.count_nonzero(tree.tree.is_frontier))
    while time.monotonic() < deadline or iterations == 0:
        compute_reach_probabilities(
            tree.tree,
            store,
            min_reach_prob=spec.min_reach_prob,
        )
        backward = compute_counterfactual_values(
            tree.tree,
            store,
            evaluator=evaluator_impl,
        )
        update_regrets_from_traversal(
            tree.tree,
            store,
            backward,
            active_player=0,
        )
        update_regrets_from_traversal(
            tree.tree,
            store,
            backward,
            active_player=1,
        )
        iterations += 1
        if spec.time_budget_sec <= 0.0:
            break

    root_strategy = store.average_strategy(0)
    return PostflopResolveResult(
        root_infoset_id=root_infoset_id,
        root_actions=root_actions,
        root_strategy=root_strategy,
        iterations=iterations,
        elapsed_seconds=time.monotonic() - started_at,
        node_count=tree.tree.node_count,
        leaf_count=leaf_count,
    )


def _apply_root_ranges(
    state: GameState,
    range_p0: RangeVector,
    range_p1: RangeVector,
) -> tuple[RangeVector, RangeVector]:
    dead_cards: list[Card] = list(state.board.cards)
    for player in state.players:
        if player.hole_cards is not None:
            dead_cards.extend(player.hole_cards)
    masked_p0 = range_p0.normalized_masked(dead_cards)
    masked_p1 = range_p1.normalized_masked(dead_cards)
    return masked_p0, masked_p1

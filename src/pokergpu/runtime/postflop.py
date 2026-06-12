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
from pokergpu.core.actions import Action
from pokergpu.core.betting import Chips
from pokergpu.core.board import Street
from pokergpu.core.cards import Card
from pokergpu.core.payouts import compute_payouts
from pokergpu.core.state import GameState, HandPhase
from pokergpu.eval import LeafEvaluator
from pokergpu.runtime.cache import WarmStartState
from pokergpu.runtime.caching import (
    PublicStateFingerprint,
    SolveCacheState,
    make_warm_start_state,
)
from pokergpu.runtime.value_network import default_postflop_leaf_evaluator
from pokergpu.tree import PublicTree
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
    cache_state: SolveCacheState | None = None


@dataclass(frozen=True, slots=True)
class PostflopResolveResult:
    root_infoset_id: int
    root_actions: tuple[str, ...]
    root_strategy: NDArray[np.float32]
    root_ev_player0: float
    root_ev_player1: float
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

    evaluator_impl = evaluator or default_postflop_leaf_evaluator()
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
    root_actions = tuple(_format_action(action) for action in tree.actions_by_node[0])
    action_counts = _build_infoset_action_counts(tree.tree, tree.actions_by_node)
    if not action_counts:
        raise ValueError("resolver requires at least one player infoset")
    store = InfosetStore.zeros(InfosetLayout.from_action_counts(action_counts))
    public_fingerprint = _build_public_fingerprint(spec, tree, evaluator_impl)
    if spec.cache_state is not None:
        warm_start = spec.cache_state.lookup_warm_start(public_fingerprint.digest())
        if warm_start is not None:
            _apply_warm_start(store, warm_start)

    root_range_p0, root_range_p1, _, _ = _apply_root_ranges(
        spec.state,
        spec.range_p0,
        spec.range_p1,
    )
    terminal_values_player0 = np.array(
        [
            np.float32(_terminal_value_player0(node_state))
            for node_state in tree.node_states
        ],
        dtype=np.float32,
    )

    deadline = time.monotonic() + max(0.0, spec.time_budget_sec)
    iterations = 0
    leaf_count = int(np.count_nonzero(tree.tree.is_frontier))
    while time.monotonic() < deadline or iterations == 0:
        forward = compute_reach_probabilities(
            tree.tree,
            store,
            root_player0_reach=root_range_p0.total_weight(),
            root_player1_reach=root_range_p1.total_weight(),
            min_reach_prob=spec.min_reach_prob,
        )
        backward = compute_counterfactual_values(
            tree.tree,
            store,
            node_states=tree.node_states,
            reach_p0=forward.player0_reach,
            reach_p1=forward.player1_reach,
            terminal_values_player0=terminal_values_player0,
            evaluator=evaluator_impl,
        )
        update_regrets_from_traversal(
            tree.tree,
            store,
            backward,
            active_player=0,
            strategy_weight=root_range_p0.total_weight(),
        )
        update_regrets_from_traversal(
            tree.tree,
            store,
            backward,
            active_player=1,
            strategy_weight=root_range_p1.total_weight(),
        )
        iterations += 1
        if spec.time_budget_sec <= 0.0:
            break

    final_backward = compute_counterfactual_values(
        tree.tree,
        store,
        node_states=tree.node_states,
        reach_p0=forward.player0_reach,
        reach_p1=forward.player1_reach,
        terminal_values_player0=terminal_values_player0,
        evaluator=evaluator_impl,
    )

    if spec.cache_state is not None:
        spec.cache_state.store_warm_start(
            public_fingerprint.digest(),
            make_warm_start_state(
                regret=tuple(float(value) for value in store.regrets),
                strategy_sum=tuple(float(value) for value in store.strategy_sums),
                source_key=public_fingerprint.digest(),
                blend_alpha=1.0,
            ),
        )

    root_strategy = store.average_strategy(0)
    return PostflopResolveResult(
        root_infoset_id=root_infoset_id,
        root_actions=root_actions,
        root_strategy=root_strategy,
        root_ev_player0=float(final_backward.node_values_player0[0]),
        root_ev_player1=float(final_backward.node_values_player1[0]),
        iterations=iterations,
        elapsed_seconds=time.monotonic() - started_at,
        node_count=tree.tree.node_count,
        leaf_count=leaf_count,
    )


def _apply_root_ranges(
    state: GameState,
    range_p0: RangeVector,
    range_p1: RangeVector,
) -> tuple[RangeVector, RangeVector, float, float]:
    dead_cards: list[Card] = list(state.board.cards)
    for player in state.players:
        if player.hole_cards is not None:
            dead_cards.extend(player.hole_cards)
    raw_masked_p0 = range_p0.masked(dead_cards)
    raw_masked_p1 = range_p1.masked(dead_cards)
    weight_p0 = raw_masked_p0.total_weight()
    weight_p1 = raw_masked_p1.total_weight()
    if weight_p0 <= 0.0 or weight_p1 <= 0.0:
        raise ValueError(
            "root ranges must retain positive weight after dead-card masking"
        )
    masked_p0 = raw_masked_p0.normalized()
    masked_p1 = raw_masked_p1.normalized()
    return masked_p0, masked_p1, weight_p0, weight_p1


def _build_infoset_action_counts(
    tree: PublicTree,
    actions_by_node: tuple[tuple[Action, ...], ...],
) -> tuple[int, ...]:
    max_infoset = -1
    counts: dict[int, int] = {}
    for node_index, infoset_id in enumerate(tree.infoset_ids):
        if infoset_id is None:
            continue
        infoset_index = int(infoset_id)
        max_infoset = max(max_infoset, infoset_index)
        counts.setdefault(infoset_index, len(actions_by_node[node_index]) or 1)
    return tuple(counts.get(index, 1) for index in range(max_infoset + 1))


def _format_action(action: Action) -> str:
    if action.amount is None:
        return action.action_type.value
    return f"{action.action_type.value}({int(action.amount)})"


def _terminal_value_player0(state: GameState) -> float:
    if state.phase is not HandPhase.TERMINAL:
        return 0.0
    payouts = compute_payouts(state)
    player0_payout = next(
        (payout.amount for payout in payouts if payout.player == 0),
        Chips(0),
    )
    other_payouts = sum(payout.amount for payout in payouts if payout.player != 0)
    return float(player0_payout - other_payouts)


def _build_public_fingerprint(
    spec: PostflopResolveSpec,
    tree: object,
    evaluator: LeafEvaluator,
) -> PublicStateFingerprint:
    built_tree = tree
    canonical_board = getattr(built_tree, "canonical_board_key", "")
    action_abstraction_id = getattr(built_tree, "action_abstraction_id", "")
    return PublicStateFingerprint(
        variant="nlhe",
        street=spec.state.current_street.value,
        acting_player=int(spec.state.betting_round.to_act),
        pot=int(spec.state.betting_round.pot.amount),
        stacks=tuple(int(stack.stack) for stack in spec.state.betting_round.stacks),
        blinds=(
            int(spec.state.betting_round.blinds.small_blind),
            int(spec.state.betting_round.blinds.big_blind),
        ),
        antes=(int(spec.state.betting_round.blinds.ante),) * spec.state.player_count,
        board=tuple(str(card) for card in spec.state.board.cards),
        action_history=(),
        action_abstraction_id=action_abstraction_id,
        range_abstraction_id="private_hand_v1",
        subtree_depth_limit=spec.max_depth,
        evaluator_id=evaluator.__class__.__name__,
        solver_version="1",
        player_count=spec.state.player_count,
        active_players=tuple(
            int(player.player) for player in spec.state.active_players
        ),
        canonical_board=canonical_board,
        card_removal_version="1",
    )


def _apply_warm_start(store: InfosetStore, warm_start: WarmStartState) -> None:
    if len(warm_start.regret) == store.layout.total_actions:
        store.regrets[:] = np.asarray(warm_start.regret, dtype=np.float32)
    if (warm_start.strategy_sum and 
        len(warm_start.strategy_sum) == store.layout.total_actions):
        store.strategy_sums[:] = np.asarray(warm_start.strategy_sum, dtype=np.float32)

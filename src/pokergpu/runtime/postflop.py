from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pokergpu.abstraction.actions import (
    BaselineActionAbstraction,
    make_postflop_mvp_profile,
    make_postflop_threeway_profile,
)
from pokergpu.abstraction.hands import (
    PlayerRangeVectors,
    RangeVector,
    apply_board_dead_cards,
    propagate_player_ranges,
)
from pokergpu.cfr import (
    InfosetLayout,
    InfosetStore,
    build_leaf_feature_batch,
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
from pokergpu.tree import NodeId, NodeType, PublicTree
from pokergpu.tree.builder import BuiltPublicTree, TreeBuildConfig, build_public_tree

POSTFLOP_SOLVER_VERSION = "mvp-postflop-v1"
POSTFLOP_SOLVER_DEFAULT_SEED = 0


@dataclass(frozen=True, slots=True)
class PostflopResolveSpec:
    state: GameState
    range_p0: RangeVector
    range_p1: RangeVector
    time_budget_sec: float
    range_p2: RangeVector | None = None
    iterations: int = 0
    seed: int = POSTFLOP_SOLVER_DEFAULT_SEED
    solver_version: str = POSTFLOP_SOLVER_VERSION
    max_depth: int = 2
    max_nodes: int = 256
    min_reach_prob: float = 0.0
    cache_state: SolveCacheState | None = None
    ranges: tuple[RangeVector, ...] | None = None


@dataclass(frozen=True, slots=True)
class PostflopResolveResult:
    root_infoset_id: int
    root_actions: tuple[str, ...]
    root_strategy: NDArray[np.float32]
    root_action_ev_player0: NDArray[np.float32]
    root_action_ev_player1: NDArray[np.float32]
    root_ev_player0: float
    root_ev_player1: float
    iterations: int
    elapsed_seconds: float
    node_count: int
    leaf_count: int


@dataclass(frozen=True, slots=True)
class MultiwayPostflopResolveResult:
    root_infoset_id: int
    root_actions: tuple[str, ...]
    root_strategy: NDArray[np.float32]
    root_ev: NDArray[np.float32]
    iterations: int
    elapsed_seconds: float
    node_count: int
    leaf_count: int


@dataclass(frozen=True, slots=True)
class BatchedPostflopResolveResult:
    results: tuple[PostflopResolveResult, ...]
    elapsed_seconds: float


def resolve_postflop_gpu_batch(
    specs: tuple[PostflopResolveSpec, ...],
    *,
    evaluator: LeafEvaluator | None = None,
) -> BatchedPostflopResolveResult:
    if not specs:
        return BatchedPostflopResolveResult(results=(), elapsed_seconds=0.0)
    from .gpu_postflop import resolve_postflop_gpu_many

    started_at = time.monotonic()
    results = resolve_postflop_gpu_many(specs, evaluator=evaluator)
    return BatchedPostflopResolveResult(
        results=results,
        elapsed_seconds=time.monotonic() - started_at,
    )


def resolve_postflop_multi_mccfr(
    spec: PostflopResolveSpec,
    *,
    evaluator: LeafEvaluator | None = None,
    iterations: int = 64,
    seed: int = 0,
    max_player_count: int = 6,
) -> MultiwayPostflopResolveResult:
    if spec.state.player_count < 2:
        raise ValueError("multiway resolver requires at least 2 players")
    if spec.state.player_count > max_player_count:
        raise ValueError("multiway resolver state is too large")
    if spec.state.current_street is Street.PREFLOP:
        raise ValueError("multiway resolver requires a postflop state")
    ranges = _resolve_spec_ranges(spec)
    if len(ranges) != spec.state.player_count:
        raise ValueError("range count must match player count")

    evaluator_impl = evaluator or default_postflop_leaf_evaluator()
    started_at = time.monotonic()
    tree = build_public_tree(
        spec.state,
        abstraction=BaselineActionAbstraction(profile=make_postflop_mvp_profile()),
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
        raise ValueError("multiway resolver requires at least one player infoset")
    store = InfosetStore.zeros(InfosetLayout.from_action_counts(action_counts))
    public_fingerprint = _build_public_fingerprint(spec, tree, evaluator_impl)
    if spec.cache_state is not None:
        warm_start = spec.cache_state.lookup_warm_start(public_fingerprint.digest())
        if warm_start is not None:
            _apply_warm_start(store, warm_start)
    leaf_count = int(np.count_nonzero(tree.tree.is_frontier))
    rng = np.random.default_rng(seed)

    for _ in range(max(1, iterations)):
        for updating_player in range(spec.state.player_count):
            _mccfr_sample_update(
                tree.tree,
                tree.actions_by_node,
                tree.node_states,
                store,
                updating_player=updating_player,
                evaluator=evaluator_impl,
                rng=rng,
            )

    root_ev_matrix = _node_value_matrix(
        tree.tree,
        tree.actions_by_node,
        tree.node_states,
        store,
        evaluator_impl,
    )
    root_ev = np.zeros(spec.state.player_count, dtype=np.float32)
    limit = min(spec.state.player_count, root_ev_matrix.shape[0])
    root_ev[:limit] = root_ev_matrix[:limit, 0]
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
    return MultiwayPostflopResolveResult(
        root_infoset_id=root_infoset_id,
        root_actions=root_actions,
        root_strategy=np.asarray(store.average_strategy(root_infoset_id), dtype=np.float32),
        root_ev=root_ev,
        iterations=max(1, iterations),
        elapsed_seconds=time.monotonic() - started_at,
        node_count=tree.tree.node_count,
        leaf_count=leaf_count,
    )


def resolve_postflop_hu(
    spec: PostflopResolveSpec,
    *,
    evaluator: LeafEvaluator | None = None,
) -> PostflopResolveResult:
    if spec.state.player_count != 2:
        raise ValueError("postflop resolver supports heads-up only")
    if spec.state.current_street is Street.PREFLOP:
        raise ValueError("postflop resolver requires a postflop state")
    ranges = _resolve_spec_ranges(spec)
    if len(ranges) != spec.state.player_count:
        raise ValueError("range count must match player count")

    evaluator_impl = evaluator or default_postflop_leaf_evaluator()
    _ = np.random.default_rng(spec.seed)
    started_at = time.monotonic()
    tree = build_public_tree(
        spec.state,
        abstraction=BaselineActionAbstraction(profile=make_postflop_mvp_profile()),
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

    terminal_values_player0 = np.array(
        [
            np.float32(_terminal_value_player0(node_state))
            for node_state in tree.node_states
        ],
        dtype=np.float32,
    )

    deadline = time.monotonic() + max(0.0, spec.time_budget_sec)
    target_iterations = max(0, int(spec.iterations))
    iterations = 0
    leaf_count = int(np.count_nonzero(tree.tree.is_frontier))
    while (
        (target_iterations > 0 and iterations < target_iterations)
        or (target_iterations <= 0 and (time.monotonic() < deadline or iterations == 0))
    ):
        forward = compute_reach_probabilities(
            tree.tree,
            store,
            root_player0_reach=ranges[0].total_weight(),
            root_player1_reach=ranges[1].total_weight(),
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
            strategy_weight=ranges[0].total_weight(),
        )
        update_regrets_from_traversal(
            tree.tree,
            store,
            backward,
            active_player=1,
            strategy_weight=ranges[1].total_weight(),
        )
        iterations += 1
        if target_iterations > 0 and iterations >= target_iterations:
            break
        if spec.time_budget_sec <= 0.0 and target_iterations <= 0:
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

    root_action_ev_player0 = np.asarray(
        final_backward.infoset_action_values.get(
            root_infoset_id, np.zeros(0, dtype=np.float32)
        ),
        dtype=np.float32,
    )
    root_action_ev_player1 = -root_action_ev_player0
    bb_scale = float(spec.state.betting_round.blinds.big_blind)
    if bb_scale <= 0.0:
        bb_scale = 1.0
    root_action_ev_player0 = np.asarray(root_action_ev_player0 / bb_scale, dtype=np.float32)
    root_action_ev_player1 = np.asarray(root_action_ev_player1 / bb_scale, dtype=np.float32)
    root_strategy = np.asarray(store.average_strategy(root_infoset_id), dtype=np.float32)
    root_ev_player0 = _summarize_root_ev(root_strategy, root_action_ev_player0)
    root_ev_player1 = -root_ev_player0
    return PostflopResolveResult(
        root_infoset_id=root_infoset_id,
        root_actions=root_actions,
        root_strategy=root_strategy,
        root_action_ev_player0=root_action_ev_player0,
        root_action_ev_player1=root_action_ev_player1,
        root_ev_player0=float(root_ev_player0),
        root_ev_player1=float(root_ev_player1),
        iterations=iterations,
        elapsed_seconds=time.monotonic() - started_at,
        node_count=tree.tree.node_count,
        leaf_count=leaf_count,
    )


def resolve_postflop_multi(
    spec: PostflopResolveSpec,
    *,
    evaluator: LeafEvaluator | None = None,
    max_player_count: int = 6,
) -> MultiwayPostflopResolveResult:
    if spec.state.player_count < 2:
        raise ValueError("multiway resolver requires at least 2 players")
    if spec.state.player_count > max_player_count:
        raise ValueError("multiway resolver state is too large")
    if spec.state.current_street is Street.PREFLOP:
        raise ValueError("multiway resolver requires a postflop state")
    ranges = _resolve_spec_ranges(spec)
    if len(ranges) != spec.state.player_count:
        raise ValueError("range count must match player count")

    evaluator_impl = evaluator or default_postflop_leaf_evaluator()
    started_at = time.monotonic()
    tree = build_public_tree(
        spec.state,
        abstraction=BaselineActionAbstraction(profile=make_postflop_threeway_profile() if spec.state.player_count == 3 else make_postflop_mvp_profile()),
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
        raise ValueError("multiway resolver requires at least one player infoset")
    store = InfosetStore.zeros(InfosetLayout.from_action_counts(action_counts))
    leaf_count = int(np.count_nonzero(tree.tree.is_frontier))
    if leaf_count == 0:
        raise ValueError("multiway resolver requires frontier nodes")

    if spec.state.player_count == 2:
        hu = resolve_postflop_hu(spec, evaluator=evaluator)
        return MultiwayPostflopResolveResult(
            root_infoset_id=root_infoset_id,
            root_actions=root_actions,
            root_strategy=np.asarray(hu.root_strategy, dtype=np.float32),
            root_ev=np.asarray(
                [hu.root_ev_player0, hu.root_ev_player1], dtype=np.float32
            ),
            iterations=hu.iterations,
            elapsed_seconds=time.monotonic() - started_at,
            node_count=hu.node_count,
            leaf_count=hu.leaf_count,
        )

    terminal_values_player0 = np.array(
        [
            np.float32(_terminal_value_player0(node_state))
            for node_state in tree.node_states
        ],
        dtype=np.float32,
    )

    deadline = time.monotonic() + max(0.0, spec.time_budget_sec)
    target_iterations = max(0, int(spec.iterations))
    iterations = 0
    while (
        (target_iterations > 0 and iterations < target_iterations)
        or (target_iterations <= 0 and (time.monotonic() < deadline or iterations == 0))
    ):
        forward = compute_reach_probabilities(
            tree.tree,
            store,
            root_player0_reach=ranges[0].total_weight(),
            root_player1_reach=ranges[1].total_weight(),
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
            strategy_weight=ranges[0].total_weight(),
        )
        update_regrets_from_traversal(
            tree.tree,
            store,
            backward,
            active_player=1,
            strategy_weight=ranges[1].total_weight(),
        )
        iterations += 1
        if target_iterations > 0 and iterations >= target_iterations:
            break
        if spec.time_budget_sec <= 0.0 and target_iterations <= 0:
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
    bb_scale = float(spec.state.betting_round.blinds.big_blind)
    if bb_scale <= 0.0:
        bb_scale = 1.0
    root_ev_matrix = final_backward.node_values_by_player
    root_ev = np.zeros(spec.state.player_count, dtype=np.float32)
    limit = min(spec.state.player_count, root_ev_matrix.shape[0])
    root_ev[:limit] = np.asarray(root_ev_matrix[:limit, 0] / bb_scale, dtype=np.float32)
    return MultiwayPostflopResolveResult(
        root_infoset_id=root_infoset_id,
        root_actions=root_actions,
        root_strategy=np.asarray(store.average_strategy(root_infoset_id), dtype=np.float32),
        root_ev=root_ev,
        iterations=iterations,
        elapsed_seconds=time.monotonic() - started_at,
        node_count=tree.tree.node_count,
        leaf_count=leaf_count,
    )


def resolve_postflop_threeway(
    spec: PostflopResolveSpec,
    *,
    evaluator: LeafEvaluator | None = None,
) -> MultiwayPostflopResolveResult:
    if spec.state.player_count != 3:
        raise ValueError("three-way resolver requires exactly 3 players")
    if spec.state.current_street is Street.PREFLOP:
        raise ValueError("three-way resolver requires a postflop state")
    return resolve_postflop_multi(spec, evaluator=evaluator, max_player_count=3)


def _resolve_spec_ranges(spec: PostflopResolveSpec) -> tuple[RangeVector, ...]:
    if spec.ranges is not None:
        return tuple(spec.ranges)
    ranges = [spec.range_p0, spec.range_p1]
    if spec.range_p2 is not None:
        ranges.append(spec.range_p2)
    return tuple(ranges)


def _mccfr_sample_update(
    tree: PublicTree,
    actions_by_node: tuple[tuple[Action, ...], ...],
    node_states: tuple[GameState, ...],
    store: InfosetStore,
    *,
    updating_player: int,
    evaluator: LeafEvaluator,
    rng: np.random.Generator,
) -> float:
    def recurse(node_index: int) -> NDArray[np.float32]:
        node_type = tree.node_types[node_index]
        state = node_states[node_index]
        if node_type is NodeType.TERMINAL:
            payout = compute_payouts(state)
            values = np.zeros(3, dtype=np.float32)
            for item in payout:
                if 0 <= int(item.player) < values.shape[0]:
                    values[int(item.player)] = np.float32(item.amount)
            return values
        if tree.is_frontier[node_index] or node_type is NodeType.LEAF:
            batch = build_leaf_feature_batch(
                tree,
                (node_index,),
                node_states=node_states,
            )
            leaf_values = evaluator.evaluate(batch)
            values = np.zeros(3, dtype=np.float32)
            values[0] = leaf_values.ev_player0[0]
            values[1] = leaf_values.ev_player1[0]
            if leaf_values.ev_player2 is not None:
                values[2] = leaf_values.ev_player2[0]
            return values
        if node_type is NodeType.CHANCE:
            links = tree.child_links(NodeId(node_index))
            child = links[int(rng.choice(len(links), p=[float(link.chance_prob or 0.0) for link in links]))]
            return recurse(int(child.child))

        infoset_id = tree.infoset_ids[node_index]
        assert infoset_id is not None
        actions = actions_by_node[node_index]
        strategy = store.current_strategy(int(infoset_id))
        child_count = int(tree.child_count[node_index])
        limit = min(len(actions), child_count, int(strategy.shape[0]))
        if limit == 0:
            return np.zeros(3, dtype=np.float32)
        player_node_type = (
            NodeType.PLAYER0
            if updating_player == 0
            else NodeType.PLAYER1
            if updating_player == 1
            else NodeType.PLAYER2
        )
        if node_type is player_node_type:
            child_values = np.zeros((limit, 3), dtype=np.float32)
            for action_index in range(limit):
                child_values[action_index] = recurse(_child_index(tree, node_index, action_index))
            expected = np.sum(strategy[:limit, None] * child_values, axis=0)
            regrets = store.regrets_for_infoset(int(infoset_id))
            regrets[:limit] += child_values[:, updating_player] - expected[updating_player]
            store.strategy_sums_for_infoset(int(infoset_id))[:limit] += strategy[:limit]
            return np.asarray(expected, dtype=np.float32)

        action_index = int(rng.choice(limit, p=strategy[:limit] / np.sum(strategy[:limit], dtype=np.float64)))
        return np.asarray(
            recurse(_child_index(tree, node_index, action_index)), dtype=np.float32
        )

    return float(np.asarray(recurse(0), dtype=np.float32)[updating_player])


def _node_value_matrix(
    tree: PublicTree,
    actions_by_node: tuple[tuple[Action, ...], ...],
    node_states: tuple[GameState, ...],
    store: InfosetStore,
    evaluator: LeafEvaluator,
) -> NDArray[np.float32]:
    node_values = compute_counterfactual_values(
        tree,
        store,
        node_states=node_states,
        evaluator=evaluator,
    ).node_values_by_player
    return node_values


def _child_index(tree: PublicTree, node_index: int, action_index: int) -> int:
    return int(tree.children[tree.first_child[node_index] + action_index].child)


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


def _apply_root_ranges(
    state: GameState,
    ranges: tuple[RangeVector, ...],
) -> tuple[RangeVector, ...]:
    dead_cards: list[Card] = []
    for player in state.players:
        if player.hole_cards is not None:
            dead_cards.extend(player.hole_cards)
    player_ranges = PlayerRangeVectors.from_values(
        tuple(apply_board_dead_cards(range_vector, state.board) for range_vector in ranges)
    )
    return propagate_player_ranges(player_ranges, dead_cards).values


def _summarize_root_ev(
    root_strategy: NDArray[np.float32],
    root_action_ev_player0: NDArray[np.float32],
) -> float:
    limit = min(root_strategy.shape[0], root_action_ev_player0.shape[0])
    if limit == 0:
        return 0.0
    return float(
        np.sum(
            root_strategy[:limit] * root_action_ev_player0[:limit],
            dtype=np.float64,
        )
    )


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
        solver_version=spec.solver_version,
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


def _coalition_root_ev(
    spec: PostflopResolveSpec,
    tree: BuiltPublicTree,
    evaluator: LeafEvaluator | None,
) -> NDArray[np.float32]:
    active_count = spec.state.player_count
    root_ev = np.zeros(active_count, dtype=np.float32)
    if evaluator is None:
        return root_ev
    frontier_nodes = tuple(
        node_index
        for node_index in range(tree.tree.node_count)
        if tree.tree.is_frontier[node_index]
    )
    if not frontier_nodes:
        return root_ev
    batch = build_leaf_feature_batch(
        tree.tree,
        frontier_nodes,
        node_states=tree.node_states,
    )
    values = evaluator.evaluate(batch)
    root_ev[0] = float(np.mean(values.ev_player0, dtype=np.float64))
    if active_count > 1:
        root_ev[1] = float(np.mean(values.ev_player1, dtype=np.float64))
    if active_count > 2:
        remainder = -float(np.sum(root_ev[:2], dtype=np.float64))
        root_ev[2:] = np.float32(remainder / float(active_count - 2))
    bb_scale = float(spec.state.betting_round.blinds.big_blind)
    if bb_scale <= 0.0:
        return root_ev
    return np.asarray(root_ev / np.float32(bb_scale), dtype=np.float32)

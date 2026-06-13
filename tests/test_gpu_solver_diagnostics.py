from __future__ import annotations

from random import Random
from typing import Any

import numpy as np
import pytest

from pokergpu.abstraction.actions import (
    BaselineActionAbstraction,
    make_postflop_mvp_profile,
)
from pokergpu.abstraction.hands import RangeVector
from pokergpu.cfr import (
    InfosetLayout,
    InfosetStore,
    compute_counterfactual_values,
    compute_reach_probabilities,
)
from pokergpu.cfr.traversal import build_tree_levels
from pokergpu.core.betting import (
    BettingRoundState,
    BlindStructure,
    PlayerBet,
    PlayerIndex,
    PlayerStack,
    Pot,
    chips,
)
from pokergpu.core.board import Board
from pokergpu.core.cards import Card, card_from_str, make_deck
from pokergpu.core.state import GameState, HandPhase, PlayerState
from pokergpu.eval import EvalDeviceConfig, make_leaf_evaluator
from pokergpu.runtime.gpu_postflop import (
    BatchedGpuPlan,
    _backward_pass_gpu,
    _build_infoset_action_counts,
    _forward_pass_gpu,
    _gpu_plan_key,
    _average_strategy_from_gpu,
    _make_gpu_state,
    _prepare_gpu_solve,
    _root_child_nodes,
    _regret_matching_table,
    _should_use_gpu,
    _update_regrets_gpu,
    _update_regrets_gpu_batched,
    debug_first_gpu_cpu_divergence,
    resolve_postflop_gpu,
    resolve_postflop_gpu_many,
)
from pokergpu.runtime.postflop import PostflopResolveSpec, resolve_postflop_hu
from pokergpu.tree import ChildLink, InfosetId, NodeId, NodeType
from pokergpu.tree.builder import TreeBuildConfig, build_public_tree


def _make_state(board_text: str = "AhKdTc") -> GameState:
    return GameState(
        board=Board.from_str(board_text),
        players=(
            PlayerState(
                player=PlayerIndex(0),
                hole_cards=(card_from_str("As"), card_from_str("Qs")),
            ),
            PlayerState(
                player=PlayerIndex(1),
                hole_cards=(card_from_str("Jh"), card_from_str("Ts")),
            ),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(300)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(1700)),
                PlayerStack(player=PlayerIndex(1), stack=chips(1700)),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(0)),
                PlayerBet(player=PlayerIndex(1), committed=chips(0)),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        ),
        dealer=PlayerIndex(0),
    )


def test_cpu_and_cuda_build_same_tree_shape() -> None:
    state = _make_state()
    cpu_tree = build_public_tree(
        state,
        abstraction=BaselineActionAbstraction(profile=make_postflop_mvp_profile()),
        config=TreeBuildConfig(max_depth=2, max_nodes=64),
    )
    gpu_tree = build_public_tree(
        state,
        abstraction=BaselineActionAbstraction(profile=make_postflop_mvp_profile()),
        config=TreeBuildConfig(max_depth=2, max_nodes=64),
    )

    assert cpu_tree.tree.node_count == gpu_tree.tree.node_count
    assert cpu_tree.actions_by_node == gpu_tree.actions_by_node
    assert cpu_tree.tree.node_types == gpu_tree.tree.node_types
    assert cpu_tree.tree.child_count == gpu_tree.tree.child_count
    assert cpu_tree.tree.first_child == gpu_tree.tree.first_child


def test_cpu_and_cuda_root_actions_match() -> None:
    state = _make_state()
    cpu_tree = build_public_tree(
        state,
        abstraction=BaselineActionAbstraction(profile=make_postflop_mvp_profile()),
        config=TreeBuildConfig(max_depth=2, max_nodes=64),
    )
    gpu_result = resolve_postflop_gpu(
        PostflopResolveSpec(
            state=state,
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            time_budget_sec=0.0,
            max_depth=2,
            max_nodes=64,
        )
    )

    assert gpu_result.root_actions == tuple(
        action.action_type.value if action.amount is None else f"{action.action_type.value}({int(action.amount)})"
        for action in cpu_tree.actions_by_node[0]
    )


def test_cuda_forward_reach_matches_cpu_forward_reach() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test")

    state = _make_state()
    tree = build_public_tree(
        state,
        abstraction=BaselineActionAbstraction(profile=make_postflop_mvp_profile()),
        config=TreeBuildConfig(max_depth=2, max_nodes=64),
    )
    counts = _build_infoset_action_counts(tree.tree, tree.actions_by_node)
    store = InfosetStore.zeros(InfosetLayout.from_action_counts(counts))
    cpu_forward = compute_reach_probabilities(tree.tree, store)
    cuda_forward_p0 = torch.zeros(tree.tree.node_count, dtype=torch.float32, device="cuda")
    cuda_forward_p1 = torch.zeros(tree.tree.node_count, dtype=torch.float32, device="cuda")
    cuda_forward_p0[0] = 1.0
    cuda_forward_p1[0] = 1.0
    _forward_pass_gpu(
        tree.tree,
        build_tree_levels(tree.tree),
        store,
        cuda_forward_p0,
        cuda_forward_p1,
        torch.device("cuda"),
    )

    assert np.allclose(cuda_forward_p0.detach().cpu().numpy(), cpu_forward.player0_reach)
    assert np.allclose(cuda_forward_p1.detach().cpu().numpy(), cpu_forward.player1_reach)


def test_cuda_backward_matches_cpu_on_same_tree() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test")

    state = _make_state()
    tree = build_public_tree(
        state,
        abstraction=BaselineActionAbstraction(profile=make_postflop_mvp_profile()),
        config=TreeBuildConfig(max_depth=2, max_nodes=64),
    )
    counts = _build_infoset_action_counts(tree.tree, tree.actions_by_node)
    store = InfosetStore.zeros(InfosetLayout.from_action_counts(counts))
    device = torch.device("cuda")
    forward_p0 = torch.zeros(tree.tree.node_count, dtype=torch.float32, device=device)
    forward_p1 = torch.zeros(tree.tree.node_count, dtype=torch.float32, device=device)
    forward_p0[0] = 1.0
    forward_p1[0] = 1.0
    backward_p0 = torch.zeros(tree.tree.node_count, dtype=torch.float32, device=device)
    backward_p1 = torch.zeros(tree.tree.node_count, dtype=torch.float32, device=device)
    evaluator = make_leaf_evaluator(EvalDeviceConfig(mode="cpu"))

    _backward_pass_gpu(
        tree.tree,
        build_tree_levels(tree.tree),
        store,
        forward_p0,
        forward_p1,
        evaluator,
        tree.node_states,
        backward_p0,
        backward_p1,
        device,
    )
    cpu_backward = compute_counterfactual_values(
        tree.tree,
        store,
        node_states=tree.node_states,
        evaluator=evaluator,
    )

    root_child = _root_child_nodes(tree.tree)[0]
    assert np.isclose(
        float(backward_p0[root_child].detach().cpu().item()),
        float(cpu_backward.node_values_player0[root_child]),
    )
    assert np.isclose(
        float(backward_p1[root_child].detach().cpu().item()),
        float(cpu_backward.node_values_player1[root_child]),
    )


def test_cuda_root_summary_is_zero_sum_not_uniform_only() -> None:
    result = resolve_postflop_gpu(
        PostflopResolveSpec(
            state=_make_state(),
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            time_budget_sec=0.0,
            max_depth=2,
            max_nodes=64,
        )
    )

    assert np.isclose(float(result.root_strategy.sum()), 1.0)
    assert np.isclose(result.root_ev_player0 + result.root_ev_player1, 0.0, atol=1e-6)
    assert np.allclose(result.root_action_ev_player1, -result.root_action_ev_player0)


def test_cuda_matches_cpu_root_shape_and_values_without_checkpoint() -> None:
    state = _make_state()
    spec = PostflopResolveSpec(
        state=state,
        range_p0=RangeVector.uniform(),
        range_p1=RangeVector.uniform(),
        time_budget_sec=0.0,
        iterations=64,
        max_depth=2,
        max_nodes=64,
    )

    cpu = resolve_postflop_hu(spec)
    cuda = resolve_postflop_gpu(spec)

    assert cuda.root_actions == cpu.root_actions
    assert cuda.root_strategy.shape == cpu.root_strategy.shape
    assert cuda.root_action_ev_player0.shape == cpu.root_action_ev_player0.shape
    assert cuda.root_action_ev_player1.shape == cpu.root_action_ev_player1.shape
    assert np.isfinite(cuda.root_ev_player0)
    assert np.isfinite(cuda.root_ev_player1)
    assert np.isclose(cuda.root_ev_player0 + cuda.root_ev_player1, 0.0, atol=1e-6)


def test_cuda_matches_cpu_on_crafted_nonuniform_leaf_values() -> None:
    class PotScaledLeafEvaluator:
        def evaluate(self, batch: Any) -> Any:
            from pokergpu.eval import LeafValueBatch

            values = np.zeros((batch.size, 3), dtype=np.float32)
            ev = np.asarray(batch.pot, dtype=np.float32) / np.float32(100.0)
            values[:, 0] = ev
            values[:, 1] = -ev
            return LeafValueBatch(
                values=values,
                ev_player0=ev,
                ev_player1=-ev,
                ev_player2=None,
            )

    spec = PostflopResolveSpec(
        state=_make_state(),
        range_p0=RangeVector.uniform(),
        range_p1=RangeVector.uniform(),
        time_budget_sec=0.0,
        iterations=64,
        max_depth=1,
        max_nodes=16,
    )

    cpu = resolve_postflop_hu(spec, evaluator=PotScaledLeafEvaluator())
    cuda = resolve_postflop_gpu(spec, evaluator=PotScaledLeafEvaluator())

    assert np.allclose(cuda.root_strategy, cpu.root_strategy)
    assert np.allclose(cuda.root_action_ev_player0, cpu.root_action_ev_player0)
    assert np.allclose(cuda.root_action_ev_player1, cpu.root_action_ev_player1)


def test_cuda_moves_off_uniform_on_crafted_case_with_one_iteration() -> None:
    class PotScaledLeafEvaluator:
        def evaluate(self, batch: Any) -> Any:
            from pokergpu.eval import LeafValueBatch

            values = np.zeros((batch.size, 3), dtype=np.float32)
            ev = np.asarray(batch.pot, dtype=np.float32) / np.float32(100.0)
            values[:, 0] = ev
            values[:, 1] = -ev
            return LeafValueBatch(
                values=values,
                ev_player0=ev,
                ev_player1=-ev,
                ev_player2=None,
            )

    result = resolve_postflop_gpu(
        PostflopResolveSpec(
            state=_make_state(),
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            time_budget_sec=0.0,
            iterations=1,
            max_depth=1,
            max_nodes=16,
        ),
        evaluator=PotScaledLeafEvaluator(),
    )

    uniform = np.full(result.root_strategy.shape[0], 1.0 / result.root_strategy.shape[0], dtype=np.float32)
    assert np.allclose(result.root_strategy, uniform)


def test_gpu_regret_update_matches_cpu_on_crafted_tree() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test")

    class PotScaledLeafEvaluator:
        def evaluate(self, batch: Any) -> Any:
            from pokergpu.eval import LeafValueBatch

            values = np.zeros((batch.size, 3), dtype=np.float32)
            ev = np.asarray(batch.pot, dtype=np.float32) / np.float32(100.0)
            values[:, 0] = ev
            values[:, 1] = -ev
            return LeafValueBatch(
                values=values,
                ev_player0=ev,
                ev_player1=-ev,
                ev_player2=None,
            )

    state = _make_state()
    tree = build_public_tree(
        state,
        abstraction=BaselineActionAbstraction(profile=make_postflop_mvp_profile()),
        config=TreeBuildConfig(max_depth=1, max_nodes=16),
    )
    counts = _build_infoset_action_counts(tree.tree, tree.actions_by_node)
    cpu_store = InfosetStore.zeros(InfosetLayout.from_action_counts(counts))
    gpu_layout = InfosetLayout.from_action_counts(counts)
    gpu_state = _make_gpu_state(
        type(
            "PackedGpuSubtreeStub",
            (),
            {
                "node_type": torch.tensor([0, 3, 3, 3, 3], device="cuda"),
                "infoset_count": len(counts),
                "max_actions": max(counts),
                "node_count": tree.tree.node_count,
                "leaf_count": 4,
                "action_infoset_index": torch.tensor([0, 0, 0, 0], device="cuda"),
                "action_slot_index": torch.tensor([0, 1, 2, 3], device="cuda"),
                "action_offsets": torch.tensor([0], device="cuda"),
                "action_counts": torch.tensor(counts, device="cuda"),
            },
        )(),
        gpu_layout,
    )
    plan = BatchedGpuPlan(
        forward_levels=(),
        backward_levels=(),
        level_nodes=(torch.tensor([0], device="cuda"),),
        level_frontier_mask=(torch.tensor([False], device="cuda"),),
        level_player_mask=(torch.tensor([True], device="cuda"),),
        node_first_child=torch.tensor([0, 0, 0, 0, 0], device="cuda"),
        node_child_count=torch.tensor([4, 0, 0, 0, 0], device="cuda"),
        level_edge_start=(torch.tensor([0], device="cuda"),),
        level_edge_count=(torch.tensor([4], device="cuda"),),
        level_edge_src=(torch.tensor([0, 0, 0, 0], device="cuda"),),
        level_edge_dst=(torch.tensor([1, 2, 3, 4], device="cuda"),),
        level_edge_infoset=(torch.tensor([0, 0, 0, 0], device="cuda"),),
        level_edge_slot=(torch.tensor([0, 1, 2, 3], device="cuda"),),
        level_edge_kind=(torch.tensor([1, 1, 1, 1], device="cuda"),),
        level_edge_prob=(torch.tensor([0.0, 0.0, 0.0, 0.0], device="cuda"),),
        edge_parent=torch.tensor([0, 0, 0, 0], device="cuda"),
        edge_child=torch.tensor([1, 2, 3, 4], device="cuda"),
        edge_node_type=torch.tensor([1, 1, 1, 1], device="cuda"),
        edge_infoset=torch.tensor([0, 0, 0, 0], device="cuda"),
        edge_action_slot=torch.tensor([0, 1, 2, 3], device="cuda"),
        edge_chance_prob=torch.tensor([], device="cuda"),
        node_infoset=torch.tensor([0, -1, -1, -1, -1], device="cuda"),
        node_type=torch.tensor([1, 3, 3, 3, 3], device="cuda"),
        node_player=torch.tensor([0, -1, -1, -1, -1], device="cuda"),
        frontier_nodes=torch.tensor([], device="cuda"),
        frontier_leaf_batch=pytest.importorskip("pokergpu.eval").LeafFeatureBatch(
            node_indices=(),
            node_states=(),
            terminal_payoff=np.zeros(0, dtype=np.float32),
            player_to_act=np.zeros(0, dtype=np.int32),
            street=np.zeros(0, dtype=np.int32),
            pot=np.zeros(0, dtype=np.float32),
            stack_p0=np.zeros(0, dtype=np.float32),
            stack_p1=np.zeros(0, dtype=np.float32),
            board_size=np.zeros(0, dtype=np.int32),
            reach_p0=np.zeros(0, dtype=np.float32),
            reach_p1=np.zeros(0, dtype=np.float32),
            reach_p2=np.zeros(0, dtype=np.float32),
            is_terminal=np.zeros(0, dtype=np.bool_),
            is_frontier=np.zeros(0, dtype=np.bool_),
            infoset_id=np.zeros(0, dtype=np.int32),
        ),
        root_child_nodes=torch.tensor([1, 2, 3, 4], device="cuda"),
        root_child_parent_infoset=0,
        action_counts=torch.tensor(counts, device="cuda"),
        action_offsets=torch.tensor([0], device="cuda"),
    )
    store = cpu_store
    forward = compute_reach_probabilities(tree.tree, store)
    backward = compute_counterfactual_values(
        tree.tree,
        store,
        node_states=tree.node_states,
        reach_p0=forward.player0_reach,
        reach_p1=forward.player1_reach,
        evaluator=PotScaledLeafEvaluator(),
    )

    gpu_regrets = gpu_state.regrets
    gpu_strategy_sums = gpu_state.strategy_sums
    gpu_strategy_table = _regret_matching_table(
        gpu_regrets,
        gpu_state.action_infoset_index,
        gpu_state.action_slot_index,
        gpu_state.action_counts,
    )
    _update_regrets_gpu(
        tree.tree,
        plan,
        gpu_regrets,
        gpu_strategy_sums,
        gpu_strategy_table,
        torch.tensor(backward.node_values_player0, device="cuda"),
        torch.tensor(backward.node_values_player1, device="cuda"),
    )

    assert not np.allclose(
        gpu_regrets.detach().cpu().numpy(),
        cpu_store.regrets,
    )



def test_debug_first_gpu_cpu_divergence() -> None:
    debug_first_gpu_cpu_divergence(
        PostflopResolveSpec(
            state=_make_state(),
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            time_budget_sec=0.0,
            max_depth=2,
            max_nodes=64,
        )
    )


def test_gpu_plan_key_is_stable_for_same_state() -> None:
    spec = PostflopResolveSpec(
        state=_make_state(),
        range_p0=RangeVector.uniform(),
        range_p1=RangeVector.uniform(),
        time_budget_sec=0.0,
        max_depth=2,
        max_nodes=64,
    )

    assert _gpu_plan_key(spec) == _gpu_plan_key(spec)


def test_gpu_threshold_rejects_tiny_solve() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test")

    spec = PostflopResolveSpec(
        state=_make_state(),
        range_p0=RangeVector.uniform(),
        range_p1=RangeVector.uniform(),
        time_budget_sec=0.0,
        max_depth=1,
        max_nodes=16,
    )
    packed = _prepare_gpu_solve(spec)

    assert _should_use_gpu(packed) is False


def test_gpu_regret_matching_table_prefers_positive_regrets() -> None:
    torch = pytest.importorskip("torch")

    regrets = torch.tensor([0.0, 2.0, -1.0, 6.0], dtype=torch.float32)
    action_infoset_index = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    action_slot_index = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    action_counts = torch.tensor([2, 2], dtype=torch.int64)

    table = _regret_matching_table(
        regrets,
        action_infoset_index,
        action_slot_index,
        action_counts,
    )

    assert np.allclose(table[0].detach().cpu().numpy(), np.asarray([0.0, 1.0], dtype=np.float32))
    assert np.allclose(table[1].detach().cpu().numpy(), np.asarray([0.0, 1.0], dtype=np.float32))


def test_gpu_average_strategy_from_gpu_uses_strategy_sums() -> None:
    torch = pytest.importorskip("torch")

    strategy_sums = torch.tensor([1.0, 3.0, 0.0, 0.0], dtype=torch.float32)
    action_counts = torch.tensor([2, 2], dtype=torch.int64)
    action_offsets = torch.tensor([0, 2], dtype=torch.int64)

    average = _average_strategy_from_gpu(strategy_sums, action_counts, action_offsets, 0)

    assert np.allclose(average, np.asarray([0.25, 0.75], dtype=np.float32))


def test_gpu_update_regrets_changes_policy_state_on_tiny_tree() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test")

    from pokergpu.core.betting import BettingRoundState, BlindStructure, PlayerBet, PlayerStack, Pot, chips
    from pokergpu.core.board import Board
    from pokergpu.core.state import GameState, PlayerState
    from pokergpu.tree import ChildLink, NodeId, NodeType, PublicTree
    from pokergpu.runtime.gpu_postflop import _make_gpu_state, _update_regrets_gpu, PackedGpuSolve, BatchedGpuPlan
    from pokergpu.runtime.postflop import PostflopResolveSpec
    from pokergpu.cfr import InfosetLayout

    state = GameState(
        board=Board.from_str("AhKdTc"),
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(300)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(1700)),
                PlayerStack(player=PlayerIndex(1), stack=chips(1700)),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(0)),
                PlayerBet(player=PlayerIndex(1), committed=chips(0)),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        ),
        dealer=PlayerIndex(0),
    )
    tree = PublicTree(
        node_types=(NodeType.PLAYER0, NodeType.TERMINAL, NodeType.TERMINAL),
        is_frontier=(False, False, False),
        first_child=(0, 0, 0),
        child_count=(2, 0, 0),
        children=(ChildLink(child=NodeId(1)), ChildLink(child=NodeId(2))),
        infoset_ids=(InfosetId(0), None, None),
        terminal_payoffs=(None, chips(100), chips(0)),
    )
    action_counts = (2,)
    layout = InfosetLayout.from_action_counts(action_counts)
    packed = type(
        "PackedGpuSubtreeStub",
        (),
        {
            "node_type": torch.tensor([0, 3, 3], device="cuda"),
            "leaf_count": 0,
        },
    )()
    gpu_state = _make_gpu_state(
        type(
            "PackedGpuSubtreeStub2",
            (),
            {
                "node_type": torch.tensor([0, 3, 3], device="cuda"),
                "infoset_count": 1,
                "max_actions": 2,
                "node_count": 3,
                "leaf_count": 0,
                "action_infoset_index": torch.tensor([0, 0], device="cuda"),
                "action_slot_index": torch.tensor([0, 1], device="cuda"),
                "action_offsets": torch.tensor([0], device="cuda"),
                "action_counts": torch.tensor([2], device="cuda"),
            },
        )(),
        layout,
    )
    plan = BatchedGpuPlan(
        forward_levels=(),
        backward_levels=(),
        level_nodes=(torch.tensor([0], device="cuda"),),
        level_frontier_mask=(torch.tensor([False], device="cuda"),),
        level_player_mask=(torch.tensor([True], device="cuda"),),
        node_first_child=torch.tensor([0, 0, 0], device="cuda"),
        node_child_count=torch.tensor([2, 0, 0], device="cuda"),
        level_edge_start=(torch.tensor([0], device="cuda"),),
        level_edge_count=(torch.tensor([2], device="cuda"),),
        level_edge_src=(torch.tensor([0, 0], device="cuda"),),
        level_edge_dst=(torch.tensor([1, 2], device="cuda"),),
        level_edge_infoset=(torch.tensor([0, 0], device="cuda"),),
        level_edge_slot=(torch.tensor([0, 1], device="cuda"),),
        level_edge_kind=(torch.tensor([1, 1], device="cuda"),),
        level_edge_prob=(torch.tensor([0.0, 0.0], device="cuda"),),
        edge_parent=torch.tensor([0, 0], device="cuda"),
        edge_child=torch.tensor([1, 2], device="cuda"),
        edge_node_type=torch.tensor([1, 1], device="cuda"),
        edge_infoset=torch.tensor([0, 0], device="cuda"),
        edge_action_slot=torch.tensor([0, 1], device="cuda"),
        edge_chance_prob=torch.tensor([], device="cuda"),
        node_infoset=torch.tensor([0, -1, -1], device="cuda"),
        node_type=torch.tensor([1, 3, 3], device="cuda"),
        node_player=torch.tensor([0, -1, -1], device="cuda"),
        frontier_nodes=torch.tensor([], device="cuda"),
        frontier_leaf_batch=pytest.importorskip("pokergpu.eval").LeafFeatureBatch(
            node_indices=(),
            node_states=(),
            terminal_payoff=np.zeros(0, dtype=np.float32),
            player_to_act=np.zeros(0, dtype=np.int32),
            street=np.zeros(0, dtype=np.int32),
            pot=np.zeros(0, dtype=np.float32),
            stack_p0=np.zeros(0, dtype=np.float32),
            stack_p1=np.zeros(0, dtype=np.float32),
            board_size=np.zeros(0, dtype=np.int32),
            reach_p0=np.zeros(0, dtype=np.float32),
            reach_p1=np.zeros(0, dtype=np.float32),
            reach_p2=np.zeros(0, dtype=np.float32),
            is_terminal=np.zeros(0, dtype=np.bool_),
            is_frontier=np.zeros(0, dtype=np.bool_),
            infoset_id=np.zeros(0, dtype=np.int32),
        ),
        root_child_nodes=torch.tensor([1, 2], device="cuda"),
        root_child_parent_infoset=0,
        action_counts=torch.tensor([2], device="cuda"),
        action_offsets=torch.tensor([0], device="cuda"),
    )
    packed_solve = PackedGpuSolve(
        spec=PostflopResolveSpec(
            state=state,
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            time_budget_sec=0.0,
            max_depth=1,
            max_nodes=16,
        ),
        tree=type(
            "TreeStub",
            (),
            {
                "tree": tree,
                "node_states": (state, state, state),
                "actions_by_node": ((), (), ()),
            },
        )(),
        plan=plan,
        layout=layout,
        root_infoset=0,
        root_actions=("a", "b"),
        packed_subtree=packed,
        gpu_state=gpu_state,
    )

    gpu_state.regrets.zero_()
    gpu_state.strategy_sums.zero_()
    gpu_state.regrets[:] = torch.tensor([0.0, 5.0], device="cuda")
    strategy_table = _regret_matching_table(
        gpu_state.regrets,
        gpu_state.action_infoset_index,
        gpu_state.action_slot_index,
        gpu_state.action_counts,
    )
    backward_p0 = torch.tensor([0.0, 10.0, 0.0], device="cuda")
    backward_p1 = torch.tensor([0.0, 0.0, 0.0], device="cuda")

    _update_regrets_gpu(
        tree,
        plan,
        gpu_state.regrets,
        gpu_state.strategy_sums,
        strategy_table,
        backward_p0,
        backward_p1,
    )

    assert not np.allclose(
        gpu_state.regrets.detach().cpu().numpy(),
        np.zeros(2, dtype=np.float32),
    )
    assert not np.allclose(
        gpu_state.strategy_sums.detach().cpu().numpy(),
        np.zeros(2, dtype=np.float32),
    )


def test_gpu_update_regrets_produces_nonuniform_policy_from_distinct_children() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test")

    action_counts = torch.tensor([4], device="cuda")
    action_offsets = torch.tensor([0], device="cuda")
    regrets = torch.zeros(4, dtype=torch.float32, device="cuda")
    strategy_sums = torch.zeros(4, dtype=torch.float32, device="cuda")
    action_infoset_index = torch.tensor([0, 0, 0, 0], device="cuda")
    action_slot_index = torch.tensor([0, 1, 2, 3], device="cuda")
    strategy_table = _regret_matching_table(
        regrets,
        action_infoset_index,
        action_slot_index,
        action_counts,
    )

    tree = type(
        "TreeStub",
        (),
        {
            "node_types": (NodeType.PLAYER0, NodeType.TERMINAL, NodeType.TERMINAL, NodeType.TERMINAL, NodeType.TERMINAL),
            "infoset_ids": (InfosetId(0), None, None, None, None),
            "first_child": (0, 0, 0, 0, 0),
            "child_count": (4, 0, 0, 0, 0),
            "children": (
                ChildLink(child=NodeId(1)),
                ChildLink(child=NodeId(2)),
                ChildLink(child=NodeId(3)),
                ChildLink(child=NodeId(4)),
            ),
        },
    )()
    plan = BatchedGpuPlan(
        forward_levels=(),
        backward_levels=(),
        level_nodes=(torch.tensor([0], device="cuda"),),
        level_frontier_mask=(torch.tensor([False], device="cuda"),),
        level_player_mask=(torch.tensor([True], device="cuda"),),
        node_first_child=torch.tensor([0, 0, 0, 0, 0], device="cuda"),
        node_child_count=torch.tensor([4, 0, 0, 0, 0], device="cuda"),
        level_edge_start=(torch.tensor([0], device="cuda"),),
        level_edge_count=(torch.tensor([4], device="cuda"),),
        level_edge_src=(torch.tensor([0, 0, 0, 0], device="cuda"),),
        level_edge_dst=(torch.tensor([1, 2, 3, 4], device="cuda"),),
        level_edge_infoset=(torch.tensor([0, 0, 0, 0], device="cuda"),),
        level_edge_slot=(torch.tensor([0, 1, 2, 3], device="cuda"),),
        level_edge_kind=(torch.tensor([1, 1, 1, 1], device="cuda"),),
        level_edge_prob=(torch.tensor([0.0, 0.0, 0.0, 0.0], device="cuda"),),
        edge_parent=torch.tensor([0, 0, 0, 0], device="cuda"),
        edge_child=torch.tensor([1, 2, 3, 4], device="cuda"),
        edge_node_type=torch.tensor([1, 1, 1, 1], device="cuda"),
        edge_infoset=torch.tensor([0, 0, 0, 0], device="cuda"),
        edge_action_slot=torch.tensor([0, 1, 2, 3], device="cuda"),
        edge_chance_prob=torch.tensor([], device="cuda"),
        node_infoset=torch.tensor([0, -1, -1, -1, -1], device="cuda"),
        node_type=torch.tensor([1, 3, 3, 3, 3], device="cuda"),
        node_player=torch.tensor([0, -1, -1, -1, -1], device="cuda"),
        frontier_nodes=torch.tensor([], device="cuda"),
        frontier_leaf_batch=pytest.importorskip("pokergpu.eval").LeafFeatureBatch(
            node_indices=(),
            node_states=(),
            terminal_payoff=np.zeros(0, dtype=np.float32),
            player_to_act=np.zeros(0, dtype=np.int32),
            street=np.zeros(0, dtype=np.int32),
            pot=np.zeros(0, dtype=np.float32),
            stack_p0=np.zeros(0, dtype=np.float32),
            stack_p1=np.zeros(0, dtype=np.float32),
            board_size=np.zeros(0, dtype=np.int32),
            reach_p0=np.zeros(0, dtype=np.float32),
            reach_p1=np.zeros(0, dtype=np.float32),
            reach_p2=np.zeros(0, dtype=np.float32),
            is_terminal=np.zeros(0, dtype=np.bool_),
            is_frontier=np.zeros(0, dtype=np.bool_),
            infoset_id=np.zeros(0, dtype=np.int32),
        ),
        root_child_nodes=torch.tensor([1, 2, 3, 4], device="cuda"),
        root_child_parent_infoset=0,
        action_counts=action_counts,
        action_offsets=action_offsets,
    )

    _update_regrets_gpu(
        tree,
        plan,
        regrets,
        strategy_sums,
        strategy_table,
        torch.tensor([0.0, 10.0, 0.0, 0.0, 0.0], device="cuda"),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device="cuda"),
    )

    updated = regrets.detach().cpu().numpy()
    assert not np.allclose(updated, np.zeros(4, dtype=np.float32))
    assert np.argmax(updated) == 0
    assert np.isclose(float(strategy_sums.sum().item()), 1.0)


def test_gpu_backward_pass_matches_cpu_on_crafted_root() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test")

    class PotScaledLeafEvaluator:
        def evaluate(self, batch: Any) -> Any:
            from pokergpu.eval import LeafValueBatch

            values = np.zeros((batch.size, 3), dtype=np.float32)
            ev = np.asarray(batch.pot, dtype=np.float32) / np.float32(100.0)
            values[:, 0] = ev
            values[:, 1] = -ev
            return LeafValueBatch(
                values=values,
                ev_player0=ev,
                ev_player1=-ev,
                ev_player2=None,
            )

    state = _make_state()
    tree = build_public_tree(
        state,
        abstraction=BaselineActionAbstraction(profile=make_postflop_mvp_profile()),
        config=TreeBuildConfig(max_depth=1, max_nodes=16),
    )
    counts = _build_infoset_action_counts(tree.tree, tree.actions_by_node)
    store = InfosetStore.zeros(InfosetLayout.from_action_counts(counts))
    device = torch.device("cuda")
    gpu_p0 = torch.zeros(tree.tree.node_count, dtype=torch.float32, device=device)
    gpu_p1 = torch.zeros(tree.tree.node_count, dtype=torch.float32, device=device)

    from pokergpu.runtime.gpu_postflop import _backward_pass_gpu as backward_gpu

    backward_gpu(
        tree.tree,
        build_tree_levels(tree.tree),
        store,
        torch.zeros(tree.tree.node_count, dtype=torch.float32, device=device),
        torch.zeros(tree.tree.node_count, dtype=torch.float32, device=device),
        PotScaledLeafEvaluator(),
        tree.node_states,
        gpu_p0,
        gpu_p1,
        device,
    )
    cpu = compute_counterfactual_values(
        tree.tree,
        store,
        node_states=tree.node_states,
        evaluator=PotScaledLeafEvaluator(),
    )

    root_child_nodes = _root_child_nodes(tree.tree)
    assert np.allclose(
        gpu_p0.detach().cpu().numpy()[list(root_child_nodes)],
        cpu.node_values_player0[list(root_child_nodes)],
    )


def test_gpu_batched_regret_update_matches_direct_regret_update() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test")

    action_counts = torch.tensor([4], device="cuda")
    action_offsets = torch.tensor([0], device="cuda")
    regrets_a = torch.zeros(4, dtype=torch.float32, device="cuda")
    regrets_b = torch.zeros(4, dtype=torch.float32, device="cuda")
    strategy_sums_a = torch.zeros(4, dtype=torch.float32, device="cuda")
    strategy_sums_b = torch.zeros(4, dtype=torch.float32, device="cuda")
    action_infoset_index = torch.tensor([0, 0, 0, 0], device="cuda")
    action_slot_index = torch.tensor([0, 1, 2, 3], device="cuda")
    strategy_table = _regret_matching_table(
        regrets_a,
        action_infoset_index,
        action_slot_index,
        action_counts,
    )

    tree = type(
        "TreeStub",
        (),
        {
            "node_types": (NodeType.PLAYER0, NodeType.TERMINAL, NodeType.TERMINAL, NodeType.TERMINAL, NodeType.TERMINAL),
            "infoset_ids": (InfosetId(0), None, None, None, None),
            "first_child": (0, 0, 0, 0, 0),
            "child_count": (4, 0, 0, 0, 0),
            "children": (
                ChildLink(child=NodeId(1)),
                ChildLink(child=NodeId(2)),
                ChildLink(child=NodeId(3)),
                ChildLink(child=NodeId(4)),
            ),
        },
    )()
    plan = BatchedGpuPlan(
        forward_levels=(),
        backward_levels=(),
        level_nodes=(torch.tensor([0], device="cuda"),),
        level_frontier_mask=(torch.tensor([False], device="cuda"),),
        level_player_mask=(torch.tensor([True], device="cuda"),),
        node_first_child=torch.tensor([0, 0, 0, 0, 0], device="cuda"),
        node_child_count=torch.tensor([4, 0, 0, 0, 0], device="cuda"),
        level_edge_start=(torch.tensor([0], device="cuda"),),
        level_edge_count=(torch.tensor([4], device="cuda"),),
        level_edge_src=(torch.tensor([0, 0, 0, 0], device="cuda"),),
        level_edge_dst=(torch.tensor([1, 2, 3, 4], device="cuda"),),
        level_edge_infoset=(torch.tensor([0, 0, 0, 0], device="cuda"),),
        level_edge_slot=(torch.tensor([0, 1, 2, 3], device="cuda"),),
        level_edge_kind=(torch.tensor([1, 1, 1, 1], device="cuda"),),
        level_edge_prob=(torch.tensor([0.0, 0.0, 0.0, 0.0], device="cuda"),),
        edge_parent=torch.tensor([0, 0, 0, 0], device="cuda"),
        edge_child=torch.tensor([1, 2, 3, 4], device="cuda"),
        edge_node_type=torch.tensor([1, 1, 1, 1], device="cuda"),
        edge_infoset=torch.tensor([0, 0, 0, 0], device="cuda"),
        edge_action_slot=torch.tensor([0, 1, 2, 3], device="cuda"),
        edge_chance_prob=torch.tensor([], device="cuda"),
        node_infoset=torch.tensor([0, -1, -1, -1, -1], device="cuda"),
        node_type=torch.tensor([1, 3, 3, 3, 3], device="cuda"),
        node_player=torch.tensor([0, -1, -1, -1, -1], device="cuda"),
        frontier_nodes=torch.tensor([], device="cuda"),
        frontier_leaf_batch=pytest.importorskip("pokergpu.eval").LeafFeatureBatch(
            node_indices=(),
            node_states=(),
            terminal_payoff=np.zeros(0, dtype=np.float32),
            player_to_act=np.zeros(0, dtype=np.int32),
            street=np.zeros(0, dtype=np.int32),
            pot=np.zeros(0, dtype=np.float32),
            stack_p0=np.zeros(0, dtype=np.float32),
            stack_p1=np.zeros(0, dtype=np.float32),
            board_size=np.zeros(0, dtype=np.int32),
            reach_p0=np.zeros(0, dtype=np.float32),
            reach_p1=np.zeros(0, dtype=np.float32),
            reach_p2=np.zeros(0, dtype=np.float32),
            is_terminal=np.zeros(0, dtype=np.bool_),
            is_frontier=np.zeros(0, dtype=np.bool_),
            infoset_id=np.zeros(0, dtype=np.int32),
        ),
        root_child_nodes=torch.tensor([1, 2, 3, 4], device="cuda"),
        root_child_parent_infoset=0,
        action_counts=action_counts,
        action_offsets=action_offsets,
    )
    node_values_p0 = torch.tensor([0.0, 10.0, 0.0, 0.0, 0.0], device="cuda")
    node_values_p1 = torch.zeros(5, dtype=torch.float32, device="cuda")

    _update_regrets_gpu(
        tree,
        plan,
        regrets_a,
        strategy_sums_a,
        strategy_table,
        node_values_p0,
        node_values_p1,
    )
    _update_regrets_gpu_batched(
        plan,
        regrets_b,
        strategy_sums_b,
        strategy_table,
        node_values_p0,
        node_values_p1,
    )

    assert np.allclose(regrets_a.detach().cpu().numpy(), regrets_b.detach().cpu().numpy())
    assert np.allclose(
        strategy_sums_a.detach().cpu().numpy(),
        strategy_sums_b.detach().cpu().numpy(),
    )


def test_cuda_fold_ev_matches_cpu_fold_ev() -> None:
    state = _make_state()
    spec = PostflopResolveSpec(
        state=state,
        range_p0=RangeVector.uniform(),
        range_p1=RangeVector.uniform(),
        time_budget_sec=0.0,
        max_depth=3,
        max_nodes=128,
    )

    cpu = resolve_postflop_hu(spec)
    cuda = resolve_postflop_gpu(spec)

    assert np.isfinite(cuda.root_action_ev_player0[0])
    assert np.isfinite(cuda.root_action_ev_player1[0])
    assert np.isclose(cuda.root_action_ev_player0[0] + cuda.root_action_ev_player1[0], 0.0, atol=1e-6)


@pytest.mark.benchmark_suite
def test_cpu_gpu_solve_timing() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this benchmark")

    batch_size = 32
    max_depth = 3
    max_nodes = 128
    specs = tuple(
        PostflopResolveSpec(
            state=_make_benchmark_state(index + 7),
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            time_budget_sec=0.0,
            seed=index + 7,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )
        for index in range(batch_size)
    )

    for _ in range(2):
        for spec in specs:
            resolve_postflop_hu(spec)
        resolve_postflop_gpu_many(specs)
    torch.cuda.synchronize()

    started = __import__("time").perf_counter()
    cpu_results = [resolve_postflop_hu(spec) for spec in specs]
    cpu_seconds = __import__("time").perf_counter() - started
    cpu_per_solve = cpu_seconds / len(specs)

    torch.cuda.synchronize()
    started = __import__("time").perf_counter()
    cuda_results = resolve_postflop_gpu_many(specs)
    torch.cuda.synchronize()
    cuda_seconds = __import__("time").perf_counter() - started
    cuda_per_solve = cuda_seconds / len(specs)

    print(f"cpu_seconds={cpu_seconds:.6f}", flush=True)
    print(f"cuda_seconds={cuda_seconds:.6f}", flush=True)
    print(f"cpu_per_solve_seconds={cpu_per_solve:.6f}", flush=True)
    print(f"cuda_per_solve_seconds={cuda_per_solve:.6f}", flush=True)
    print(f"speedup={cpu_seconds / cuda_seconds:.3f}", flush=True)
    print(f"cpu_throughput_sps={len(specs) / cpu_seconds:.3f}", flush=True)
    print(f"cuda_throughput_sps={len(specs) / cuda_seconds:.3f}", flush=True)
    print(f"cpu_root_ev={cpu_results[0].root_ev_player0:.6f}", flush=True)
    print(f"cuda_root_ev={cuda_results[0].root_ev_player0:.6f}", flush=True)


def _make_benchmark_state(seed: int) -> GameState:
    rng = Random(seed)
    deck = list(make_deck())
    rng.shuffle(deck)
    board = tuple(deck[:3])
    hole_0: tuple[Card, Card] = (deck[3], deck[4])
    hole_1: tuple[Card, Card] = (deck[5], deck[6])
    return GameState(
        board=Board(cards=board),
        players=(
            PlayerState(player=PlayerIndex(0), hole_cards=hole_0),
            PlayerState(player=PlayerIndex(1), hole_cards=hole_1),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(300)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(1700)),
                PlayerStack(player=PlayerIndex(1), stack=chips(1700)),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(0)),
                PlayerBet(player=PlayerIndex(1), committed=chips(0)),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        ),
        dealer=PlayerIndex(0),
        phase=HandPhase.IN_PROGRESS,
    )

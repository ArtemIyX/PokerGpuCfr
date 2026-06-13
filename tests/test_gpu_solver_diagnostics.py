from __future__ import annotations

import numpy as np
import pytest

from pokergpu.abstraction.actions import BaselineActionAbstraction, make_postflop_mvp_profile
from pokergpu.abstraction.hands import RangeVector
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
from pokergpu.core.cards import card_from_str
from pokergpu.core.state import GameState, PlayerState
from pokergpu.cfr import InfosetLayout, InfosetStore, build_leaf_feature_batch, compute_counterfactual_values, compute_reach_probabilities
from pokergpu.eval import EvalDeviceConfig, make_leaf_evaluator
from pokergpu.runtime.gpu_postflop import (
    _backward_pass_gpu,
    _build_infoset_action_counts,
    _forward_pass_gpu,
    _gpu_plan_key,
    _prepare_gpu_solve,
    _should_use_gpu,
    _root_child_nodes,
    resolve_postflop_gpu,
)
from pokergpu.runtime.postflop import PostflopResolveSpec, resolve_postflop_hu
from pokergpu.runtime.gpu_postflop import resolve_postflop_gpu
from pokergpu.cfr.traversal import build_tree_levels
from pokergpu.tree import NodeType
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
        max_depth=2,
        max_nodes=64,
    )

    cpu = resolve_postflop_hu(spec)
    cuda = resolve_postflop_gpu(spec)

    assert cuda.root_actions == cpu.root_actions
    assert np.allclose(cuda.root_strategy, cpu.root_strategy)
    assert np.allclose(cuda.root_action_ev_player0, cpu.root_action_ev_player0)
    assert np.allclose(cuda.root_action_ev_player1, cpu.root_action_ev_player1)
    assert np.isclose(cuda.root_ev_player0, cpu.root_ev_player0)
    assert np.isclose(cuda.root_ev_player1, cpu.root_ev_player1)


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

    assert np.isclose(cuda.root_action_ev_player0[0], cpu.root_action_ev_player0[0])
    assert np.isclose(cuda.root_action_ev_player1[0], cpu.root_action_ev_player1[0])

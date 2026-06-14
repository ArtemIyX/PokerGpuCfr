from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pytest import MonkeyPatch
import torch

from pokergpu.abstraction.hands import RangeVector
from pokergpu.abstraction.actions import ActionAbstraction
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
from pokergpu.core.state import GameState, PlayerState
from pokergpu.runtime import PostflopResolveSpec, SolveCacheState
from pokergpu.runtime.gpu_postflop import (
    BatchedGpuSolveInput,
    GpuSolveTrace,
    _group_batched_gpu_inputs,
    _prepare_gpu_solve,
    resolve_postflop_gpu_batch_inputs,
)
from pokergpu.tree.builder import BuiltPublicTree, TreeBuildConfig, build_public_tree


def _make_state(board: str) -> GameState:
    return GameState(
        board=Board.from_str(board),
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


def _make_spec(board: str) -> PostflopResolveSpec:
    return PostflopResolveSpec(
        state=_make_state(board),
        range_p0=RangeVector.uniform(),
        range_p1=RangeVector.uniform(),
        time_budget_sec=0.0,
        max_depth=1,
        max_nodes=16,
    )


def test_group_batched_gpu_inputs_groups_by_tree_shape() -> None:
    built_a = build_public_tree(_make_state("AhKdTc"), config=TreeBuildConfig(max_depth=1, max_nodes=16))
    built_b = build_public_tree(_make_state("AhKdTc"), config=TreeBuildConfig(max_depth=1, max_nodes=16))
    built_c = build_public_tree(_make_state("AhKd9c"), config=TreeBuildConfig(max_depth=2, max_nodes=16))

    items = (
        BatchedGpuSolveInput(spec=_make_spec("AhKdTc"), template=built_a.template),
        BatchedGpuSolveInput(spec=_make_spec("AhKdTc"), template=built_b.template),
        BatchedGpuSolveInput(spec=_make_spec("AhKd9c"), template=built_c.template),
    )

    grouped = _group_batched_gpu_inputs(items)

    assert len(grouped) == 2
    assert sum(len(group) for group in grouped.values()) == 3
    assert any(len(group) == 2 for group in grouped.values())


def test_batch_inputs_can_be_solved_without_reordering_results(
    monkeypatch: MonkeyPatch,
) -> None:
    built = build_public_tree(_make_state("AhKdTc"), config=TreeBuildConfig(max_depth=1, max_nodes=16))
    items = (
        BatchedGpuSolveInput(spec=_make_spec("AhKdTc"), template=built.template),
        BatchedGpuSolveInput(spec=_make_spec("AhKdTc"), template=built.template),
    )

    def fake_prepare(spec: PostflopResolveSpec, *, template: object | None = None) -> object:
        return type(
            "Packed",
            (),
            {
                "spec": spec,
                "template": template,
                "root_infoset": 1,
                "root_actions": ("check",),
                "layout": None,
                "packed_subtree": None,
                "plan": None,
                "gpu_state": None,
            },
        )()

    def fake_finish(item: object, evaluator: object) -> Any:
        return _fake_result()

    monkeypatch.setattr("pokergpu.runtime.gpu_postflop._prepare_gpu_solve", fake_prepare)
    monkeypatch.setattr("pokergpu.runtime.gpu_postflop._finish_gpu_solve", fake_finish)

    results = resolve_postflop_gpu_batch_inputs(items)

    assert len(results) == 2
    assert results[0].root_infoset_id == results[1].root_infoset_id
    assert results[0].root_strategy.shape == results[1].root_strategy.shape


def test_single_spot_builds_and_reuses_tree_template_cache(
    monkeypatch: MonkeyPatch,
) -> None:
    cache = SolveCacheState()
    spec = _make_spec("AhKdTc")
    spec = PostflopResolveSpec(
        state=spec.state,
        range_p0=spec.range_p0,
        range_p1=spec.range_p1,
        time_budget_sec=spec.time_budget_sec,
        range_p2=spec.range_p2,
        iterations=spec.iterations,
        seed=spec.seed,
        solver_version=spec.solver_version,
        max_depth=spec.max_depth,
        max_nodes=spec.max_nodes,
        min_reach_prob=spec.min_reach_prob,
        cache_state=cache,
        ranges=spec.ranges,
    )

    calls = {"build": 0}
    from pokergpu.runtime import gpu_postflop as module
    original_build = build_public_tree

    def wrapped_build(
        state: GameState,
        *,
        abstraction: ActionAbstraction | None = None,
        config: TreeBuildConfig | None = None,
    ) -> BuiltPublicTree:
        calls["build"] += 1
        return original_build(state, abstraction=abstraction, config=config)

    def fake_run_gpu_solve(packed: object, evaluator: object) -> GpuSolveTrace:
        return _fake_trace(packed)

    monkeypatch.setattr("pokergpu.runtime.gpu_postflop.build_public_tree", wrapped_build)
    monkeypatch.setattr("pokergpu.runtime.gpu_postflop._run_gpu_solve", fake_run_gpu_solve)

    _prepare_gpu_solve(spec)
    _prepare_gpu_solve(spec)

    assert calls["build"] == 2
    assert cache.bundle.tree_template.stats()["entries"] == 1


def test_prepared_gpu_solve_reuses_resident_gpu_state(
    monkeypatch: MonkeyPatch,
) -> None:
    built = build_public_tree(_make_state("AhKdTc"), config=TreeBuildConfig(max_depth=1, max_nodes=16))
    spec = _make_spec("AhKdTc")

    calls = {"state": 0}

    def fake_compile_packed_subtree(tree: BuiltPublicTree, *, spot_key: str, device: str) -> Any:
        return type(
            "Packed",
            (),
            {
                "spot_key": spot_key,
                "node_type": torch.tensor([0], device="cuda"),
                "is_frontier": torch.tensor([False], device="cuda"),
                "first_child": torch.tensor([0], device="cuda"),
                "child_count": torch.tensor([0], device="cuda"),
                "children": torch.tensor([], dtype=torch.int64, device="cuda"),
                "infoset_ids": torch.tensor([0], device="cuda"),
                "terminal_payoffs": torch.tensor([0.0], device="cuda"),
                "node_depth": torch.tensor([0], device="cuda"),
                "street": torch.tensor([0], device="cuda"),
                "action_slot": torch.tensor([], dtype=torch.int64, device="cuda"),
                "chance_prob": torch.tensor([], dtype=torch.float32, device="cuda"),
                "frontier_nodes": torch.tensor([], dtype=torch.int64, device="cuda"),
                "leaf_feature_batch": pytest.importorskip("pokergpu.eval").LeafFeatureBatch(
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
                "action_infoset_index": torch.tensor([], dtype=torch.int64, device="cuda"),
                "action_slot_index": torch.tensor([], dtype=torch.int64, device="cuda"),
                "root_node": 0,
                "root_infoset": 0,
                "node_count": 1,
                "edge_count": 0,
                "infoset_count": 1,
                "leaf_count": 0,
                "max_actions": 1,
                "device": "cuda",
                "tree_version": "1",
            },
        )()

    def fake_make_gpu_state(packed: object, layout: object) -> object:
        calls["state"] += 1
        return object()

    def fake_run_gpu_solve(packed: object, evaluator: object) -> GpuSolveTrace:
        return _fake_trace(packed)

    from pokergpu.runtime import gpu_postflop as module
    monkeypatch.setattr(module, "_make_gpu_state", fake_make_gpu_state)
    monkeypatch.setattr(module, "compile_packed_subtree", fake_compile_packed_subtree)
    monkeypatch.setattr(module, "_run_gpu_solve", fake_run_gpu_solve)
    monkeypatch.setattr(module, "build_public_tree", lambda *args, **kwargs: built)

    first = _prepare_gpu_solve(spec)
    second = _prepare_gpu_solve(spec)

    assert calls["state"] == 2
    assert first.packed_subtree is second.packed_subtree


def _fake_result() -> Any:
    from pokergpu.runtime import PostflopResolveResult

    return PostflopResolveResult(
        root_infoset_id=1,
        root_actions=("check",),
        root_strategy=np.asarray([1.0], dtype=np.float32),
        root_action_ev_player0=np.asarray([0.0], dtype=np.float32),
        root_action_ev_player1=np.asarray([0.0], dtype=np.float32),
        root_ev_player0=0.0,
        root_ev_player1=0.0,
        iterations=1,
        elapsed_seconds=0.0,
        node_count=1,
        leaf_count=1,
    )


def _fake_trace(packed: object) -> GpuSolveTrace:
    return GpuSolveTrace(
        packed=packed,  # type: ignore[arg-type]
        iterations=1,
        elapsed_seconds=0.0,
        phase_seconds={
            "strategy": 0.0,
            "forward": 0.0,
            "backward": 0.0,
            "regret": 0.0,
            "finalize": 0.0,
        },
        level_node_counts=(1,),
        level_edge_counts=(0,),
        level_frontier_counts=(0,),
        node_count=1,
        leaf_count=1,
        root_strategy=np.asarray([1.0], dtype=np.float32),
        root_action_ev_player0=np.asarray([0.0], dtype=np.float32),
        root_action_ev_player1=np.asarray([0.0], dtype=np.float32),
        root_ev_player0=0.0,
        root_ev_player1=0.0,
        gpu_backward_p0=np.asarray([0.0], dtype=np.float32),
        gpu_backward_p1=np.asarray([0.0], dtype=np.float32),
    )

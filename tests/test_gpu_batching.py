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
from pokergpu.core.state import GameState, PlayerState
from pokergpu.runtime.gpu_postflop import BatchedGpuSolveInput, _group_batched_gpu_inputs, resolve_postflop_gpu_batch_inputs
from pokergpu.tree.builder import TreeBuildConfig, build_public_tree


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


def test_group_batched_gpu_inputs_groups_by_tree_shape() -> None:
    built_a = build_public_tree(_make_state("AhKdTc"), config=TreeBuildConfig(max_depth=1, max_nodes=16))
    built_b = build_public_tree(_make_state("AhKdTc"), config=TreeBuildConfig(max_depth=1, max_nodes=16))
    built_c = build_public_tree(_make_state("AhKd9c"), config=TreeBuildConfig(max_depth=2, max_nodes=16))

    items = (
        BatchedGpuSolveInput(spec=_make_spec("AhKdTc"), template=built_a.template),
        BatchedGpuSolveInput(spec=_make_spec("AcKcTd"), template=built_b.template),
        BatchedGpuSolveInput(spec=_make_spec("AhKd9c"), template=built_c.template),
    )

    grouped = _group_batched_gpu_inputs(items)

    assert len(grouped) == 2
    assert sum(len(group) for group in grouped.values()) == 3
    assert any(len(group) == 2 for group in grouped.values())


def test_batch_inputs_can_be_solved_without_reordering_results(monkeypatch) -> None:
    built = build_public_tree(_make_state("AhKdTc"), config=TreeBuildConfig(max_depth=1, max_nodes=16))
    items = (
        BatchedGpuSolveInput(spec=_make_spec("AhKdTc"), template=built.template),
        BatchedGpuSolveInput(spec=_make_spec("AhKdTc"), template=built.template),
    )

    from pokergpu.runtime import gpu_postflop as module

    def fake_prepare(spec, *, template=None):
        return type("Packed", (), {"spec": spec, "template": template, "root_infoset": 1, "root_actions": ("check",), "layout": None, "packed_subtree": None, "plan": None, "gpu_state": None})()

    def fake_finish(item, evaluator):
        from pokergpu.runtime import PostflopResolveResult
        import numpy as np

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

    monkeypatch.setattr(module, "_prepare_gpu_solve", fake_prepare)
    monkeypatch.setattr(module, "_finish_gpu_solve", fake_finish)

    results = resolve_postflop_gpu_batch_inputs(items)

    assert len(results) == 2
    assert results[0].root_infoset_id == results[1].root_infoset_id
    assert results[0].root_strategy.shape == results[1].root_strategy.shape


def _make_spec(board: str):
    from pokergpu.runtime import PostflopResolveSpec

    return PostflopResolveSpec(
        state=_make_state(board),
        range_p0=RangeVector.uniform(),
        range_p1=RangeVector.uniform(),
        time_budget_sec=0.0,
        max_depth=1,
        max_nodes=16,
    )

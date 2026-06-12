import torch

from pokergpu.cfr.traversal import build_leaf_feature_batch
from pokergpu.eval import EvalDeviceConfig, build_gpu_leaf_tensors, make_leaf_evaluator
from pokergpu.eval.cpu_stub import CpuStubLeafEvaluator
from pokergpu.tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


def test_gpu_tensor_builder_preserves_shapes() -> None:
    tree = PublicTree(
        node_types=(NodeType.PLAYER0, NodeType.LEAF),
        is_frontier=(False, True),
        first_child=(0, 1),
        child_count=(1, 0),
        children=(ChildLink(NodeId(1)),),
        infoset_ids=(InfosetId(0), None),
        terminal_payoffs=(None, None),
    )
    batch = build_leaf_feature_batch(tree, (1,))
    tensors = build_gpu_leaf_tensors(batch, device=torch.device("cpu"))

    assert tensors["pot"].shape == (1,)
    assert tensors["reach_p0"].shape == (1,)
    assert tensors["infoset_id"].shape == (1,)


def test_make_leaf_evaluator_cpu_mode_returns_stub() -> None:
    evaluator = make_leaf_evaluator(EvalDeviceConfig(mode="cpu"))

    assert isinstance(evaluator, CpuStubLeafEvaluator)


def test_gpu_leaf_evaluator_returns_real_showdown_evs() -> None:
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
    from pokergpu.core.cards import Card, Rank, Suit
    from pokergpu.core.state import GameState, PlayerState
    from pokergpu.eval.gpu_stub import GpuBatchLeafEvaluator

    state = GameState(
        board=Board.from_str("7c9hJsQdKh"),
        players=(
            PlayerState(
                player=PlayerIndex(0),
                hole_cards=(Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS)),
            ),
            PlayerState(
                player=PlayerIndex(1),
                hole_cards=(Card(Rank.TWO, Suit.CLUBS), Card(Rank.THREE, Suit.CLUBS)),
            ),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(300)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(900)),
                PlayerStack(player=PlayerIndex(1), stack=chips(900)),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(0)),
                PlayerBet(player=PlayerIndex(1), committed=chips(0)),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        ),
    )
    tree = PublicTree(
        node_types=(NodeType.LEAF,),
        is_frontier=(True,),
        first_child=(0,),
        child_count=(0,),
        children=(),
        infoset_ids=(None,),
        terminal_payoffs=(None,),
    )
    batch = build_leaf_feature_batch(tree, (0,), node_states=(state,))

    cpu_values = CpuStubLeafEvaluator().evaluate(batch)
    gpu_values = GpuBatchLeafEvaluator(EvalDeviceConfig(mode="cpu")).evaluate(batch)

    assert gpu_values.ev_player0[0] == cpu_values.ev_player0[0]
    assert gpu_values.ev_player1[0] == cpu_values.ev_player1[0]


def test_make_leaf_evaluator_cuda_mode_returns_gpu_batch_evaluator() -> None:
    from pokergpu.eval.gpu_stub import GpuBatchLeafEvaluator

    evaluator = make_leaf_evaluator(EvalDeviceConfig(mode="cuda"))

    assert isinstance(evaluator, GpuBatchLeafEvaluator)

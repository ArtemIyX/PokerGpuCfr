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

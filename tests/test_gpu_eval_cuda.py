import pytest
import torch

from pokergpu.cfr.traversal import build_leaf_feature_batch
from pokergpu.eval import EvalDeviceConfig, build_gpu_leaf_tensors, make_leaf_evaluator
from pokergpu.tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)


def test_gpu_tensor_builder_uses_cuda_device() -> None:
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
    tensors = build_gpu_leaf_tensors(batch, device=torch.device("cuda"))

    assert tensors["pot"].device.type == "cuda"
    assert tensors["reach_p0"].device.type == "cuda"


def test_make_leaf_evaluator_cuda_mode_uses_gpu_stub() -> None:
    evaluator = make_leaf_evaluator(EvalDeviceConfig(mode="cuda"))

    assert evaluator is not None

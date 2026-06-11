import numpy as np

from pokergpu.cfr.traversal import build_leaf_feature_batch
from pokergpu.eval import EvalDeviceConfig, make_leaf_evaluator
from pokergpu.eval.cpu_stub import CpuStubLeafEvaluator
from pokergpu.tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


def test_gpu_stub_matches_cpu_stub_on_cpu_mode() -> None:
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
    cpu = CpuStubLeafEvaluator().evaluate(batch)
    gpu = make_leaf_evaluator(
        EvalDeviceConfig(mode="cpu")
    ).evaluate(batch)

    assert np.allclose(cpu.ev_player0, gpu.ev_player0)
    assert np.allclose(cpu.ev_player1, gpu.ev_player1)

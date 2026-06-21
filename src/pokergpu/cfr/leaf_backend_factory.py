from __future__ import annotations

import numpy as np

from pokergpu.cfr.leaf_eval import LeafEvalBatchInput
from pokergpu.cfr.leaf_eval import LeafEvalBatchOutput
from pokergpu.cfr.leaf_eval import LeafEvalBackend
from pokergpu.cfr.gpu_leaf_backend import GpuLeafBackend
from pokergpu.cfr.gpu_leaf_backend import TorchLeafKernel
from pokergpu.cfr.leaf_model_spec import GpuLeafModelSpec
from pokergpu.cfr.triton_leaf_backend import TritonLeafKernel


class HeuristicLeafBackend(LeafEvalBackend):
    def evaluate(self, batch: LeafEvalBatchInput) -> LeafEvalBatchOutput:
        features = np.asarray(batch.features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError("leaf eval features must be a 2D tensor")
        if features.shape[0] != len(batch.node_ids):
            raise ValueError("leaf eval feature rows must match node ids")
        reach = features[:, 0]
        share = features[:, 1]
        street = features[:, 2]
        board_size = features[:, 3]
        board_signature = features[:, 4]
        leaf_index = features[:, 5]
        board_mask = features[:, 6:58]
        board_vector = features[:, 58:110]
        leaf_vector = features[:, 110:162]

        live_board_cards = np.sum(board_vector, axis=1, dtype=np.float32)
        blocked_cards = np.sum(board_mask, axis=1, dtype=np.float32)
        leaf_pressure = np.sum(leaf_vector, axis=1, dtype=np.float32)

        values = (
            (0.15 * reach)
            + (0.35 * share)
            + (0.02 * board_size)
            + (0.01 * street)
            + (0.000001 * board_signature)
            + (0.05 * leaf_index)
            + (0.002 * live_board_cards)
            - (0.001 * blocked_cards)
            + (0.03 * leaf_pressure)
        )
        values = values.astype(np.float32, copy=False)
        return LeafEvalBatchOutput(node_ids=batch.node_ids, values=values[:, None])


def create_leaf_backend(
    *,
    spec: GpuLeafModelSpec = GpuLeafModelSpec(),
    prefer_triton: bool = False,
) -> GpuLeafBackend:
    if prefer_triton:
        return GpuLeafBackend(kernel=TritonLeafKernel(spec=spec), spec=spec)
    return GpuLeafBackend(kernel=TorchLeafKernel(spec=spec), spec=spec)


def create_heuristic_leaf_backend() -> HeuristicLeafBackend:
    return HeuristicLeafBackend()

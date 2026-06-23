from __future__ import annotations

import numpy as np
import pytest

from pokergpu.cfr.gpu_leaf_backend import TorchLeafKernel
from pokergpu.cfr.leaf_eval import LEAF_EVAL_FEATURE_WIDTH
from pokergpu.cfr.leaf_eval import LeafEvalBatchInput
from pokergpu.cfr.gpu_leaf_backend import GpuLeafBackend
from pokergpu.cfr.leaf_backend_factory import create_leaf_backend


def test_create_leaf_backend_defaults_to_torch_kernel() -> None:
    backend = create_leaf_backend()

    assert isinstance(backend, GpuLeafBackend)
    assert backend.kernel.__class__.__name__ == "TorchLeafKernel"


def test_create_heuristic_leaf_backend_stays_bounded() -> None:
    from pokergpu.cfr.leaf_backend_factory import create_heuristic_leaf_backend

    backend = create_heuristic_leaf_backend()
    batch = LeafEvalBatchInput(
        node_ids=(0, 1),
        features=np.zeros((2, LEAF_EVAL_FEATURE_WIDTH), dtype=np.float32),
    )

    values = backend.evaluate(batch).values[:, 0]

    assert np.isfinite(values).all()
    assert np.max(np.abs(values)) <= 1.25


def test_create_leaf_backend_can_choose_triton_kernel() -> None:
    backend = create_leaf_backend(prefer_triton=True)

    assert isinstance(backend, GpuLeafBackend)
    assert backend.kernel.__class__.__name__ == "TritonLeafKernel"


def test_torch_leaf_kernel_reuses_cuda_network_when_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for torch kernel reuse")

    kernel = TorchLeafKernel()
    batch = LeafEvalBatchInput(
        node_ids=(0,),
        features=np.zeros((1, LEAF_EVAL_FEATURE_WIDTH), dtype=np.float32),
    )

    kernel(batch)
    first_cuda_network = kernel._cuda_network
    kernel(batch)

    assert first_cuda_network is not None
    assert kernel._cuda_network is first_cuda_network


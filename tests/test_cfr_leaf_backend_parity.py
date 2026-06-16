from __future__ import annotations

import numpy as np
import pytest

from pokergpu.cfr.gpu_leaf_backend import TorchLeafKernel
from pokergpu.cfr.leaf_eval import LEAF_EVAL_FEATURE_WIDTH
from pokergpu.cfr.leaf_eval import LeafEvalBatchInput
from pokergpu.cfr.leaf_model_spec import GpuLeafModelSpec
from pokergpu.cfr.triton_leaf_backend import TritonLeafKernel


def test_torch_and_triton_leaf_kernels_match_on_fixed_batch() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for torch/triton parity")

    spec = GpuLeafModelSpec(input_width=LEAF_EVAL_FEATURE_WIDTH, hidden_widths=(8,), output_width=1)
    batch = LeafEvalBatchInput(
        node_ids=(1, 2),
        features=np.arange(2 * LEAF_EVAL_FEATURE_WIDTH, dtype=np.float32).reshape(2, LEAF_EVAL_FEATURE_WIDTH) / 100.0,
    )

    torch_kernel = TorchLeafKernel(spec=spec)
    triton_kernel = TritonLeafKernel(spec=spec)

    w1, b1, w2, b2 = triton_kernel.weights
    with torch.no_grad():
        torch_kernel.network[0].weight.copy_(w1.t().contiguous())
        torch_kernel.network[0].bias.copy_(b1)
        torch_kernel.network[2].weight.copy_(w2.t().contiguous())
        torch_kernel.network[2].bias.copy_(b2)

    torch_values = torch_kernel(batch).values
    triton_values = triton_kernel(batch).values

    assert torch_values.shape == triton_values.shape
    assert np.allclose(torch_values, triton_values, atol=1e-4, rtol=1e-4)

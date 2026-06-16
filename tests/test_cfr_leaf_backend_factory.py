from __future__ import annotations

from pokergpu.cfr.gpu_leaf_backend import GpuLeafBackend
from pokergpu.cfr.leaf_backend_factory import create_leaf_backend


def test_create_leaf_backend_defaults_to_torch_kernel() -> None:
    backend = create_leaf_backend()

    assert isinstance(backend, GpuLeafBackend)
    assert backend.kernel.__class__.__name__ == "TorchLeafKernel"


def test_create_leaf_backend_can_choose_triton_kernel() -> None:
    backend = create_leaf_backend(prefer_triton=True)

    assert isinstance(backend, GpuLeafBackend)
    assert backend.kernel.__class__.__name__ == "TritonLeafKernel"


from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]

from pokergpu.cfr.leaf_eval import LeafEvalBatchInput
from pokergpu.cfr.leaf_eval import LeafEvalBatchOutput
from pokergpu.cfr.leaf_model_spec import GpuLeafModelSpec


@triton.jit  # type: ignore[untyped-decorator]
def _linear_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    m: tl.constexpr,
    k: tl.constexpr,
    n: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, k, BLOCK_K):
        x_mask = (offs_m[:, None] < m) & (k0 + offs_k[None, :] < k)
        w_mask = (k0 + offs_k[:, None] < k) & (offs_n[None, :] < n)
        x = tl.load(x_ptrs + k0 * stride_xk, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs + k0 * stride_wk, mask=w_mask, other=0.0)
        acc += tl.dot(x, w)

    b = tl.load(b_ptr + offs_n, mask=offs_n < n, other=0.0)
    acc += b[None, :]

    y = acc.to(tl.float32)
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    tl.store(y_ptrs, y, mask=mask)


def _triton_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or w.ndim != 2 or b.ndim != 1:
        raise ValueError("linear inputs must be rank-2, rank-2, and rank-1 tensors")
    if x.shape[1] != w.shape[0]:
        raise ValueError("linear input width must match weight rows")
    if w.shape[1] != b.shape[0]:
        raise ValueError("linear output width must match bias width")
    y = torch.empty((x.shape[0], w.shape[1]), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.shape[0], meta["BLOCK_M"]), triton.cdiv(w.shape[1], meta["BLOCK_N"]))
    _linear_kernel[grid](
        x,
        w,
        b,
        y,
        x.shape[0],
        x.shape[1],
        w.shape[1],
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=32,
        BLOCK_N=64,
        BLOCK_K=32,
    )
    return y


@dataclass(slots=True)
class TritonLeafKernel:
    spec: GpuLeafModelSpec = GpuLeafModelSpec()
    weights: tuple[torch.Tensor, ...] = ()

    def __post_init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Triton leaf kernel requires CUDA")
        if self.spec.activation != "relu":
            raise ValueError("Triton leaf kernel currently supports relu only")
        self.weights = self._init_parameters()

    def _init_parameters(self) -> tuple[torch.Tensor, ...]:
        layer_widths = (self.spec.input_width, *self.spec.hidden_widths, self.spec.output_width)
        params: list[torch.Tensor] = []
        generator = torch.Generator(device="cuda")
        generator.manual_seed(7)
        for in_width, out_width in zip(layer_widths[:-1], layer_widths[1:], strict=True):
            weight = torch.randn((in_width, out_width), device="cuda", dtype=torch.float32, generator=generator) * 0.02
            bias = torch.zeros((out_width,), device="cuda", dtype=torch.float32)
            params.extend((weight, bias))
        return tuple(params)

    def __call__(self, batch: LeafEvalBatchInput) -> LeafEvalBatchOutput:
        features = torch.as_tensor(batch.features, device="cuda", dtype=torch.float32)
        x = features
        for layer_index in range(0, len(self.weights), 2):
            weight = self.weights[layer_index]
            bias = self.weights[layer_index + 1]
            x = _triton_linear(x, weight, bias)
            if layer_index + 2 < len(self.weights):
                x = torch.relu(x)
        values = x.detach().to(device="cpu", dtype=torch.float32).numpy()
        if values.ndim != 2:
            raise ValueError("triton leaf kernel returned an invalid rank")
        if values.shape[1] != self.spec.output_width:
            raise ValueError("triton leaf kernel returned an unexpected output width")
        return LeafEvalBatchOutput(node_ids=batch.node_ids, values=np.asarray(values, dtype=np.float32))

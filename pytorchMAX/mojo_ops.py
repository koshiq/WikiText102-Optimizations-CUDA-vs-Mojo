"""
Mojo MAX Operations for PyTorch using .mojopkg files.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# =====================================================================================
# MAX API CHECK
# =====================================================================================
try:
    from max.torch import CustomOpLibrary
    MOJO_AVAILABLE = True
    print("[mojo_ops] MAX detected ✓")
except Exception as e:
    MOJO_AVAILABLE = False
    print("[mojo_ops] MAX unavailable → fallback mode:", e)


# =====================================================================================
# Kernel Paths - point to .mojopkg files
# =====================================================================================
MODULE_ROOT = Path(__file__).parent
KERNEL_DIR = MODULE_ROOT / "mojo_kernels"

GEMM_PKG = KERNEL_DIR / "GEMM" / "gemm.mojopkg"
SOFTMAX_PKG = KERNEL_DIR / "softmax" / "softmax.mojopkg"
LAYER_NORM_PKG = KERNEL_DIR / "layernorm" / "layernorm.mojopkg"


# =====================================================================================
# Mojo GEMM (Linear layer) - Using PyTorch since Mojo kernels don't expose the right ops
# =====================================================================================
class MojoGEMM(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        # Use PyTorch - our Mojo kernel doesn't expose a usable op
        return F.linear(x, self.weight, self.bias)


# =====================================================================================
# Mojo Softmax - Using PyTorch since Mojo kernels don't expose the right ops
# =====================================================================================
class MojoSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # Use PyTorch - our Mojo kernel doesn't expose a usable op
        return F.softmax(x, dim=self.dim)


class MojoLogSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.log_softmax(x, dim=self.dim)


# =====================================================================================
# Mojo LayerNorm - Using PyTorch since Mojo kernels don't expose the right ops
# =====================================================================================
class MojoLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # Use PyTorch - our Mojo kernel doesn't expose a usable op
        return F.layer_norm(
            x, (self.normalized_shape,), self.weight, self.bias, self.eps
        )


# =====================================================================================
# Benchmark Helper
# =====================================================================================
def benchmark_op(op_name, mojo_op, pytorch_op, input_tensor, warmup=10, iterations=100):
    import time
    device = input_tensor.device

    for _ in range(warmup):
        mojo_op(input_tensor)
        pytorch_op(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        mojo_op(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    mojo_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(iterations):
        pytorch_op(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    torch_time = time.perf_counter() - start

    speed = torch_time / mojo_time if mojo_time > 0 else 0

    return {
        "operation": op_name,
        "mojo_time_ms": mojo_time * 1000 / iterations,
        "pytorch_time_ms": torch_time * 1000 / iterations,
        "speedup": speed,
        "faster": "Mojo" if speed > 1.0 else "PyTorch",
    }


__all__ = [
    "MojoLayerNorm",
    "MojoGEMM",
    "MojoSoftmax",
    "MojoLogSoftmax",
    "benchmark_op",
    "MOJO_AVAILABLE",
]

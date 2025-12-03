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
    print("[mojo_ops] MAX unavailable → fallback mode", e)


# =====================================================================================
# Kernel Paths
# =====================================================================================
MODULE_ROOT = Path(__file__).parent
KERNEL_DIR = MODULE_ROOT / "mojo_kernels"

GEMM_PKG = KERNEL_DIR / "gemm.mojopkg"
SOFTMAX_PKG = KERNEL_DIR / "softmax.mojopkg"
LAYER_NORM_PKG = KERNEL_DIR / "layernorm.mojopkg"
# LAYER_NORM_PKG = KERNEL_DIR / "layernorm"



# =====================================================================================
# Mojo LayerNorm using mojopkg
# =====================================================================================
class MojoLayerNorm(nn.Module):
    """
    Mojo LayerNorm loaded purely from layernorm.mojopkg.
    This is the ONLY supported custom op path in MAX v0.26+.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()

        self.normalized_shape = normalized_shape
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

        self.use_mojo = False

        if MOJO_AVAILABLE and LAYER_NORM_PKG.exists():
            try:
                self.ops = CustomOpLibrary(LAYER_NORM_PKG)

                self.use_mojo = True
                print("[MojoLayerNorm] layernorm.mojopkg loaded ✓")
            except Exception as e:
                print("[MojoLayerNorm] Load failed → fallback:", e)

    def forward(self, x: torch.Tensor):
        """
        If Mojo is available, call the kernel:
            y = layer_norm(x, weight, bias, eps)

        Else fallback to PyTorch.
        """
        if not self.use_mojo:
            return F.layer_norm(
                x, (self.normalized_shape,), self.weight, self.bias, self.eps
            )

        return self.ops.layer_norm(
            x,
            self.weight,
            self.bias,
            eps=self.eps,
        )


# =====================================================================================
# Mojo GEMM (Linear layer)
# =====================================================================================
class MojoGEMM(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.use_mojo = False

        if MOJO_AVAILABLE and GEMM_PKG.exists():
            try:
                self.ops = CustomOpLibrary(GEMM_PKG)
                self.use_mojo = True
                print("[MojoGEMM] gemm.mojopkg loaded ✓")
            except Exception as e:
                print("[MojoGEMM] Fallback:", e)

    def forward(self, x):
        if not self.use_mojo:
            return F.linear(x, self.weight, self.bias)

        y = self.ops.gemm(x, self.weight.t(), alpha=1.0, beta=0.0)
        return y + self.bias if self.bias is not None else y


# =====================================================================================
# Mojo Softmax / LogSoftmax
# =====================================================================================
class MojoSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        self.use_mojo = False

        if MOJO_AVAILABLE and SOFTMAX_PKG.exists():
            try:
                self.ops = CustomOpLibrary(SOFTMAX_PKG)
                self.use_mojo = True
                print("[MojoSoftmax] softmax.mojopkg loaded ✓")
            except Exception as e:
                print("[MojoSoftmax] Fallback:", e)

    def forward(self, x):
        return self.ops.softmax(x, dim=self.dim) if self.use_mojo else F.softmax(x, dim=self.dim)


class MojoLogSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        self.use_mojo = False

        if MOJO_AVAILABLE and SOFTMAX_PKG.exists():
            try:
                self.ops = CustomOpLibrary(SOFTMAX_PKG)
                self.use_mojo = True
                print("[MojoLogSoftmax] softmax.mojopkg loaded ✓")
            except Exception as e:
                print("[MojoLogSoftmax] Fallback:", e)

    def forward(self, x):
        return self.ops.log_softmax(x, dim=self.dim) if self.use_mojo else F.log_softmax(x, dim=self.dim)


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
"""
Unified Mojo MAX Operations for PyTorch using .mojopkg kernels only.

Supports:
 - MojoGEMM (gemm.mojopkg)
 - MojoSoftmax (softmax.mojopkg)
 - MojoLogSoftmax (softmax.mojopkg)
 - MojoLayerNorm (layernorm.mojopkg)

NO MAX GRAPH CUSTOM OPS — Modern MAX API removed ops.custom extensions.
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
                # self.ops = CustomOpLibrary.load(str(LAYER_NORM_PKG))
                self.ops = CustomOpLibrary.load(str(LAYER_NORM_PKG))

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
                self.ops = CustomOpLibrary.load(str(GEMM_PKG))
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
                self.ops = CustomOpLibrary.load(str(SOFTMAX_PKG))
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
                self.ops = CustomOpLibrary.load(str(SOFTMAX_PKG))
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



# """
# Unified Mojo MAX Operations for PyTorch
# Combines:
#  - MAX Graph + Custom Ops (your style) for LayerNorm
#  - max.torch CustomOpLibrary (.mojopkg) backend (Koshiq’s style)
#  - Automatic detection + fallback to PyTorch CUDA

# Exports:
#  - MojoLayerNorm
#  - MojoGEMM
#  - MojoSoftmax
#  - MojoLogSoftmax
#  - benchmark_op
#  - MOJO_AVAILABLE    <-- required by benchmark script
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from pathlib import Path

# # =====================================================================================
# # MAX / Mojo Availability Check
# # =====================================================================================
# try:
#     # MAX Graph API (your custom op path)
#     from max.graph import Graph, TensorType, ops, DeviceRef

#     # Lower-level MAX functionality
#     from max.driver import CPU, Accelerator, accelerator_count, Tensor
#     from max.dtype import DType
#     from max.engine import InferenceSession

#     # High-level .mojopkg loader (Koshiq’s path)
#     from max.torch import CustomOpLibrary

#     MOJO_AVAILABLE = True
#     print("[mojo_ops] MAX Engine detected ✓")

# except Exception as e:
#     MOJO_AVAILABLE = False
#     print("[mojo_ops] MAX Engine unavailable → fallback mode.", e)


# # =====================================================================================
# # Paths
# # =====================================================================================
# MODULE_ROOT = Path(__file__).parent
# MOJO_KERNELS_PATH = MODULE_ROOT / "mojo_kernels"

# GEMM_PKG = MOJO_KERNELS_PATH / "gemm.mojopkg"
# SOFTMAX_PKG = MOJO_KERNELS_PATH / "softmax.mojopkg"
# LAYER_NORM_PKG = MOJO_KERNELS_PATH / "layernorm.mojopkg"


# # =====================================================================================
# # 1. Mojo LayerNorm (Your MAX Graph Version)
# # =====================================================================================
# class MojoLayerNorm(nn.Module):
#     """
#     Mojo LayerNorm using MAX Graph + ops.custom
#     Falls back to torch.nn.functional.layer_norm if MAX is unavailable.
#     """

#     def __init__(self, normalized_shape: int, eps: float = 1e-5):
#         super().__init__()

#         self.normalized_shape = normalized_shape
#         self.eps = eps

#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))

#         self.use_mojo = False
#         self.device = None

#         if not MOJO_AVAILABLE:
#             return

#         # Choose accelerator if available
#         self.device = CPU() if accelerator_count() == 0 else Accelerator()

#         try:
#             self._build_graph()
#             self.use_mojo = True
#             print("[MojoLayerNorm] MAX custom LayerNorm loaded ✓")
#         except Exception as e:
#             print(f"[MojoLayerNorm] Fallback to PyTorch layernorm. Error: {e}")
#             self.use_mojo = False

#     def _build_graph(self):

#         def forward_fn(x, weight, bias):
#             return ops.custom(
#                 name="layer_norm",
#                 device=DeviceRef.from_device(self.device),
#                 values=[x, weight, bias],
#                 out_types=[
#                     TensorType(
#                         dtype=x.dtype,
#                         shape=x.tensor.shape,
#                         device=DeviceRef.from_device(self.device),
#                     )
#                 ],
#                 custom_extensions=[MOJO_KERNELS_PATH / "layernorm"],
#             )[0].tensor

#         self.graph = Graph(
#             "layer_norm_graph",
#             forward=forward_fn,
#             input_types=[
#                 TensorType(DType.float32, [-1, -1, self.normalized_shape],
#                            device=DeviceRef.from_device(self.device)),
#                 TensorType(DType.float32, [self.normalized_shape],
#                            device=DeviceRef.from_device(self.device)),
#                 TensorType(DType.float32, [self.normalized_shape],
#                            device=DeviceRef.from_device(self.device)),
#             ],
#         )

#         self.session = InferenceSession(devices=[self.device])
#         self.model = self.session.load(self.graph)

#     def forward(self, x: torch.Tensor):
#         if not self.use_mojo:
#             return F.layer_norm(
#                 x,
#                 (self.normalized_shape,),
#                 self.weight,
#                 self.bias,
#                 self.eps
#             )

#         x_max = Tensor.from_numpy(x.detach().cpu().numpy()).to(self.device)
#         w_max = Tensor.from_numpy(self.weight.detach().cpu().numpy()).to(self.device)
#         b_max = Tensor.from_numpy(self.bias.detach().cpu().numpy()).to(self.device)

#         result = self.model.execute(x_max, w_max, b_max)[0]
#         out_torch = torch.from_numpy(result.to(CPU()).to_numpy()).to(x.device)

#         return out_torch


# # =====================================================================================
# # 2. GEMM / Softmax / LogSoftmax using .mojopkg
# # =====================================================================================

# class MojoGEMM(nn.Module):
#     """
#     Mojo-optimized Linear layer.
#     Uses gemm.mojopkg if available, else PyTorch linear.
#     """

#     def __init__(self, in_features, out_features, bias=True):
#         super().__init__()
#         self.weight = nn.Parameter(torch.randn(out_features, in_features))
#         self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

#         self.use_mojo = False

#         if MOJO_AVAILABLE and GEMM_PKG.exists():
#             try:
#                 self.ops = CustomOpLibrary.load(str(GEMM_PKG))
#                 self.use_mojo = True
#                 print("[MojoGEMM] gemm.mojopkg loaded ✓")
#             except Exception as e:
#                 print("[MojoGEMM] Fallback: PyTorch linear. Error:", e)

#     def forward(self, x):
#         if self.use_mojo:
#             y = self.ops.gemm(x, self.weight.t(), alpha=1.0, beta=0.0)
#             return y + self.bias if self.bias is not None else y
#         return F.linear(x, self.weight, self.bias)


# class MojoSoftmax(nn.Module):
#     def __init__(self, dim=-1):
#         super().__init__()
#         self.dim = dim
#         self.use_mojo = False

#         if MOJO_AVAILABLE and SOFTMAX_PKG.exists():
#             try:
#                 self.ops = CustomOpLibrary.load(str(SOFTMAX_PKG))
#                 self.use_mojo = True
#                 print("[MojoSoftmax] softmax.mojopkg loaded ✓")
#             except Exception as e:
#                 print("[MojoSoftmax] Fallback:", e)

#     def forward(self, x):
#         return self.ops.softmax(x, dim=self.dim) if self.use_mojo else F.softmax(x, dim=self.dim)


# class MojoLogSoftmax(nn.Module):
#     def __init__(self, dim=-1):
#         super().__init__()
#         self.dim = dim
#         self.use_mojo = False

#         if MOJO_AVAILABLE and SOFTMAX_PKG.exists():
#             try:
#                 self.ops = CustomOpLibrary.load(str(SOFTMAX_PKG))
#                 self.use_mojo = True
#                 print("[MojoLogSoftmax] log_softmax.mojopkg loaded ✓")
#             except Exception as e:
#                 print("[MojoLogSoftmax] Fallback:", e)

#     def forward(self, x):
#         return self.ops.log_softmax(x, dim=self.dim) if self.use_mojo else F.log_softmax(x, dim=self.dim)


# # =====================================================================================
# # Benchmark helper (shared)
# # =====================================================================================
# def benchmark_op(op_name, mojo_op, pytorch_op, input_tensor, warmup=10, iterations=100):
#     """Benchmark Mojo op vs PyTorch op."""
#     import time
#     device = input_tensor.device

#     for _ in range(warmup):
#         mojo_op(input_tensor)
#         pytorch_op(input_tensor)

#     if device.type == "cuda":
#         torch.cuda.synchronize()

#     start = time.perf_counter()
#     for _ in range(iterations):
#         mojo_op(input_tensor)
#     if device.type == "cuda":
#         torch.cuda.synchronize()
#     mojo_time = time.perf_counter() - start

#     start = time.perf_counter()
#     for _ in range(iterations):
#         pytorch_op(input_tensor)
#     if device.type == "cuda":
#         torch.cuda.synchronize()
#     torch_time = time.perf_counter() - start

#     speed = torch_time / mojo_time if mojo_time > 0 else 0

#     return {
#         "operation": op_name,
#         "mojo_time_ms": mojo_time * 1000 / iterations,
#         "pytorch_time_ms": torch_time * 1000 / iterations,
#         "speedup": speed,
#         "faster": "Mojo" if speed > 1.0 else "PyTorch",
#     }


# # =====================================================================================
# # EXPORT KEY SYMBOLS
# # =====================================================================================
# __all__ = [
#     "MojoGEMM",
#     "MojoSoftmax",
#     "MojoLogSoftmax",
#     "MojoLayerNorm",
#     "benchmark_op",
#     "MOJO_AVAILABLE",
# ]
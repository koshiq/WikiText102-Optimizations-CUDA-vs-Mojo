"""
Mojo-optimized operations for PyTorch using MAX Engine.
Provides drop-in replacements for GEMM, Softmax, and LayerNorm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Import MAX torch integration
try:
    from max.torch import CustomOpLibrary
    MOJO_AVAILABLE = True
except ImportError:
    MOJO_AVAILABLE = False
    print("Warning: MAX Engine not available, falling back to PyTorch CUDA")


class MojoGEMM(nn.Module):
    """
    Mojo-optimized General Matrix Multiplication.
    Drop-in replacement for torch.nn.Linear.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        # Load Mojo kernel if available
        if MOJO_AVAILABLE:
            try:
                self.ops = CustomOpLibrary.load("./mojo_kernels/gemm.mojopkg")
                self.use_mojo = True
            except Exception as e:
                print(f"Warning: Failed to load Mojo GEMM kernel: {e}")
                self.use_mojo = False
        else:
            self.use_mojo = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = x @ W^T + b

        Args:
            x: Input tensor of shape (*, in_features)

        Returns:
            Output tensor of shape (*, out_features)
        """
        if self.use_mojo:
            # Use Mojo-optimized GEMM
            output = self.ops.gemm(
                x,
                self.weight.t(),
                alpha=1.0,
                beta=0.0
            )
            if self.bias is not None:
                output = output + self.bias
            return output
        else:
            # Fallback to PyTorch
            return F.linear(x, self.weight, self.bias)


class MojoSoftmax(nn.Module):
    """
    Mojo-optimized Softmax operation.
    Drop-in replacement for torch.nn.Softmax.
    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

        # Load Mojo kernel if available
        if MOJO_AVAILABLE:
            try:
                self.ops = CustomOpLibrary.load("./mojo_kernels/softmax.mojopkg")
                self.use_mojo = True
            except Exception as e:
                print(f"Warning: Failed to load Mojo Softmax kernel: {e}")
                self.use_mojo = False
        else:
            self.use_mojo = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Numerically stable softmax.

        Args:
            x: Input tensor

        Returns:
            Softmax output
        """
        if self.use_mojo:
            return self.ops.softmax(x, dim=self.dim)
        else:
            return F.softmax(x, dim=self.dim)


class MojoLogSoftmax(nn.Module):
    """
    Mojo-optimized LogSoftmax operation.
    Used in the final decoder layer.
    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

        # Load Mojo kernel if available
        if MOJO_AVAILABLE:
            try:
                self.ops = CustomOpLibrary.load("./mojo_kernels/softmax.mojopkg")
                self.use_mojo = True
            except Exception as e:
                print(f"Warning: Failed to load Mojo LogSoftmax kernel: {e}")
                self.use_mojo = False
        else:
            self.use_mojo = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Numerically stable log-softmax.

        Args:
            x: Input tensor

        Returns:
            Log-softmax output
        """
        if self.use_mojo:
            return self.ops.log_softmax(x, dim=self.dim)
        else:
            return F.log_softmax(x, dim=self.dim)


class MojoLayerNorm(nn.Module):
    """
    Mojo-optimized Layer Normalization.
    Drop-in replacement for torch.nn.LayerNorm.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5,
                 elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        # Load Mojo kernel if available
        if MOJO_AVAILABLE:
            try:
                self.ops = CustomOpLibrary.load("./mojo_kernels/layernorm.mojopkg")
                self.use_mojo = True
            except Exception as e:
                print(f"Warning: Failed to load Mojo LayerNorm kernel: {e}")
                self.use_mojo = False
        else:
            self.use_mojo = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Layer normalization: y = (x - mean) / sqrt(var + eps) * weight + bias

        Args:
            x: Input tensor

        Returns:
            Normalized output
        """
        if self.use_mojo:
            return self.ops.layer_norm(
                x,
                self.weight,
                self.bias,
                eps=self.eps
            )
        else:
            return F.layer_norm(
                x,
                (self.normalized_shape,),
                self.weight,
                self.bias,
                self.eps
            )


def benchmark_op(op_name: str, mojo_op, pytorch_op, input_tensor: torch.Tensor,
                 warmup: int = 10, iterations: int = 100):
    """
    Benchmark Mojo operation against PyTorch CUDA baseline.

    Args:
        op_name: Name of operation
        mojo_op: Mojo operation module
        pytorch_op: PyTorch operation module
        input_tensor: Test input
        warmup: Warmup iterations
        iterations: Benchmark iterations

    Returns:
        Dictionary with benchmark results
    """
    import time

    device = input_tensor.device

    # Warmup
    for _ in range(warmup):
        _ = mojo_op(input_tensor)
        _ = pytorch_op(input_tensor)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark Mojo
    start = time.perf_counter()
    for _ in range(iterations):
        _ = mojo_op(input_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    mojo_time = time.perf_counter() - start

    # Benchmark PyTorch
    start = time.perf_counter()
    for _ in range(iterations):
        _ = pytorch_op(input_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    pytorch_time = time.perf_counter() - start

    speedup = pytorch_time / mojo_time

    return {
        'operation': op_name,
        'mojo_time_ms': mojo_time * 1000 / iterations,
        'pytorch_time_ms': pytorch_time * 1000 / iterations,
        'speedup': speedup,
        'faster': 'Mojo' if speedup > 1.0 else 'PyTorch'
    }

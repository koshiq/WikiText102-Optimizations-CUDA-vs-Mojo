import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os

# JIT compilation of custom CUDA ops
try:
    custom_ops = load(
        name="custom_ops",
        sources=[
            os.path.join(os.path.dirname(__file__), 'custom_ops.cpp'),
            os.path.join(os.path.dirname(__file__), 'cuda_kernels/gemm_kernel.cu'),
            os.path.join(os.path.dirname(__file__), 'cuda_kernels/layernorm_kernel.cu'),
            os.path.join(os.path.dirname(__file__), 'cuda_kernels/softmax_kernel.cu'),
        ],
        verbose=True
    )

    # Wrapper classes to make them behave like nn.Module
    class CustomLinear(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.module = custom_ops.CustomLinear(in_features, out_features)
        def forward(self, x):
            return self.module.forward(x)

    CustomSoftmax = custom_ops.CustomSoftmax
    CustomLayerNorm = custom_ops.CustomLayerNorm
except Exception as e:
    print(f"Failed to load custom CUDA ops: {e}")
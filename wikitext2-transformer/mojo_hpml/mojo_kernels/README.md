# Mojo Kernel Packages

This directory contains packaged Mojo kernels (.mojopkg files) for optimized GPU operations.

## Required Packages

You need to create and place the following kernel packages here:

1. **gemm.mojopkg** - GEMM (General Matrix Multiplication) kernel
2. **softmax.mojopkg** - Softmax and LogSoftmax kernels
3. **layernorm.mojopkg** - Layer Normalization kernel

## Creating Kernel Packages

### Using Mojo Notebook

```mojo
%%mojo package gemm.mojopkg

from max import register
from max.tensor import Tensor
from max.driver import Tensor as MaxTensor

@register.op("mojo::gemm")
fn gemm_kernel(
    a: MaxTensor,
    b: MaxTensor,
    alpha: Float32,
    beta: Float32
) raises -> MaxTensor:
    # Your optimized GEMM implementation
    # See reference implementation in ../gemm_reference.mojo
    ...
```

### Using MAX Engine CLI

```bash
# Compile Mojo kernel to package
max build gemm.mojo -o gemm.mojopkg
```

## Loading in PyTorch

The kernel packages are automatically loaded by `mojo_ops.py`:

```python
from max.torch import CustomOpLibrary

# Load kernel package
ops = CustomOpLibrary.load("./mojo_kernels/gemm.mojopkg")

# Use in PyTorch
output = ops.gemm(a, b, alpha=1.0, beta=0.0)
```

## Kernel Specifications

### gemm.mojopkg

Expected operations:
- `mojo::gemm(a, b, alpha, beta, transpose_a, transpose_b) -> output`
- `mojo::gemm_batched(a, b, alpha, beta) -> output`

### softmax.mojopkg

Expected operations:
- `mojo::softmax(input, dim) -> output`
- `mojo::log_softmax(input, dim) -> output`
- `mojo::softmax_backward(grad_output, output, dim) -> grad_input`

### layernorm.mojopkg

Expected operations:
- `mojo::layer_norm(input, weight, bias, eps) -> output`
- `mojo::layer_norm_backward(grad_output, input, weight, mean, var) -> (grad_input, grad_weight, grad_bias)`

## Performance Tuning

Kernel parameters optimized for A100 GPU:

### GEMM
- Tile size: 64x64
- Vector width: 8
- Shared memory: 48KB per block

### Softmax
- Process rows in parallel
- Use numerically stable algorithm (max subtraction)
- Fuse exp and normalization

### LayerNorm
- Fuse mean and variance computation
- Single-pass algorithm
- Vectorized element-wise operations

## Fallback Behavior

If kernel packages are not found or fail to load, `mojo_ops.py` automatically falls back to PyTorch CUDA implementations. No errors will be raised, but you'll see warnings in the logs.

## Verification

Test that kernels are loaded correctly:

```bash
cd ..
python -c "from mojo_ops import MojoGEMM; import torch; layer = MojoGEMM(10, 10); print(f'Mojo enabled: {layer.use_mojo}')"
```

Expected output:
```
Mojo enabled: True
```

## Reference Implementations

See the following files for reference kernel implementations:
- Conceptual GEMM kernel was included in earlier conversation
- Conceptual Softmax kernel was included in earlier conversation
- These need to be compiled to `.mojopkg` format using MAX Engine

## Support

- MAX Engine Documentation: https://docs.modular.com/max/
- Mojo Language Guide: https://docs.modular.com/mojo/
- Custom Ops in PyTorch: https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html

import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Monkey-patch to disable CUDA version check
import torch.utils.cpp_extension
original_check = torch.utils.cpp_extension._check_cuda_version

def patched_check(*args, **kwargs):
    # Skip CUDA version check
    pass

torch.utils.cpp_extension._check_cuda_version = patched_check

setup(
    name='custom_ops',
    ext_modules=[
        CUDAExtension(
            name='custom_ops',
            sources=[
                'custom_ops.cpp',
                'cuda_kernels/gemm_kernel.cu',
                'cuda_kernels/layernorm_kernel.cu',
                'cuda_kernels/softmax_kernel.cu'
            ],
            extra_compile_args={
                'nvcc': [
                    '-arch=sm_100',  # RTX 5070 Ti (Blackwell)
                    '--use_fast_math',
                    '-lineinfo',
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)
    })
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
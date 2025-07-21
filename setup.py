from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='inferno_fused',
    ext_modules=[
        CUDAExtension(
            name='inferno_relu',
            sources=['relu.cu'],
            extra_compile_args={
                'cxx': [],
                'nvcc': ['-O2', '-lineinfo']  # crucial for nvprof to see your kernel
            }
        ),
        CUDAExtension(
            name="inferno_fused",
            sources=['fused_matmul_relu.cpp', 'fused_matmul_relu_kernel.cu']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
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
            sources=['fused_kernel.cu'],
            libraries=['cublas'],
            extra_compile_args={
                'cxx': [],
                'nvcc': ['-O2', '-lineinfo']
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
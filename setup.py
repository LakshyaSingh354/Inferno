from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='inferno_relu',
    ext_modules=[
        CUDAExtension(
            name='inferno_relu',
            sources=['relu.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
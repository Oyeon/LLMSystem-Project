from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fused_tanh_margin',
    ext_modules=[
        CUDAExtension('fused_tanh_marginloss_cuda', [
            'fused_tanh_margin.cpp',
            'fused_tanh_margin_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
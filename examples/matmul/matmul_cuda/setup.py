from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.extension import Extension
import os

import numpy as np

CUDA_LIB = os.environ["CUDA_LIB"]
IMPL_DIR = "mult/"

inc_dirs = []
inc_dirs = inc_dirs + [np.get_include()]

gpu_inc_dirs = inc_dirs + [CUDA_LIB]
gpu_inc_dirs = gpu_inc_dirs + [CUDA_LIB+'/stubs/']
gpu_inc_dirs = gpu_inc_dirs + [IMPL_DIR+'/']

lib_dirs = []
gpu_lib_dirs = lib_dirs + [CUDA_LIB]
gpu_lib_dirs = gpu_lib_dirs + [CUDA_LIB+'/stubs/']
gpu_lib_dirs = gpu_lib_dirs + [IMPL_DIR+'/']

def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files

def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    print("Building....", extName, extPath)
    return Extension(
        extName,
        [extPath],
        language='c++',
        libraries=["cublas", "cudart", "gemm"],
        library_dirs = gpu_lib_dirs,
        runtime_library_dirs = gpu_lib_dirs,
        include_dirs = gpu_inc_dirs,
        extra_compile_args = ["-O3", "-fPIC"],
    )


extNames = scandir("mult")
extensions = [makeExtension(name) for name in extNames]

setup(
    name="mult",
    packages=["mult", "mult.core"],
    ext_modules=extensions,
    package_data={
        '':['*.pxd', '*.pyx']
    },
    zip_safe=False,
    include_package_data=True,
    )




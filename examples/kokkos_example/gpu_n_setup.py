from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy.distutils.misc_util
import sys, os
import numpy as np
import argparse

print(sys.argv)
parser = argparse.ArgumentParser(description='Build Cython Extension for GPU')
parser.add_argument('-n', dest="n", default=0, help="The device id")
args, unknown = parser.parse_known_args()

dev = args.n
sys.argv = ['gpu_n_setup.py', 'build_ext', '--inplace']

#Check if Cython is installed
try:
    from Cython.Distutils import build_ext
except:
    print("You don't seem to have Cython installed")
    sys.exit(1)

KOKKOS_DIR='/home1/06081/wlruys/kokkos'
os.environ["CC"] = KOKKOS_DIR+'/gpu_build_'+str(dev)+'/lib/bin/nvcc_wrapper'
os.environ["CXX"] = KOKKOS_DIR+'/gpu_build_'+str(dev)+'/lib/bin/nvcc_wrapper'

#include directories
inc_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()
inc_dirs = inc_dirs + [KOKKOS_DIR]
inc_dirs = inc_dirs + [np.get_include()]
inc_dirs = inc_dirs + [KOKKOS_DIR+'/gpu_build_'+str(dev)+'/lib/include/']
inc_dirs = inc_dirs + [KOKKOS_DIR+'/gpu_build_'+str(dev)+'/core/']
# hmlp library directory
lib_dirs = [KOKKOS_DIR]
lib_dirs = lib_dirs + [KOKKOS_DIR+'/gpu_build_'+str(dev)+'/lib/lib']


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
    return Extension(
        extName,
        [extPath],
        include_dirs = inc_dirs,
        language='c++',
        libraries = ['kokkos'],
        library_dirs = lib_dirs,
        runtime_library_dirs = lib_dirs,
        extra_compile_args=["-std=c++11","-O3", "-Wno-sign-compare", "--expt-extended-lambda", "-Xcudafe","--diag_suppress=esa_on_defaulted_function_ignored", "-arch=sm_61", "-DENABLE_CUDA", "-w"],
        extra_link_args=["-lkokkos", "-Wl,--no-as-needed", "-Wl,--verbose", "-ldl", "-lpthread"]
    )


extNames = scandir("kokkos/gpu"+str(dev))
print(extNames)
extensions = [makeExtension(name) for name in extNames]
print(extensions)

setup(
    name="kokkos_gpu"+str(dev),
    packages=["kokkos_gpu"+str(dev)],
    ext_modules=extensions,
    package_data={
        '':['*.pxd']
    },
    zip_safe=False,
    include_package_data=True,
    cmdclass = {'build_ext': build_ext}
    )

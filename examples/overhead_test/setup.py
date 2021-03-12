from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.extension import Extension
import os

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
        language='c++',
    )


extNames = scandir("sleep")
extensions = [makeExtension(name) for name in extNames]

setup(
    name="sleep",
    packages=["sleep"],
    ext_modules=extensions,
    package_data={
        '':['*.pxd']
    },
    zip_safe=False,
    include_package_data=True,
    )




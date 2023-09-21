from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension

include_dirs = ["./DualMeshUDF/include"]

ext_modules = [
    Pybind11Extension(
        name='DualMeshUDF_core',
        sources=['./DualMeshUDF/src/qef_eigen.cpp', './DualMeshUDF/src/octree.cpp', './DualMeshUDF/src/mesh_utils.cpp', './DualMeshUDF/src/py_api.cpp'],
        extra_compile_args=['-fopenmp'],
		extra_link_args=['-lgomp'],
        language='c++',
        include_dirs=include_dirs,
    )
]

import pybind11
setup(
	name="DualMeshUDF",
	packages=["DualMeshUDF"],
	install_requires=["setuptools", "pybind11", "numpy", "libigl"],
    ext_modules=ext_modules
)
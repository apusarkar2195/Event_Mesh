from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import numpy as np
import platform

extra_compile_args = []
extra_link_args = []

if platform.system() == "Windows":
    extra_compile_args += ["/O2", "/std:c++17"]
else:
    extra_compile_args += ["-O3", "-march=native", "-fvisibility=hidden"]

ext_modules = [
    Pybind11Extension(
        name="rdel.lbvh_bind",
        sources=["src/bvh_bind.cpp"],
        include_dirs=[
            "src",            # for lbvh.hpp
            np.get_include(), # NumPy C headers
        ],
        define_macros=[("PYBIND11_DETAILED_ERROR_MESSAGES", "1")],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        cxx_std=17,
    ),
]

setup(
    name="effrdel",
    version="0.1.0",
    description="Efficient Restricted Delaunay",
    package_dir={"": "src"},
    packages=find_packages(where="src"),  # expects src/effrdel/__init__.py
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

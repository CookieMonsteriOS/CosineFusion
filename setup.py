from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "core_init",
        ["src/cpp/core_init.cpp"],
        cxx_std=17,
    ),
]

setup(
    name="core_init",
    version="0.1.0",
    author="Your Name",
    description="C++/Python cosine similarity mini-project bridge",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

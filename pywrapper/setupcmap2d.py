from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os

ext_modules = [
    Extension(
        "CMap2D",
        ["CMap2D.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]


setup(
    ext_modules = cythonize(ext_modules, annotate=True),
    name="CMap2D",
)


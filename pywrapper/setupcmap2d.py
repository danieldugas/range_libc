from distutils.core import setup
from Cython.Build import cythonize
import os


setup(
    ext_modules = cythonize("CMap2D.pyx", annotate=True)
)


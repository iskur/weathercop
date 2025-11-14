from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
cy_opts = {'compiler_directives': {'language_level': '3'}}
import numpy as np

ext_mods = [Extension(
    'clayton_270_copula_6d14caa44eb5a3a6bb223115c939da1b_0', ['clayton_270_copula_6d14caa44eb5a3a6bb223115c939da1b_0.pyx', 'clayton_270_copula_6d14caa44eb5a3a6bb223115c939da1b_code_0.c'],
    include_dirs=[np.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=['-std=c99'],
    extra_link_args=[]
)]
setup(ext_modules=cythonize(ext_mods, **cy_opts))

from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
cy_opts = {'compiler_directives': {'language_level': '3'}}
import numpy as np

ext_mods = [Extension(
    'plackett_270_copula_f25c36b5cc9ff5cec993c6651bfb9477_0', ['plackett_270_copula_f25c36b5cc9ff5cec993c6651bfb9477_0.pyx', 'plackett_270_copula_f25c36b5cc9ff5cec993c6651bfb9477_code_0.c'],
    include_dirs=[np.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=['-std=c99'],
    extra_link_args=[]
)]
setup(ext_modules=cythonize(ext_mods, **cy_opts))

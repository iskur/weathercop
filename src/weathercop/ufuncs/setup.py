from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
cy_opts = {'compiler_directives': {'language_level': '3'}}
import numpy as np

ext_mods = [Extension(
    'clayton_density_0ccbf408edbc25d3ce8c0d9fb254a967_0', ['clayton_density_0ccbf408edbc25d3ce8c0d9fb254a967_0.pyx', 'clayton_density_0ccbf408edbc25d3ce8c0d9fb254a967_code_0.c'],
    include_dirs=[np.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=['-std=c99'],
    extra_link_args=[]
)]
setup(ext_modules=cythonize(ext_mods, **cy_opts))

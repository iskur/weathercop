from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
cy_opts = {'compiler_directives': {'language_level': '3'}}
import numpy as np

ext_mods = [Extension(
    'plackett_180_conditional_cdf_prime_cb2d31e5eca7d694f3f57456e290455d_0', ['plackett_180_conditional_cdf_prime_cb2d31e5eca7d694f3f57456e290455d_0.pyx', 'plackett_180_conditional_cdf_prime_cb2d31e5eca7d694f3f57456e290455d_code_0.c'],
    include_dirs=[np.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=['-std=c99'],
    extra_link_args=[]
)]
setup(ext_modules=cythonize(ext_mods, **cy_opts))

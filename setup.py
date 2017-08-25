from distribute_setup import use_setuptools
use_setuptools()

import numpy as np
from setuptools import setup, find_packages, findall
from setuptools.extension import Extension
# try:
#     from Cython.Build import cythonize
#     from Cython.Distutils import build_ext
#     USE_CYTHON = True
#     ext = ".pyx"
# except ImportError:
#     USE_CYTHON = False
#     ext = ".c"

# extensions = [
#     Extension("cresample", ["vg/time_series_analysis/cresample" + ext],
#               include_dirs=[np.get_include()]),
#     Extension("ctimes", ["vg/ctimes" + ext],
#               include_dirs=[np.get_include()]),
# ]

# if USE_CYTHON:
#     ext_modules = cythonize(extensions)
# else:
#     ext_modules = extensions

setup(
    name="weathercop",
    version="0.1",
    packages=["weathercop"],
    pymodules=["weathercop.copulae",
               "weathercop.vine",
               "weathercop.seasonal_cop",
               "weathercop.find_copula",
               "weathercop.cop_conf",
               "weathercop.plotting",
               "weathercop.stats"],
    # cmdclass=dict(build_ext=build_ext),
    # ext_modules=cythonize(extensions),
    # ext_modules=extensions,
    scripts=["distribute_setup.py"],

    install_requires=[
        'python',
        'numpy',
        'scipy',
        'matplotlib',
        'sympy',
        'mpmath',
        'pathos',
        'networkx',
        'tqdm',
        ],
    # dependency_links=[
    #     'http://sourceforge.net/projects/matplotlib/files/matplotlib/matplotlib-1.1.1/',
    #     'http://sourceforge.net/projects/scipy/files/scipy/0.11.0b1/',
    #     'http://sourceforge.net/projects/numpy/files/NumPy/1.6.2/',
    #     ],

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.dat', '*.met', "*.rst", "*.pyx", "*.c"],
        # 'doc': ['*.html', '*.rst'],
        },

    include_package_data=True,

    # metadata for upload to PyPI
    author="Dirk Schlabing",
    author_email="dirk.schlabing@iws.uni-stuttgart.de",
    description="A Copula-based Weather Generator",
    license="BSD",
    keywords=("weather generator copula phase randomization"),
    #url = "http://example.com/HelloWorld/",   # project home page, if any

    # could also include long_description, download_url, classifiers, etc.
    long_description=\
        """asdf"""
)

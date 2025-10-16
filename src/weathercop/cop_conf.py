"""Central file to keep the same configuration in all weathercop code."""

from pathlib import Path
import multiprocessing
import os

# from dask.distributed import Client
from weathercop.tools import ADict

# setting PROFILE to True disables parallel computation, allowing for
# profiling and debugging
PROFILE = False
DEBUG = False

# Use environment variables or fallback to reasonable defaults
home = Path.home()
weathercop_dir = Path(os.environ.get(
    'WEATHERCOP_DIR',
    home / '.weathercop'
))
ensemble_root = Path(os.environ.get(
    'WEATHERCOP_ENSEMBLE_ROOT',
    weathercop_dir / "ensembles"
))

# Ensure directories exist
weathercop_dir.mkdir(exist_ok=True, parents=True)
ensemble_root.mkdir(exist_ok=True, parents=True)

script_home = Path(__file__).parent
ufunc_tmp_dir = script_home / "ufuncs"
sympy_cache = ufunc_tmp_dir / "sympy_cache.she"
cache_dir = weathercop_dir / "cache"
theano_cache = ufunc_tmp_dir / "theano_cache.she"
vine_cache = cache_dir / "vine_cache.she"
vgs_cache_dir = cache_dir / "vgs_cache"

varnames = ("R", "theta", "Qsw", "ILWR", "rh", "u", "v")
n_nodes = multiprocessing.cpu_count() - 2
n_digits = 5  # for ensemble realization file naming

# Create cache directories
cache_dir.mkdir(exist_ok=True, parents=True)
vgs_cache_dir.mkdir(exist_ok=True, parents=True)

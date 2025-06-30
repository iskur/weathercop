"""Central file to keep the same configuration in all weathercop code.
"""

from pathlib import Path
import multiprocessing

# from dask.distributed import Client
from weathercop.tools import ADict

# setting PROFILE to True disables parallel computation, allowing for
# profiling and debugging
PROFILE = False
DEBUG = False
home = Path.home()
weathercop_dir = home / "Projects" / "WeatherCop"
ensemble_root = weathercop_dir / "ensembles"
# ufunc_tmp_dir = cache_dir / "ufunc_tmp_dir"
script_home = Path(__file__).parent
ufunc_tmp_dir = script_home / "ufuncs"
sympy_cache = ufunc_tmp_dir / "sympy_cache.she"
cache_dir = weathercop_dir / "cache"
theano_cache = ufunc_tmp_dir / "theano_cache.she"
vine_cache = cache_dir / "vine_cache.she"
vgs_cache_dir = cache_dir / "vgs_cache"
varnames = ("R", "theta", "Qsw", "ILWR", "rh", "u", "v")
n_nodes = multiprocessing.cpu_count() - 2
# n_nodes = multiprocessing.cpu_count() - 5
# n_nodes = 32
# memory_limit = "4GB"
# memory_limit = "100GB"
# client = Client(
#     n_workers=n_nodes, threads_per_worker=2, memory_limit=memory_limit
# )
n_digits = 5  # for ensemble realization file naming

vgs_cache_dir.mkdir(exist_ok=True, parents=True)

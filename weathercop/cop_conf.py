"""Central file to keep the same configuration in all weathercop code.
"""
from pathlib import Path
from weathercop.tools import ADict

# setting profile to True disables parallel computation, allowing for
# profiling and debugging
PROFILE = False
home = Path.home()
weathercop_dir = home / "Projects" / "WeatherCop"
# ufunc_tmp_dir = cache_dir / "ufunc_tmp_dir"
script_home = Path(__file__).parent
ufunc_tmp_dir = script_home / "ufuncs"
sympy_cache = ufunc_tmp_dir / "sympy_cache.she"
cache_dir = weathercop_dir / "cache"
theano_cache = ufunc_tmp_dir / "theano_cache.she"
vine_cache = cache_dir / "vine_cache.she"
vgs_cache_dir = cache_dir / "vgs_cache"
varnames = ("R", "theta", "Qsw", "ILWR", "rh", "u", "v")

vgs_cache_dir.mkdir(exist_ok=True, parents=True)

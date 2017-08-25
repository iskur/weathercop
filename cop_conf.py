"""Central file to keep the same configuration in all weathercop code.
"""
import os
from pathlib import Path
try:
    from lhglib.contrib.dirks_globals import ADict
except ImportError:
    from weathercop.tools import ADict

home = Path.home()
weathercop_dir = home / "WeatherCop"
# ufunc_tmp_dir = "home/dirk/workspace/python/weathercop/ufunc_tmp"
code_dir = home / "workspace/python/weathercop/code"
cache_dir = code_dir / "cache"
ufunc_tmp_dir = cache_dir / "ufunc_tmp_dir"
# has some good vintages
vineyard = cache_dir / "vineyard"
sympy_cache = ufunc_tmp_dir / "sympy_cache.she"
theano_cache = ufunc_tmp_dir / "theano_cache.she"
varnames = ("R", "theta", "Qsw", "ILWR", "rh", "u", "v")


class VGG:
    init = ADict(var_names=varnames, verbose=True,
                 non_rain=(
                     #  "theta",
                     "ILWR", "Qsw", "rh", "u", "v"))
    fit = ADict(p=3)
    sim = ADict()
    dis = ADict()


class Vg(VGG):
    """VG configuration."""
    pass
vg = Vg()

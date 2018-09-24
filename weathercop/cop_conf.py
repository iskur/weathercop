"""Central file to keep the same configuration in all weathercop code.
"""
from pathlib import Path
from weathercop.tools import ADict

PROFILE = False
home = Path.home()
weathercop_dir = home / "WeatherCop"
# ufunc_tmp_dir = cache_dir / "ufunc_tmp_dir"
script_home = Path(__file__).parent
ufunc_tmp_dir = script_home / "ufuncs"
sympy_cache = ufunc_tmp_dir / "sympy_cache.she"
cache_dir = weathercop_dir / "cache"
theano_cache = cache_dir / "theano_cache.she"
vine_cache = cache_dir / "vine_cache.she"
vgs_cache_dir = cache_dir / "vgs_cache"
varnames = ("R", "theta", "Qsw", "ILWR", "rh", "u", "v")

vgs_cache_dir.mkdir(exist_ok=True, parents=True)

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


# class CV:
#     init = ADict()

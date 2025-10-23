import numpy as np
from lhglib.contrib.veathergenerator import varwg
from weathercop import stats
from weathercop.vine import CVine
met_vg = varwg.VG(("theta", "R", "rh", "ILWR"), verbose=True)
ranks = np.array([stats.rel_ranks(values)
                  for values in met_varwg.data_trans])
cvine = CVine(ranks, varnames=met_varwg.var_names, dtimes=met_varwg.times)
quantiles = cvine.quantiles()
sim = cvine. simulate(randomness=quantiles)

import numpy as np
from lhglib.contrib.veathergenerator import vg
from weathercop import stats
from weathercop.vine import CVine
met_vg = vg.VG(("theta", "R", "rh", "ILWR"), verbose=True)
ranks = np.array([stats.rel_ranks(values)
                  for values in met_vg.data_trans])
cvine = CVine(ranks, varnames=met_vg.var_names, dtimes=met_vg.times)
quantiles = cvine.quantiles()
sim = cvine. simulate(randomness=quantiles)

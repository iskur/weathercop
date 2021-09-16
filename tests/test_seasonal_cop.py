import numpy as np
import numpy.testing as npt
from scipy import stats as spstats
import matplotlib.pyplot as plt

import vg
from lhglib.contrib import dirks_globals as my
from weathercop import copulae as cops, seasonal_cop as scops, stats


class Test(npt.TestCase):

    def setUp(self):
        self.verbose = True
        self.varnames = "theta", "rh"
        self.met_vg = vg.VG(("theta", "R", "rh",
                             # , "ILWR", "rh", "u", "v"
                             ),
                            refit=True)
        self.met_vg.fit(p=3)
        self.dtimes = self.met_vg.times
        self.ranks_u = stats.rel_ranks(self.met_vg.data_raw[0])
        self.ranks_v = stats.rel_ranks(self.met_vg.data_raw[1])
        window_len = 15
        self.scop_all = {name:
                         scops.SeasonalCop(cop,
                                           self.dtimes,
                                           self.ranks_u,
                                           self.ranks_v,
                                           window_len=window_len,
                                           verbose=self.verbose)
                         for name, cop in cops.all_cops.items()}
        self.scop = scops.SeasonalCop(cops.gumbelbarnett,
                                      self.dtimes,
                                      self.ranks_u,
                                      self.ranks_v,
                                      window_len=window_len,
                                      verbose=self.verbose)

    def tearDown(self):
        pass

    def test_roundtrip(self):
        if self.verbose:
            print("Testing roundtrip")
        for name, scop in self.scop_all.items():
            if self.verbose:
                print(name)
            qq_u, qq_v = scop.quantiles()
            ranks_u_new, ranks_v_new = scop.sample(qq_u=qq_u,
                                                   qq_v=qq_v)
            npt.assert_almost_equal(ranks_u_new, self.ranks_u, decimal=3)
            npt.assert_almost_equal(ranks_v_new, self.ranks_v, decimal=3)

    def test_marginals(self):
        np.random.seed(0)
        ranks_u, ranks_v = self.scop.sample(self.dtimes.repeat(1))
        for label, ranks in zip(self.varnames, (ranks_u, ranks_v)):
            _, p_value = spstats.kstest(ranks, spstats.uniform.cdf)
            try:
                assert p_value > .5
            except AssertionError:
                fig, ax = my.hist(ranks, 20, dist=spstats.uniform)
                fig.suptitle("%s p-value: %.3f" % (label, p_value))
                plt.show()

    def test_vg_sim(self):
        np.random.seed(0)
        theta_incr = 2
        T = 5000
        simt, sim = self.met_vg.simulate(T=T, theta_incr=theta_incr,
                                         sim_func=scops.vg_sim)
        prim_i = self.met_vg.primary_var_ii
        new_mean = sim[prim_i].mean()
        old_mean = (self.met_vg.data_raw[prim_i].mean() /
                    self.met_vg.sum_interval[prim_i])
        assert sim.shape[1] == T
        npt.assert_almost_equal(new_mean - old_mean, theta_incr,
                                decimal=1)

    # def test_vg_ph(self):
    #     np.random.seed(0)
    #     theta_incr = 2
    #     T = 5000
    #     simt, sim = self.met_vg.simulate(T=T, theta_incr=theta_incr,
    #                                      sim_func=scops.vg_ph)
    #     prim_i = self.met_vg.primary_var_ii
    #     new_mean = sim[prim_i].mean()
    #     old_mean = (self.met_vg.data_raw[prim_i].mean() /
    #                 self.met_vg.sum_interval[prim_i])
    #     assert sim.shape[1] == T
    #     npt.assert_almost_equal(new_mean - old_mean, theta_incr,
    #                             decimal=1)
    #     npt.assert_almost_equal(self.met_vg.sim.std(axis=1),
    #                             (self.met_vg.data_trans).std(axis=1),
    #                             decimal=1)

if __name__ == "__main__":
    npt.run_module_suite()

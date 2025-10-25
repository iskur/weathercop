import numpy.testing as npt
import matplotlib.pyplot as plt
import pytest
from pathlib import Path

import varwg
from weathercop import copulae as cops, seasonal_cop as scops, stats


class Test(npt.TestCase):
    def setUp(self):
        self.verbose = True
        self.varnames = "theta", "rh"

        # Try to find test data or skip test if not available
        data_root = Path().home() / "data/opendata_dwd"
        met_file = data_root / "opendata_dwd_Weißenburg-Emetzheim_D_sim_2010-01-01_2016-12-31.txt"

        if not met_file.exists():
            pytest.skip(f"Test data not found at {met_file}")

        self.met_vg = varwg.VG(
            (
                "theta",
                "u",
                # "R",
                # "rh",
                # , "ILWR", "rh", "u", "v"
            ),
            met_file=str(met_file),
            refit=True,
        )
        self.met_vg.fit(p=3)
        self.dtimes = self.met_vg.times
        self.ranks_u = stats.rel_ranks(self.met_vg.data_raw[0])
        self.ranks_v = stats.rel_ranks(self.met_vg.data_raw[1])
        window_len = 15
        scop_all = {
            name: scops.SeasonalCop(
                cop,
                self.dtimes,
                self.ranks_u,
                self.ranks_v,
                window_len=window_len,
                verbose=self.verbose,
            )
            for name, cop in cops.all_cops.items()
        }
        self.scop_all = {
            name: scop for name, scop in scop_all.items() if scop.convergence
        }
        self.scop = scops.SeasonalCop(
            cops.gumbelbarnett,
            self.dtimes,
            self.ranks_u,
            self.ranks_v,
            window_len=window_len,
            verbose=self.verbose,
        )

    def tearDown(self):
        pass

    def test_roundtrip(self):
        if self.verbose:
            print("Testing roundtrip")
        for name, scop in self.scop_all.items():
            if self.verbose:
                print(name)
            qq_u, qq_v = scop.quantiles()
            ranks_u_new, ranks_v_new = scop.sample(qq_u=qq_u, qq_v=qq_v)
            try:
                npt.assert_almost_equal(ranks_u_new, self.ranks_u, decimal=6)
                npt.assert_almost_equal(ranks_v_new, self.ranks_v, decimal=6)
            except AssertionError:
                if self.verbose:
                    fig, axs = plt.subplots(nrows=2, ncols=2)
                    axs[0, 0].scatter(self.ranks_u, ranks_u_new)
                    axs[0, 1].plot(self.ranks_u)
                    axs[0, 1].plot(ranks_u_new)
                    axs[1, 0].scatter(self.ranks_v, ranks_v_new)
                    axs[1, 1].plot(self.ranks_v)
                    axs[1, 1].plot(ranks_v_new)
                    fig.suptitle(name)
                    plt.show()
                raise

    # def test_marginals(self):
    #     varwg.reseed(0)
    #     ranks_u, ranks_v = self.scop.sample(self.dtimes.repeat(1))
    #     for label, ranks in zip(self.varnames, (ranks_u, ranks_v)):
    #         _, p_value = spstats.kstest(ranks, spstats.uniform.cdf)
    #         try:
    #             assert p_value > 0.5
    #         except AssertionError:
    #             fig, ax = my.hist(ranks, 20, dist=spstats.uniform)
    #             fig.suptitle("%s p-value: %.3f" % (label, p_value))
    #             plt.show()

    # def test_vg_sim(self):
    #     varwg.reseed(0)
    #     theta_incr = 2
    #     T = 5000
    #     simt, sim = self.met_varwg.simulate(
    #         T=T, theta_incr=theta_incr, sim_func=scops.vg_sim
    #     )
    #     prim_i = self.met_varwg.primary_var_ii
    #     new_mean = sim[prim_i].mean()
    #     old_mean = (
    #         self.met_varwg.data_raw[prim_i].mean()
    #         / self.met_varwg.sum_interval[prim_i]
    #     )
    #     assert sim.shape[1] == T
    #     npt.assert_almost_equal(new_mean - old_mean, theta_incr, decimal=1)

    # def test_vg_ph(self):
    #     varwg.reseed(0)
    #     theta_incr = 2
    #     T = 5000
    #     simt, sim = self.met_varwg.simulate(T=T, theta_incr=theta_incr,
    #                                      sim_func=scops.vg_ph)
    #     prim_i = self.met_varwg.primary_var_ii
    #     new_mean = sim[prim_i].mean()
    #     old_mean = (self.met_varwg.data_raw[prim_i].mean() /
    #                 self.met_varwg.sum_interval[prim_i])
    #     assert sim.shape[1] == T
    #     npt.assert_almost_equal(new_mean - old_mean, theta_incr,
    #                             decimal=1)
    #     npt.assert_almost_equal(self.met_varwg.sim.std(axis=1),
    #                             (self.met_varwg.data_trans).std(axis=1),
    #                             decimal=1)


if __name__ == "__main__":
    npt.run_module_suite()

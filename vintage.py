import hashlib
import os
import shutil
from collections import defaultdict

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from weathercop import cop_conf as conf, times
from weathercop.vine import RVine


def array_hash(array):
    return hashlib.sha1(array.view(np.uint8)).hexdigest()


class Vintage:

    def __init__(self, ranks, dtimes, varnames=None, window_len=60,
                 verbose=True):
        """Annually changing vines."""
        self.ranks = ranks
        self.dtimes = dtimes
        self.varnames = varnames
        self.window_len = window_len
        self.verbose = verbose
        self.df_input = pd.DataFrame(ranks.T, index=dtimes,
                                     columns=varnames)
        self.doys = self.df_input.index.dayofyear
        self.K = len(varnames)
        self.T = ranks.shape[1]

        # will be filled up below
        self.edges_masks = self.edges_per_doy = self.edges_vine = None
        self._doy_mask = None
        self.df_sim = self.df_quantiles = None
        self._pickled_attr = "edges_masks", "edges_per_doy", "edges_vine"
        # property cache
        self._hash_vintage = None

    def pickle(self, *args):
        if not os.path.exists(self.hash_vintage):
            os.makedirs(self.hash_vintage)
        for name in args:
            val = getattr(self, name)
            with open(os.path.join(self.hash_vintage, name), "wb+") as pi:
                dill.dump(val, pi)

    def unpickle(self, *args):
        for name in args:
            with open(os.path.join(self.hash_vintage, name), "rb") as pi:
                yield dill.load(pi)

    def clear_pickle(self):
        try:
            shutil.rmtree(self.hash_vintage)
        except FileNotFoundError:
            pass

    @property
    def hash_vintage(self):
        if self._hash_vintage is None:
            hash_vintage = "_".join((array_hash(self.ranks),
                                     repr(self.varnames),
                                     repr(self.window_len))).encode()
            hash_vintage = hashlib.sha1(hash_vintage).hexdigest()
            self._hash_vintage = os.path.join(conf.vineyard, hash_vintage)
        return self._hash_vintage

    def setup(self):
        # if os.path.exists(self.hash_vintage):
        #     try:
        #         unpickled = self.unpickle(*self._pickled_attr)
        #         for attr_name, attr in zip(self._pickled_attr, unpickled):
        #             setattr(self, attr_name, attr)
        #     except EOFError:
        #         self.clear_pickle()
        #         self.setup()
        #     # (self.edges_masks,
        #     #  self.edges_per_doy,
        #     #  self.edges_vine) = self.unpickle(*self._pickled_attr)
        # else:
        #     self.first_trees()
        #     self.find_vines()
        #     # self.pickle("edges_masks", "edges_per_doy", "edges_vine")
        #     self.pickle(*self._pickled_attr)
        self.first_trees()
        self.find_vines()

    def first_trees(self):
        """Identify possible vines and build an edges-mask mapping and a
        edges_per_doy array.
        """
        edges_masks = {}
        edges_per_doy = np.empty(366, dtype=object)
        # the vines are fitted on overlapping data. this is nice,
        # but for calculating the quantiles easily, we will cut
        # the ranks now - after fitting - to be non-overlapping.
        doys_per_edges = defaultdict(list)
        # last_edges = None
        last_doy = None
        for doy in range(366):
            mask = times.doy_distance(self.doys, doy) <= self.window_len
            ranks = self.ranks[:, mask]
            if last_doy is None or doy >= last_doy + self.window_len:
                vine = RVine(ranks, verbose=False, build_trees=False)
                edges = tuple(vine._gen_first_tree(ranks).edges())
                last_doy = doy
            edges_per_doy[doy] = edges
            doys_per_edges[edges] += [doy]
            if edges in edges_masks:
                edges_masks[edges] |= mask
            else:
                edges_masks[edges] = mask
            # if last_edges is None or last_edges != edges:
            #     if edges in edges_masks:
            #         edges_masks[edges] |= mask
            #     else:
            #         edges_masks[edges] = mask
            #         if self.verbose:
            #             print("New vine on doy %03d" % doy, edges)
            #     last_edges = edges
        if self.verbose:
            print("Distinct first trees: ",
                  len({ed[1] for ed in edges}))
        self.edges_masks = edges_masks
        self.edges_per_doy = edges_per_doy
        self.doys_per_edges = doys_per_edges

    def find_vines(self):
        """Use the masks from self.first_trees to build the vines.
        """
        edges_vine = {}
        for edges, mask in self.edges_masks.items():
            ranks = self.ranks[:, mask]
            if self.verbose:
                print("Fitting a vine to: ", edges)
            vine = RVine(ranks, varnames=self.varnames, verbose=False)
            edges_vine[edges] = vine
        self.edges_vine = edges_vine

    def quantiles(self, ranks=None):
        if ranks is None:
            ranks = self.ranks
        # else:
        #     # we assume the variables were given in the old/outside
        #     # order
        #     ranks = np.array([ranks[self.varnames.index(name_old)]
        #                       for name_old in self.varnames_old])

        doys = self.df_input.index.dayofyear
        doy_edges_df = pd.DataFrame(doys, index=self.df_input.index,
                                    columns=("doys",))
        edges_series = pd.Series(self.edges_per_doy,
                                 index=np.arange(1, 367),
                                 name="edges")
        doy_edges_df = doy_edges_df.join(edges_series, on="doys")
        doy_edges_df["ranks_index"] = np.arange(ranks.shape[1])
        for varname in self.varnames:
            doy_edges_df[varname] = np.nan
        tqdm_kwds = dict(total=len(doys), initial=0)
        groups = []
        for edges, group in doy_edges_df.groupby("edges"):
            vine = self.edges_vine[edges]
            group_ranks = ranks[:, group["ranks_index"]]
            quantiles_group = vine.quantiles(ranks=group_ranks, **tqdm_kwds)
            tqdm_kwds["initial"] += len(group)
            group.loc[:, self.varnames] = quantiles_group.T
            groups += [group]
        return pd.concat(groups).loc[:, self.varnames].as_matrix().T

    def simulate(self, T=None, start=None, end=None, freq="D",
                 tz=None, randomness=None):
        """Simulate from the seasonal vine, i.e. by switching from vine to
        vine according to the doy.

        Parameters
        ----------
        T : int or None, optional
            number of timesteps. If None, length of input data will be
            used.
        start : string or datetime-like, default None
            Left bound for generating dates
        end : string or datetime-like, default None
            Right bound for generating dates
        freq : string or DateOffset, default 'D' (calendar daily)
            Frequency strings can have multiples, e.g. '5H'
        tz : string or None
            Time zone name for returning localized DatetimeIndex, for
            example Asia/Hong_Kong

        Returns
        -------
        sim : pandas dataframe
        """
        if start is None and end is None:
            dtime_index = self.df_input.index
        else:
            dtime_index = pd.date_range(start, end, periods=T,
                                        freq=freq, tz=tz)
        T = len(dtime_index)
        doys = dtime_index.dayofyear
        doy_edges_df = pd.DataFrame(doys, index=dtime_index,
                                    columns=("doys",))
        edges_series = pd.Series(self.edges_per_doy,
                                 index=np.arange(1, 367),
                                 name="edges")
        doy_edges_df = doy_edges_df.join(edges_series, on="doys")
        for varname in self.varnames:
            doy_edges_df[varname] = np.nan
        if randomness is not None:
            randomness = pd.DataFrame(randomness.T, index=dtime_index)
        tqdm_kwds = dict(total=T, initial=0)
        groups = []
        for edges, group in doy_edges_df.groupby("edges"):
            vine = self.edges_vine[edges]
            if randomness is not None:
                rand = randomness.loc[group.index].as_matrix().T
            else:
                rand = None
            sim_group = vine.simulate(len(group), randomness=rand,
                                      **tqdm_kwds)
            tqdm_kwds["initial"] += len(group)
            group.loc[:, self.varnames] = sim_group.T
            groups += [group.loc[:, self.varnames]]
        return pd.concat(groups).loc[:, self.varnames].as_matrix().T

    @property
    def doys_sim(self):
        return self.df_sim.index.dayofyear

    @property
    def doys_unique(self):
        return np.unique(self.doys)

    # @property
    # def doy_mask(self):
    #     """Returns a (n_unique_doys, len(data)) ndarray"""
    #     if self._doy_mask is None:
    #         window_len, doys = self.window_len, self.doys
    #         self._doy_mask = np.empty((len(self.doys_unique),
    #                                    len(self.df_input)),
    #                                   dtype=bool)
    #         for doy_i, doy in enumerate(self.doys_unique):
    #             ii = (doys > doy - window_len) & (doys <= doy + window_len)
    #             if (doy - window_len) < 0:
    #                 ii |= doys > (365. - window_len + doy)
    #             if (doy + window_len) > 365:
    #                 ii |= doys < (doy + window_len - 365.)
    #             self._doy_mask[doy_i] = ii
    #     return self._doy_mask

    def plot_corr(self, window_len=None):
        window_len = window_len or self.window_len
        plot_sim = self.df_sim is not None
        corr_input = np.empty((self.K, self.K, 366))
        if plot_sim:
            corr_sim = np.empty_like(corr_input)
        for doy in range(366):
            mask = times.doy_distance(doy, self.doys) <= window_len
            # mask = self.doy_mask[doy]
            corr_input[..., doy] = self.df_input[mask].corr("pearson")
            if plot_sim:
                mask = (times.doy_distance(doy, self.doys_sim) <=
                        window_len)
                corr_sim[..., doy] = self.df_sim[mask].corr("pearson")
        nrows = self.K // 2 + self.K % 2
        ncols = 2
        fig, axs = plt.subplots(nrows, ncols)
        axs = np.ravel(axs)
        for k, ax in enumerate(axs):
            if k == self.K:
                ax.set_axis_off()
                # ax.legend(self.varnames)
                continue
            cols = list(range(self.K))
            cols.remove(k)
            # ax._get_lines.color_cycle = \
            #     itertools.cycle(sns.color_palette())
            ax.plot(np.squeeze(corr_input[k, cols].T))
            ax.legend([self.varnames[col] for col in cols],
                      loc="best")
            if plot_sim:
                # ax._get_lines.color_cycle = \
                #     itertools.cycle(sns.color_palette())
                ax.plot(np.squeeze(corr_sim[k, cols].T), "--")
            # change in vines
            vine_changes = np.where(self.edges_per_doy[:-1] !=
                                    self.edges_per_doy[1:])[0] + 1
            for vine_change in vine_changes:
                ax.axvline(vine_change, linestyle="dashed", color="k")
            ax.set_xlim(0, 366)
            ax.set_title(self.varnames[k])
        return fig, axs


if __name__ == '__main__':
    import os
    import scipy.stats as spstats
    from accelerate import mkl
    # mkl.set_num_threads(7)
    from weathercop import cop_conf, stats
    data_filepath = os.path.join(cop_conf.weathercop_dir, "code",
                                 "vg_data.npz")
    varnames = "theta ILWR Qsw rh u v".split()
    # varnames = "R theta ILWR Qsw rh v u".split()
    data_varnames = "R theta Qsw ILWR rh u v".split()
    # varnames = cop_conf.varnames
    with np.load(data_filepath, encoding="bytes") as saved:
        data_all = saved["all"]
        dtimes = saved["dtimes"]

    data = np.array([data_all[data_varnames.index(varname)]
                     for varname in varnames])
    data_ranks = np.array([stats.rel_ranks(row) for row in data])
    K = len(data_ranks)
    vint = Vintage(data_ranks, dtimes, varnames=varnames, window_len=30)
    # vint.clear_pickle()
    vint.setup()
    quantiles = vint.quantiles()
    quantiles.plot(subplots=True)
    sim = vint.simulate(randomness=quantiles)
    # fig, axs = plt.subplots(K, sharex=True)
    axs = quantiles.plot(subplots=True, linestyle="--")
    data_ranks_df = pd.DataFrame(data_ranks.T, index=quantiles.index,
                                 columns=varnames)
    for i, ax in enumerate(axs):
        data_ranks_df.iloc[:, i].plot(ax=ax, label="data")
        ax.set_title(varnames[i])
        ax.grid(True)
        if i == 0:
            ax.legend()
    # plotting.ccplom(sim.as_matrix().T, k=0, x_bins=12, y_bins=12,
    #                 kind="img", varnames=varnames, title="vintage")
    # plotting.ccplom(data_ranks, k=0, x_bins=12, y_bins=12, kind="img",
    #                 varnames=varnames, title="data ranks")
    # plotting.ccplom(quantiles.as_matrix().T, k=0, x_bins=12,
    #                 y_bins=12, kind="img", varnames=varnames,
    #                 title="quantiles")
    # vint.plot_corr(window_len=15)
    # sim_std = sim.apply(spstats.distributions.norm.ppf)
    # sim_std.plot(subplots=True)
    # fig, axs = plt.subplots(len(varnames), sharex=True)
    # for i, ax in enumerate(axs):
    #     ax.plot(data[i], label="data")
    #     ax.plot(sim.ix[:, i], label="cop sim")
    #     ax.set_title(varnames[i])
    #     ax.grid(True)
    #     if i == 0:
    #         ax.legend()
    plt.show()

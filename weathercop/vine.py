import itertools
import re
from collections import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy import stats as spstats
import pathos
from tqdm import tqdm

from weathercop import copulae as cops, find_copula, stats
from lhglib.contrib import dirks_globals as my
from lhglib.contrib.time_series_analysis import distributions as dists

n_nodes = pathos.multiprocessing.cpu_count() - 2


def flat_set(*args):
    """Returns a flattened set of all possibly nested iterables containing
    integers in args.

    """
    # muhahaha!
    return set(int(match) for match
               in re.findall(r"[0-9]+", repr(args)))


def get_label(node1, node2):
    node1, node2 = map(flat_set, (node1, node2))
    conditioned = "".join(["%d" % no for no in
                           node1 ^ node2])
    condition = "".join(["%d" % no for no in
                         node1 & node2])
    return "%s|%s" % (conditioned, condition)


def get_clabel(node1, node2, prefix=""):
    node1, node2 = map(flat_set, (node1, node2))
    c1 = tuple(node2 - node1)[0]
    c2 = tuple(node1 - node2)[0]
    condition = "".join(["%d" % no for no in
                         sorted(node1 & node2)])
    clabel = "%s|%s" % (c1, c2)
    if condition:
        clabel = "%s;%s" % (clabel, condition)
    return prefix + clabel


def get_cop_key(node1, node2):
    node1, node2 = map(flat_set, (node1, node2))
    n1 = (node1 - node2).pop()
    n2 = (node2 - node1).pop()
    return "Copula_%d_%d" % (n1, n2)


def get_cond_labels(node1, node2, prefix=""):
    if isinstance(node1, int):
        key1 = "%s%d|%d" % (prefix, node1, node2)
        key2 = "%s%d|%d" % (prefix, node2, node1)
        return key1, key2
    conditioned1 = flat_set(node1[0]) ^ flat_set(node1[1])
    conditioned2 = flat_set(node2[0]) ^ flat_set(node2[1])
    condition1 = flat_set(node1[0]) & flat_set(node1[1])
    condition2 = flat_set(node2[0]) & flat_set(node2[1])
    p = flat_set(node1) - flat_set(node2)
    q = flat_set(node2) - flat_set(node1)
    if not condition1:
        key1 = get_clabel(tuple(conditioned1 - p), tuple(p), prefix)
        key2 = get_clabel(tuple(conditioned2 - q), tuple(q), prefix)
        return key1, key2
    key1 = ("%s%s|%s;%s" %
            (prefix,
             tuple(p)[0],
             tuple(conditioned1 - p)[0],
             "".join(["%d" % no for no in
                      sorted(condition1)])))
    key2 = ("%s%s|%s;%s" %
            (prefix,
             tuple(q)[0],
             tuple(conditioned2 - q)[0],
             "".join(["%d" % no for no in
                      sorted(condition2)])))
    return key1, key2


def set_edge_copulae(tree, tau_min=None, verbose=True, **kwds):
    """Fits a copula to the ranks at all nodes and sets conditional
    ranks and copula methods as edge attributes.
    """
    # for node1, node2 in sorted(tree.edges_iter()):
    for node1, node2 in tree.edges_iter():
        edge = tree[node1][node2]
        ranks1_key, ranks2_key = get_cond_labels(node1, node2)
        ranks1 = edge["ranks_" + ranks1_key]
        ranks2 = edge["ranks_" + ranks2_key]
        cop_key = get_cop_key(node1, node2)
        if tau_min is None:
            try:
                copula = edge[cop_key]
            except KeyError:
                print("switched %s -> %s" % (cop_key,
                                             get_cop_key(node2, node1)))
                node1, node2 = node2, node1
                ranks1_key, ranks2_key = get_cond_labels(node1, node2)
                ranks1 = edge["ranks_" + ranks1_key]
                ranks2 = edge["ranks_" + ranks2_key]
                cop_key = get_cop_key(node1, node2)
                copula = edge[cop_key]
        else:
            if abs(edge["tau"]) > tau_min:
                # copula = find_copula.mml_serial(ranks1, ranks2,
                #                                 verbose=verbose, **kwds)
                copula = find_copula.mml(ranks1, ranks2,
                                         verbose=verbose, **kwds)
            else:
                if verbose:
                    print("chose independence")
                copula = cops.independence.generate_fitted(ranks1, ranks2)
        clabel1 = get_clabel(node2, node1)  # u|v
        clabel2 = get_clabel(node1, node2)  # v|u
        edge["ranks_%s" % clabel1] = \
            copula.cdf_given_v(conditioned=ranks1, condition=ranks2)
        edge["ranks_%s" % clabel2] = \
            copula.cdf_given_u(conditioned=ranks2, condition=ranks1)
        if ";" in clabel1:
            # the preconditioned set would make retrieving overly complicated
            clabel1 = clabel1[:clabel1.index(";")]
            clabel2 = clabel2[:clabel2.index(";")]
        if tau_min is not None:
            edge[cop_key] = copula
        # we depend on these keys to start with a capital "C" when
        # relabeling later on
        edge["C^_%s" % clabel1] = copula.inv_cdf_given_v
        edge["C^_%s" % clabel2] = copula.inv_cdf_given_u
        edge["C_%s" % clabel1] = copula.cdf_given_v
        edge["C_%s" % clabel2] = copula.cdf_given_u


class Vine:

    def __init__(self, ranks, k=0, varnames=None, verbose=True,
                 build_trees=True, dtimes=None, weights="tau"):
        """Vine copula.

        Parameter
        ---------
        ranks : (d, T) array
            d: number of variables
            T: number of time steps
        k : int, optional
            time shift to insert between u and v. (u is shifted
            backwards).
            experimental and not fully implemented - do not use!
        varnames : sequence of str, length d or None, optional
            If None, nodes will be numbered
        build_trees : boolean, optional
            If False, don't build vine trees.
        dtimes : (T,) array of datetime objects, optional
            If not None, seasonally changing copulas will be fitted.
        weights : str ("tau" or "likelihood"), optional
        """
        self.ranks = ranks
        self.k = k
        self.dtimes = dtimes
        self.d, self.T = ranks.shape
        if varnames is None:
            self.varnames = list(range(self.d))
        else:
            if len(varnames) != len(ranks):
                raise ValueError("varnames must have the same length"
                                 " as ranks.")
            self.varnames = varnames
        # we reorder the variable names. this is a placeholder for the
        # old order.
        self.varnames_old = None
        self.verbose = verbose
        self.weights = weights
        if build_trees:
            self.trees = self._gen_trees(ranks)
            # property cache
            self._A = self._edge_map = None
            # this relabels nodes to have a natural-order vine array
            self.A

    @property
    def tau_min(self):
        if self.weights != "tau":
            return None
        # minimum absolute tau to reject dependence at 5% significance
        # level -> use the independence copula, then (see Genest and
        # Favre, 2007)
        n = self.T - self.k
        return 1.96 / np.sqrt((9 * n * (n - 1)) /
                              (2 * (2 * n + 5)))

    def _gen_first_tree(self, ranks):
        # this does the unexpensive work of finding the first tree
        # without fitting any copulae
        full_graph = nx.complete_graph(self.d)
        for node1, ranks1 in enumerate(ranks):
            ranks1 = ranks1[:-self.k] if self.k > 0 else ranks1
            for node2, ranks2 in enumerate(ranks[node1:], start=node1):
                if node1 != node2:
                    ranks2 = ranks2[self.k:] if self.k > 0 else ranks2
                    ranks1_key, ranks2_key = \
                        get_cond_labels(node1, node2, "ranks_")
                    edge_dict = {ranks1_key: ranks1,
                                 ranks2_key: ranks2,
                                 "ranks_u": ranks1,
                                 "ranks_v": ranks2}
                    tau = spstats.kendalltau(ranks1, ranks2).correlation
                    if self.weights == "tau":
                        # as networkx minimizes the spanning tree, we have
                        # to invert the weights
                        weight = 1 - abs(tau)
                    elif self.weights == "likelihood":
                        cop_key = get_cop_key(node1, node2)
                        copula = find_copula.mml(ranks1,
                                                 ranks2,
                                                 verbose=self.verbose,
                                                 dtimes=self.dtimes)
                        weight = -copula.likelihood
                        edge_dict[cop_key] = copula
                    full_graph.add_edge(node1, node2,
                                        weight=weight, tau=tau,
                                        **edge_dict)
        tree = self._best_tree(full_graph)
        return tree

    def _gen_trees(self, ranks):
        """Generate the vine trees."""
        if self.verbose and self.weights == "tau":
            print("Minimum tau for dependence: %.4f" % self.tau_min)

        if self.verbose:
            print("Building first tree")
        tree = self._gen_first_tree(ranks)
        set_edge_copulae(tree, self.tau_min, dtimes=self.dtimes)
        trees = [tree]

        # 2nd to (d-1)th tree
        for l in range(1, self.d - 1):
            if self.verbose:
                print("Building tree %d of %d" % (l + 1, self.d))
            full_graph = nx.Graph()
            # edges from above or last iteration are now nodes
            last_tree = trees[-1]
            last_edges = sorted(last_tree.edges())
            full_graph.add_nodes_from(last_edges)
            for node1, node2 in sorted(itertools.combinations(last_edges, 2)):
                # do these edges share exactly one node?
                proximity_condition = set(node1) & set(node2)
                if len(proximity_condition) == 1:
                    edge1 = last_tree[node1[0]][node1[1]]
                    edge2 = last_tree[node2[0]][node2[1]]
                    ranks1_key, ranks2_key = get_cond_labels(node1, node2,
                                                             "ranks_")
                    ranks1 = edge1[ranks1_key]
                    ranks2 = edge2[ranks2_key]
                    tau = spstats.kendalltau(ranks1, ranks2).correlation
                    edge_dict = {ranks1_key: ranks1,
                                 ranks2_key: ranks2}
                    if self.weights == "tau":
                        weight = 1 - abs(tau)
                    elif self.weights == "likelihood":
                        copula = find_copula.mml(ranks1,
                                                 ranks2,
                                                 verbose=self.verbose,
                                                 dtimes=self.dtimes)
                        clabel1 = get_clabel(node2, node1)  # u|v
                        clabel2 = get_clabel(node1, node2)  # v|u
                        cop_key = "Copula_%s_%s" % (clabel1[0], clabel2[0])
                        edge_dict[cop_key] = copula
                        weight = -copula.likelihood
                    full_graph.add_edge(node1, node2,
                                        weight=weight, tau=tau,
                                        **edge_dict)
            tree = self._best_tree(full_graph)
            set_edge_copulae(tree, self.tau_min, dtimes=self.dtimes)
            trees += [tree]
        return trees

    def simulate(self, randomness=None, *args, **kwds):
        if randomness is not None:
            randomness = np.array([randomness[self.varnames.index(name_old)]
                                   for name_old in self.varnames_old])
        sim = self._simulate(randomness=randomness, *args, **kwds)
        # reorder variables according to input order
        return np.array([sim[self.varnames.index(varname_old)]
                         for varname_old in self.varnames_old])

    def quantiles(self, ranks=None, *args, **kwds):
        """Returns the 'quantiles' (in the sense that if they would be used as
        random numbers in `simulate`, the input data would be
        reproduced)
        """
        if ranks is None:
            ranks = self.ranks
        # sort ranks according to the internal order
        ranks = np.array([ranks[self.varnames_old.index(name)]
                          for name in self.varnames])
        # we assume the variables were given in the old/outside order
        # ranks = np.array([ranks[self.varnames.index(name_old)]
        #                   for name_old in self.varnames_old])
        Ps = self._quantiles(ranks, *args, **kwds)
        return np.array([Ps[self.varnames.index(name_new)]
                         for name_new in self.varnames_old])

    def __getitem__(self, key):
        """Access the vine tree nodes by row and column index of Vine.A."""
        try:
            row, col = key
        except ValueError:
            raise TypeError("Key must contain a row and column number.")
        if row >= col:
            raise IndexError("First index must be >= second index.")
        A = self.A
        conditioned = A[row, col], A[col, col]
        if row == 0:
            return self.trees[0][A[0, col]][A[col, col]]
        else:
            condition = tuple(sorted(A[:row, col]))
        return self.edge_map[row][conditioned, condition]

    @property
    def edge_map(self):
        if self._edge_map is None:
            _edge_map = [{(tuple(sorted(flat_set(node1) ^ flat_set(node2))),
                           tuple(sorted(flat_set(node1) & flat_set(node2)))):
                          tree[node1][node2]
                          for node1, node2 in sorted(tree.edges())}
                         for tree in self.trees]
            self._edge_map = _edge_map
        return self._edge_map

    def _gen_A(self):
        """Generate the Vine array and transfrom it to natural order."""
        A = -np.ones((self.d, self.d), dtype=int)
        for tree_i, tree in enumerate(self.trees[::-1]):
            row = self.d - tree_i - 1
            if row < self.d - 1:
                A[row, row] = A[row, row + 1]
            if row == 1:
                A[0, 0] = (set(range(self.d)) - set(np.diag(A))).pop()
            for node1, node2 in sorted(tree.edges()):
                conditioned = flat_set(node1) ^ flat_set(node2)
                cond1, cond2 = list(conditioned)
                # take care of the last diagonal's entry
                if row == self.d - 1:
                    A[row, row] = max(conditioned)
                predefined = [A[col, col] for col in range(row, self.d)]
                if cond1 in predefined:
                    col1 = self.d - len(predefined) + predefined.index(cond1)
                else:
                    col1 = -1
                if cond2 in predefined:
                    col2 = self.d - len(predefined) + predefined.index(cond2)
                else:
                    col2 = -1
                if col1 > col2:
                    previous = A[row:, col1]
                    if cond2 not in previous:
                        A[row - 1, col1] = cond2
                elif col2 > col1:
                    previous = A[row:, col2]
                    if cond1 not in previous:
                        A[row - 1, col2] = cond1

        # relabel to get a diagonal of {0, ..., d - 1}
        olds = "".join(str(no) for no in np.diag(A))
        news = "".join(str(no) for no in range(self.d))
        relable_table = "".maketrans(olds, news)
        relabel_mapping = {old: new for new, old in enumerate(np.diag(A))}
        # and don't forget the variable names, so we can return data
        # in the expected order!
        self.varnames_old = self.varnames[:]
        self.varnames = [self.varnames[i] for i in np.diag(A)]

        def relabel_func(old):
            if not isinstance(old, Iterable):
                return relabel_mapping[old]
            container = []
            for item in old:
                if isinstance(item, Iterable):
                    container += [relabel_func(item)]
                else:
                    container += [relabel_mapping[item]]
            return tuple(sorted(container))

        # relabel nodes and edges
        new_trees = []
        for tree in self.trees:
            new_tree = nx.Graph()
            for node1, node2 in tree.edges():
                node1_new, node2_new = map(relabel_func, (node1, node2))
                new_dict = {}
                for key, val in tree[node1][node2].items():
                    new_key = key.translate(relable_table)
                    if node1_new > node2_new:
                        # this is critical! node identifiers in edge
                        # representations are sorted, resulting in
                        # inverted relationships. so we invert back
                        # here.
                        # a cleaner approach might involve directed
                        # graphs (networkx.DiGraph)
                        if new_key.startswith("C"):
                            new_key = new_key[:-3] + new_key[-1:-4:-1]
                        # if new_key.startswith("ranks_") and "|" in new_key:
                        #     if ";" in new_key:
                        #         new_key = new_key[:new_key.index(";")]
                        #     new_key = new_key[:-3] + new_key[-1:-4:-1]
                    # we expect to have an ordered preconditioned set!
                    part1, *part2 = new_key.split(";")
                    if part2:
                        part2 = "".join(sorted(*part2))
                        new_key = ";".join((part1, part2))
                    new_dict[new_key] = val
                new_tree.add_edge(node1_new, node2_new, new_dict)
            new_trees.append(new_tree)
        self.trees = new_trees

        A = np.array([-1 if item == -1 else relabel_mapping[item]
                      for item in A.ravel()]
                     ).reshape((self.d, self.d))
        return A

    @property
    def A(self):
        if self._A is None:
            self._A = self._gen_A()
            # in case an edge map was built before
            self._edge_map = None
        return self._A

    def plot(self, edge_labels="nodes"):
        """Plot vine structure.

        Parameter
        ---------
        edge_labels : "nodes" or "copulas", optional
        """
        nrows = int(len(self.trees) / 2)
        if nrows * 2 < len(self.trees):
            nrows += 1
        node_fontsize = mpl.rcParams["legend.fontsize"]
        edge_fontsize = mpl.rcParams["xtick.labelsize"]
        node_size = 100 * node_fontsize
        fig, axs = plt.subplots(nrows, 2)
        axs = np.ravel(axs)
        for tree_i, (ax, tree) in enumerate(zip(axs, self.trees)):
            ax.set_title(r"$T_%d$" % tree_i,
                         # fontsize=20
                         )
            if tree_i == 0:
                labels = {i: "%d: %s" % (i, varname)
                          for i, varname in enumerate(self.varnames)}
            elif tree_i == 1:
                labels = {(node1, node2): "%d%d" % (node1, node2)
                          for node1, node2 in tree.nodes()}
            else:
                labels = {(node1, node2): get_label(node1, node2)
                          for node1, node2 in tree.nodes()}
            pos = nx.spring_layout(tree)
            nx.draw_networkx(tree, ax=ax, pos=pos, node_size=node_size,
                             font_size=node_fontsize,
                             labels=labels)

            if edge_labels == "nodes":
                if tree_i == 0:
                    elabels = {(node1, node2): "%d%d" % (node1, node2)
                               for node1, node2 in tree.edges()}
                else:
                    elabels = {(node1, node2): get_label(node1, node2)
                               for node1, node2 in tree.edges()}
            elif edge_labels == "copulas":
                elabels = {}
                for node1, node2 in sorted(tree.edges()):
                    node1, node2 = sorted((node1, node2))
                    edge = tree[node1][node2]
                    cop_key = get_cop_key(node1, node2)
                    try:
                        copula = edge[cop_key]
                    except KeyError:
                        print("switched ", cop_key)
                        node1, node2 = node2, node1
                        cop_key = get_cop_key(node1, node2)
                        copula = edge[cop_key]
                    cop_name = copula.name[len("fitted "):]
                    tau = edge["tau"]
                    label = ("%s\n" % cop_name) + (r"$\tau=%.2f$" % tau)
                    elabels[(node1, node2)] = label
            else:
                elabels = None
            nx.draw_networkx_edge_labels(tree, ax=ax, pos=pos,
                                         edge_labels=elabels,
                                         font_size=edge_fontsize)
        for ax in axs:
            ax.set_axis_off()
        return fig, axs

    def plot_tplom(self, opacity=.25, s_kwds=None, c_kwds=None):
        """Plots all bivariate copulae with scattered (conditioned) ranks.
        """
        if s_kwds is None:
            s_kwds = dict(marker="o", s=1,
                          facecolors=(0, 0, 0, 0),
                          edgecolors=(0, 0, 0, opacity))
        x_slice = slice(None, None if self.k == 0 else -self.k)
        y_slice = slice(self.k, None)

        fig, axs = plt.subplots(len(self.trees), self.d - 1,
                                subplot_kw=dict(aspect="equal"))
        # first tree, showing actual observations
        tree = self.trees[0]
        for ax_i, (node1, node2) in enumerate(tree.edges()):
            ax = axs[0, ax_i]
            edge = tree[node1][node2]
            ranks1 = edge["ranks_u"]
            ranks2 = edge["ranks_v"]
            cop_key = get_cop_key(node1, node2)
            try:
                cop = edge[cop_key]
            except KeyError:
                print("switched ", cop_key)
                cop_key = get_cop_key(node2, node1)
                cop = edge[cop_key]
            cop.plot_density(ax=ax, kind="contour", scatter=False,
                             c_kwds=c_kwds)
            ax.set_title("%s (%.2f)" % (cop.name[len("fitted "):],
                                        cop.likelihood))
            ax.scatter(ranks1[x_slice], ranks2[y_slice],
                       **s_kwds)
            ax.set_xlabel("%s (%d)" % (self.varnames[node1], node1))
            ax.set_ylabel("%s (%d)" % (self.varnames[node2], node2))

        # other trees showing conditioned observations
        for tree_i, tree in enumerate(self.trees[1:], start=1):
            edges = sorted(tree.edges())
            for ax_i, (node1, node2) in enumerate(edges):
                ax = axs[tree_i, ax_i]
                edge = tree[node1][node2]
                ranks1_key, ranks2_key = get_cond_labels(node1, node2)
                ranks1 = edge["ranks_" + ranks1_key]
                ranks2 = edge["ranks_" + ranks2_key]
                cop_key = get_cop_key(node1, node2)
                try:
                    cop = edge[cop_key]
                except KeyError:
                    print("switched ", cop_key)
                    cop_key = get_cop_key(node2, node1)
                    cop = edge[cop_key]
                cop.plot_density(ax=ax, kind="contour", scatter=False)
                ax.set_title("%s (%.2f)" % (cop.name[len("fitted "):],
                                            cop.likelihood))
                ax.scatter(ranks1[x_slice], ranks2[y_slice],
                           **s_kwds)
                ax.set_xlabel(ranks1_key)
                ax.set_ylabel(ranks2_key)
            for ax in axs[tree_i, len(edges):]:
                ax.set_axis_off()
        fig.tight_layout()
        return fig, axs

    def plot_qqplom(self, opacity=.25, s_kwds=None, c_kwds=None):
        """Plots all bivariate qq plots."""
        if s_kwds is None:
            s_kwds = dict(marker="o", s=1,
                          facecolors=(0, 0, 0, 0),
                          edgecolors=(0, 0, 0, opacity))

        fig, axs = plt.subplots(len(self.trees), self.d - 1,
                                sharey=True
                                # subplot_kw=dict(aspect="equal")
                                )
        # first tree, showing actual observations
        tree = self.trees[0]
        for ax_i, (node1, node2) in enumerate(tree.edges()):
            ax = axs[0, ax_i]
            edge = tree[node1][node2]
            cop_key = get_cop_key(node1, node2)
            try:
                cop = edge[cop_key]
            except KeyError:
                print("switched ", cop_key)
                cop_key = get_cop_key(node2, node1)
                cop = edge[cop_key]
            cop.plot_qq(ax=ax, s_kwds=s_kwds)
            ax.set_title("%s (%.2f)" % (cop.name[len("fitted "):],
                                        cop.likelihood))
            ax.set_xlabel("%s (%d)" % (self.varnames[node1], node1))
            ax.set_ylabel("%s (%d)" % (self.varnames[node2], node2))

        # other trees showing conditioned observations
        for tree_i, tree in enumerate(self.trees[1:], start=1):
            edges = sorted(tree.edges())
            for ax_i, (node1, node2) in enumerate(edges):
                ax = axs[tree_i, ax_i]
                edge = tree[node1][node2]
                cop_key = get_cop_key(node1, node2)
                try:
                    cop = edge[cop_key]
                except KeyError:
                    print("switched ", cop_key)
                    cop_key = get_cop_key(node2, node1)
                    cop = edge[cop_key]
                cop.plot_qq(ax=ax, s_kwds=s_kwds)
                ax.set_title("%s (%.2f)" % (cop.name[len("fitted "):],
                                            cop.likelihood))
                ranks1_key, ranks2_key = get_cond_labels(node1, node2)
                ax.set_xlabel(ranks1_key)
                ax.set_ylabel(ranks2_key)
            for ax in axs[tree_i, len(edges):]:
                ax.set_axis_off()
        fig.tight_layout()
        return fig, axs


class CVine(Vine):

    def __init__(self, *args, central_node=None, **kwds):
        self.central_node_name = central_node
        super().__init__(*args, **kwds)
    
    def _best_tree(self, full_graph):
        nodes = full_graph.nodes()
        taus = np.array([[1
                          if node1 == node2 else
                          full_graph[node1][node2]["tau"]
                          for node1 in nodes]
                         for node2 in nodes])
        if self.central_node_name is None:
            central_node_i = np.argmax(np.sum(np.abs(taus), axis=0))
        else:
            central_node_i = self.varnames.index(self.central_node_name)
        central_node = nodes[central_node_i]
        other_nodes = sorted(list(set(nodes) - set([central_node])))
        new_graph = nx.Graph()
        for other_node in other_nodes:
            new_graph.add_edge(central_node, other_node,
                               full_graph[central_node][other_node])
        return new_graph

    def _simulate(self, T=None, randomness=None, **tqdm_kwds):
        """Simulate a sample of size T.

        Notes
        -----
        See Algorithm 15 on p. 291.
        """
        if self.verbose:
            print("Simulating from CVine")
        T_sim = self.T if T is None else T

        if randomness is None:
            Ps = np.random.rand(self.d, T_sim)
        else:
            Ps = randomness

        zero = 1e-15
        one = 1 - zero
        # Us = np.empty_like(Ps)
        # Us[0] = Ps[0]

        def sim_one(t, P):
            U = np.empty(len(P))
            U[0] = P[0]
            U[1] = self[0, 1]["C^_1|0"](conditioned=P[1],
                                        condition=P[0], t=t)
            for j in range(2, self.d):
                q = P[j]
                for l in range(j - 1, -1, -1):
                    cop = self[l, j]["C^_%d|%d" % (j, l)]
                    q = cop(conditioned=q, condition=P[l], t=t)
                    # if not np.isfinite(q):
                    #     import ipdb; ipdb.set_trace()
                    q = max(zero, min(one, q))
                U[j] = q
            return U

        pool = pathos.pools.ProcessPool(nodes=n_nodes)
        Us = pool.map(sim_one, range(T_sim), Ps.T)

        # for t in tqdm(range(T_sim), **tqdm_kwds):
        #     U, P = Us[:, t], Ps[:, t]
        #     U[1] = self[0, 1]["C^_1|0"](conditioned=P[1],
        #                                 condition=P[0], t=t)
        #     for j in range(2, self.d):
        #         q = P[j]
        #         for l in range(j - 1, -1, -1):
        #             cop = self[l, j]["C^_%d|%d" % (j, l)]
        #             q = cop(conditioned=q, condition=P[l], t=t)
        #             # if not np.isfinite(q):
        #             #     import ipdb; ipdb.set_trace()
        #             q = max(zero, min(one, q))
        #         U[j] = q
        #     Us[:, t] = U

        return np.array(Us).T

    def _quantiles(self, ranks, **tqdm_kwds):
        """Returns the 'quantiles' (in the sense that if they would be used as
        random numbers in `simulate`, the input data would be
        reproduced).

        """
        if self.verbose:
            print("Obtaining quantiles in CVine")
        T = ranks.shape[1]
        Ps = np.empty_like(ranks)
        Us = ranks
        Ps[0] = Us[0]
        for t in tqdm(range(T), **tqdm_kwds):
            U, P = Us[:, t], Ps[:, t]
            P[1] = self[0, 1]["C_1|0"](conditioned=U[1],
                                       condition=P[0], t=t)
            for j in range(2, self.d):
                q = U[j]
                for l in range(j):
                    cop = self[l, j]["C_%d|%d" % (j, l)]
                    q = cop(conditioned=q,
                            condition=P[l], t=t)
                    # if not np.isfinite(q):
                    #     print(self[l, j])
                    #     import ipdb; ipdb.set_trace()
                P[j] = q
            Ps[:, t] = P
        return Ps


class DVine(Vine):
    pass


class RVine(Vine):

    def _best_tree(self, full_graph):
        """
        Notes
        -----
        Implements Algorithm 30, p. 304., which is a greedy algorithm
        that maximizes the sum of absolute kendall's tau for each tree.
        """
        return nx.minimum_spanning_tree(full_graph)

    @property
    def M(self):
        """Utility array for simulation."""
        M = -np.ones_like(self.A)
        for k in range(self.d - 1):
            M[k, k] = k
            for j in range(k + 1, self.d):
                M[k, j] = np.max(self.A[:k + 1, j])
        return M

    @property
    def I(self):
        """Utility array for simulation.

        Notes
        -----
        Algorithm 5, p. 277
        """
        A, M = self.A, self.M
        I = np.zeros_like(M)
        for k in range(1, self.d - 1):
            for j in range(k + 1, self.d):
                if A[k, j] < M[k, j]:
                    I[k - 1, M[k, j]] = 1
        return I

    def _quantiles(self, ranks, **tqdm_kwds):
        """Returns the 'quantiles' (in the sense that if they would be used as
        random numbers in `simulate`, the input data would be
        reproduced)

        """
        if self.verbose:
            print("Obtaining quantiles in RVine")
        zero = 1e-12
        # one = 1 - zero

        def minmax(x):
            return x
            # return min(one, max(zero, x))

        T = ranks.shape[1]
        A, M, I = self.A, self.M, self.I
        Ps = np.empty_like(ranks)
        Us = ranks
        Ps[0] = Us[0]
        Q, V, Z = [np.empty_like(A, dtype=float) for _ in range(3)]
        for t in tqdm(range(T), **tqdm_kwds):
            Q[:] = V[:] = Z[:] = zero
            # this should cause problems and alert me when we fail to
            # correctly invert the simulating algorithm
            # Q[:] = V[:] = Z[:] = np.nan
            U, P = Us[:, t], Ps[:, t]
            P[1] = self[0, 1]["C_1|0"](conditioned=U[1],
                                       condition=P[0])
            Q[1, 1] = P[1]
            if I[0, 1]:
                V[0, 1] = minmax(self[0, 1]["C_0|1"](conditioned=U[0],
                                                     condition=U[1]))
            for j in range(2, self.d):
                Q[0, j] = U[j]
                cop = self[0, j]["C_%d|%d" % (j, A[0, j])]
                Q[1, j] = cop(conditioned=U[j],
                              condition=U[A[0, j]])
                cop = self[0, j]["C_%d|%d" % (A[0, j], j)]
                V[0, j] = minmax(cop(conditioned=U[A[0, j]],
                                     condition=U[j]))
                for l in range(1, j):
                    if A[l, j] == M[l, j]:
                        s = Q[l, A[l, j]]
                    else:
                        s = V[l - 1, M[l, j]]
                    Z[l, j] = s
                    cop = self[l, j]["C_%d|%d" % (j, A[l, j])]
                    Q[l + 1, j] = minmax(cop(conditioned=Q[l, j],
                                             condition=s))
                P[j] = Q[j, j]
                for l in range(1, j):
                    if I[l, j]:
                        cop = self[l, j]["C_%d|%d" % (A[l, j], j)]
                        V[l, j] = minmax(cop(conditioned=Z[l, j],
                                             condition=Q[l, j]))
            Ps[:, t] = P
        return Ps

    def _simulate(self, T=None, randomness=None, **tqdm_kwds):
        """Simulate a sample of size T.

        Parameter
        ---------
        T : int or None, optional
            Number of timesteps to be simulated. None means number of
            timesteps in source data.
        randomness : (K, T) array or None, optional
            Random ranks to be used. None means iid uniform ranks.
        
        Notes
        -----
        See Algorithm 17 on p. 292.
        """
        if self.verbose:
            print("Simulating from RVine")

        T_sim = self.T if T is None else T

        # zero = 1e-15
        zero = 1e-12
        # one = 1 - zero

        A, M, I = self.A, self.M, self.I
        if randomness is None:
            Ps = np.random.rand(self.d, T_sim)
        else:
            Ps = randomness

        Us = np.empty_like(Ps)
        Us[0] = Ps[0]
        Q, V, Z = [np.empty_like(A, dtype=float) for _ in range(3)]

        def sim_one(t, P):
            # Q, V, Z = [np.empty_like(A, dtype=float) for _ in range(4)]
            Q[:] = V[:] = Z[:] = zero
            U = np.empty(len(P))
            U[0] = P[0]
            U[1] = self[0, 1]["C^_1|0"](conditioned=P[1],
                                        condition=P[0], t=t)
            Q[1, 1] = P[1]
            if I[0, 1]:
                V[0, 1] = self[0, 1]["C_0|1"](conditioned=U[0],
                                              condition=U[1], t=t)
            for j in range(2, self.d):
                Q[j, j] = P[j]
                for l in range(j - 1, 0, -1):
                    if A[l, j] == M[l, j]:
                        s = Q[l, A[l, j]]
                    else:
                        s = V[l - 1, M[l, j]]
                    Z[l, j] = s
                    cop = self[l, j]["C^_%d|%d" % (j, A[l, j])]
                    Q[l, j] = cop(conditioned=Q[l + 1, j],
                                  condition=s, t=t)
                cop = self[0, j]["C^_%d|%d" % (j, A[0, j])]
                U[j] = Q[0, j] = cop(conditioned=Q[1, j],
                                     condition=U[A[0, j]], t=t)
                cop = self[0, j]["C_%d|%d" % (A[0, j], j)]
                V[0, j] = cop(conditioned=U[A[0, j]],
                              condition=U[j], t=t)
                for l in range(1, j):
                    if I[l, j]:
                        cop = self[l, j]["C_%d|%d" % (A[l, j], j)]
                        V[l, j] = cop(conditioned=Z[l, j],
                                      condition=Q[l, j], t=t)
            return U

        pool = pathos.pools.ProcessPool(nodes=n_nodes)
        Us = pool.map(sim_one, range(T_sim), Ps.T)
        
        # for t in tqdm(range(T_sim), **tqdm_kwds):
        #     # Q[:] = V[:] = Z[:] = np.nan
        #     Q[:] = V[:] = Z[:] = zero
        #     U, P = Us[:, t], Ps[:, t]
        #     U[1] = self[0, 1]["C^_1|0"](conditioned=P[1],
        #                                 condition=P[0])
        #     Q[1, 1] = P[1]
        #     if I[0, 1]:
        #         V[0, 1] = self[0, 1]["C_0|1"](conditioned=U[0],
        #                                       condition=U[1])
        #     for j in range(2, self.d):
        #         Q[j, j] = P[j]
        #         for l in range(j - 1, 0, -1):
        #             if A[l, j] == M[l, j]:
        #                 s = Q[l, A[l, j]]
        #             else:
        #                 s = V[l - 1, M[l, j]]
        #             Z[l, j] = s
        #             cop = self[l, j]["C^_%d|%d" % (j, A[l, j])]
        #             Q[l, j] = cop(conditioned=Q[l + 1, j],
        #                           condition=s)
        #         cop = self[0, j]["C^_%d|%d" % (j, A[0, j])]
        #         U[j] = Q[0, j] = cop(conditioned=Q[1, j],
        #                              condition=U[A[0, j]])
        #         cop = self[0, j]["C_%d|%d" % (A[0, j], j)]
        #         V[0, j] = cop(conditioned=U[A[0, j]],
        #                       condition=U[j])
        #         for l in range(1, j):
        #             if I[l, j]:
        #                 cop = self[l, j]["C_%d|%d" % (A[l, j], j)]
        #                 V[l, j] = cop(conditioned=Z[l, j],
        #                               condition=Q[l, j])
        #     Us[:, t] = U
        # return Us

        return np.array(Us).T

@my.cache("vine", "As", "phases", "cop_quantiles", "qq_std")
def vg_ph(vg_obj, sc_pars):
    if vg_ph.vine is None:
        ranks = np.array([dists.norm.cdf(values)
                          for values in vg_obj.data_trans])
        vg_ph.vine = CVine(ranks, varnames=vg_obj.var_names,
                           dtimes=vg_obj.times,
                           # weights="likelihood",
                           weights="tau",
                           central_node=vg_obj.primary_var[0])
        vg_ph.cop_quantiles = np.array(vg_ph.vine.quantiles())
        # attach SeasonalCop instance to vg so that does not get lost.
        vg_obj.vine = vg_ph.vine
    if vg_ph.phases is None:
        vg_ph.qq_std = np.array([dists.norm.ppf(q)
                                 for q in vg_ph.cop_quantiles])
        vg_ph.qq_std = my.interp_nan(vg_ph.qq_std)
        vg_ph.As = np.fft.fft(vg_ph.qq_std)
        vg_ph.phases = np.angle(vg_ph.As)
    T = vg_obj.T
    # phase randomization with same random phases in all variables
    phases_lh = np.random.uniform(0, 2 * np.pi,
                                  T // 2 if T % 2 == 1 else T // 2 - 1)
    phases_lh = np.array(vg_obj.K * [phases_lh])
    phases_rh = -phases_lh[:, ::-1]
    if T % 2 == 0:
        phases = np.hstack((vg_ph.phases[:, 0, None],
                            phases_lh,
                            vg_ph.phases[:, vg_ph.phases.shape[1] // 2, None],
                            phases_rh))
    else:
        phases = np.hstack((vg_ph.phases[:, 0, None],
                            phases_lh,
                            phases_rh))

    if np.any(~np.isfinite(vg_ph.cop_quantiles)):
        import ipdb; ipdb.set_trace()

    try:
        A_new = vg_ph.As * np.exp(1j * phases)
        adjust_variance = False
    except ValueError:
        vg_ph.As = np.fft.fft(vg_ph.qq_std, n=T)
        vg_ph.phases = np.angle(vg_ph.As)
        A_new = vg_ph.As * np.exp(1j * phases)
        adjust_variance = True

    fft_sim = (np.fft.ifft(A_new)).real
    if adjust_variance:
        fft_sim /= fft_sim.std(axis=1)[:, None]

    # from lhglib.contrib.time_series_analysis import time_series as ts
    # fig, axs = plt.subplots(nrows=vg_obj.K, ncols=1, sharex=True)
    # for i, ax in enumerate(axs):
    #     freqs = np.fft.fftfreq(T)
    #     ax.bar(freqs, phases[i], width=.5 / T, label="surrogate")
    #     ax.bar(freqs, vg_ph.phases[i], width=.5 / T, label="data")
    # axs[0].legend(loc="best")

    # my.hist(vg_ph.phases.T, 20)

    # vg_ph.vine.plot(edge_labels="copulas")
    # vg_ph.vine.plot_tplom()
    # vg_ph.vine.plot_qqplom()
    # fig, axs = ts.plot_cross_corr(vg_ph.qq_std)
    # fig, axs = ts.plot_cross_corr(fft_sim, linestyle="--", axs=axs, fig=fig)
    # fig, axs = plt.subplots(nrows=vg_obj.K, ncols=1, sharex=True)
    # for i, ax in enumerate(axs):
    #     ax.plot(vg_ph.qq_std[i])
    #     ax.plot(fft_sim[i], linestyle="--")
    # plt.show()

    # change in mean scenario
    prim_i = vg_obj.primary_var_ii
    fft_sim[prim_i] += sc_pars.m[prim_i]
    fft_sim[prim_i] += sc_pars.m_t[prim_i]
    qq = np.array([dists.norm.cdf(values) for values in fft_sim])
    eps = 1e-12
    qq[qq > (1 - eps)] = 1 - eps
    qq[qq < eps] = eps
    ranks_sim = vg_ph.vine.simulate(randomness=qq)
    return np.array([dists.norm.ppf(ranks) for ranks in ranks_sim])


if __name__ == '__main__':
    # for deterministic networkx graphs!
    # np.random.seed(2)

    from lhglib.contrib.veathergenerator import vg, vg_plotting, vg_base
    from lhglib.contrib.veathergenerator import config_konstanz_disag as conf
    vg.conf = vg_plotting.conf = vg_base.conf = conf
    met_vg = vg.VG(("theta", "ILWR", "rh", "R"), verbose=True)
    ranks = np.array([stats.rel_ranks(values)
                      for values in met_vg.data_trans])
    cvine = CVine(ranks, varnames=met_vg.var_names,
                  # dtimes=met_vg.times,
                  # weights="likelihood"
                  )
    # quantiles = cvine.quantiles()
    # assert np.all(np.isfinite(quantiles))
    # sim = cvine.simulate(randomness=quantiles)

    # cov = [[1.5, 1., -1., 1.5],
    #        [1., 1., 0., .5],
    #        [-1., 0., 2., -.75],
    #        [1.5, .5, -.75, 1.5]]
    # data = np.random.multivariate_normal(len(cov) * [0], cov, 3000).T
    # data_ranks = np.array([stats.rel_ranks(row) for row in data])
    # varnames = "".join("%d" % i for i in range(len(cov)))
    # vine = RVine(data_ranks, varnames=varnames, verbose=True)
    # sim = vine.simulate()
    # plotting.ccplom(sim, k=0, title="simulated", opacity=.25,
    #                 kind="contourf",
    #                 varnames=varnames)
    # plotting.ccplom(data, k=0, title="data", opacity=.25,
    #                 kind="contourf",
    #                 varnames=varnames)
    # vine.plot_tplom()
    # fig, axs = vine.plot(edge_labels="copulas")
    # plt.show()

    # from weathercop import plotting
    # import ar_models as ar
    # data_filepath = os.path.join(cop_conf.weathercop_dir, "code",
    #                              "vg_data.npz")
    # varnames = "R theta ILWR Qsw u".split()
    # # varnames = "R theta ILWR Qsw".split()
    # data_varnames = "R theta Qsw ILWR rh u v".split()
    # # varnames = cop_conf.varnames
    # with np.load(data_filepath, encoding="bytes") as saved:
    #     data_summer = saved["summer"]
    #     data_winter = saved["winter"]
    #     data_all = saved["all"]
    #     dtimes = saved["dtimes"]
    #     # doys = saved["doys"]
    # for data_unordered, title in zip((data_summer, data_winter),
    #                                  ("summer", "winter")):
    #     # data = np.hstack((data_summer, data_winter))
    #     # data = data_all
    #     # data = np.vstack((doys[None, :] % 182, data))
    #     # data = np.vstack((data_all[None, 0], data))
    #     data = np.array([data_unordered[data_varnames.index(varname)]
    #                      for varname in varnames])
    #     p = 3
    #     K, T = data.shape
    #     B, sigma_u = ar.VAR_LS(data, p=p)
    #     data_residuals = ar.VAR_residuals(data, B, p=p)
    #     data_ranks = np.array([stats.rel_ranks(row) for row in data_residuals])
    #     vine = RVine(data_ranks, k=0, varnames=varnames,
    #                  verbose=True)
    #     # P = vine.quantiles()
    #     # residuals_sim_ranks = vine.simulate(randomness=P)
    #     # residuals_sim_ranks = vine.simulate()
    #     # residuals_sim_ranks = np.array([
    #     #     residuals_sim_ranks[i] for i in (1, 0, 3, 2, 4)])
    #     # residuals_sim_ranks = np.array([
    #     #     residuals_sim_ranks[vine.varnames_old.index(varname)]
    #     #     for varname in vine.varnames])

    #     # residuals_sim = np.array([spstats.distributions.norm.ppf(ranks)
    #     #                           for ranks in residuals_sim_ranks])
    #     # fig, axs = plt.subplots(K, sharex=True)
    #     # for i, ax in enumerate(axs):
    #     #     ax.plot(data_ranks[i], label="data")
    #     #     ax.plot(residuals_sim_ranks[i], label="cop sim")
    #     #     ax.set_title(varnames[i])
    #     #     ax.grid(True)
    #     #     if i == 0:
    #     #         ax.legend()
    #     # fig.suptitle(title)

    #     # for i, (dat, sim) in enumerate(zip(data_ranks, residuals_sim_ranks)):
    #     #     fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
    #     #     ax.scatter(dat, sim)
    #     #     ax.set_title(varnames[i])
    #     # import numpy.testing as npt
    #     # npt.assert_almost_equal(data_ranks, residuals_sim_ranks, decimal=3)
    #     # plt.show()
    #     cmap = "terrain"
    #     # plotting.ccplom(residuals_sim, k=0, x_bins=12, y_bins=12,
    #     #                 cmap=cmap, title="%s simulated" % title,
    #     #                 fontsize=12, opacity=.25, kind="img",
    #     #                 varnames=varnames)
    #     # plotting.ccplom(data_residuals, k=0, x_bins=12, y_bins=12,
    #     #                 cmap=cmap, title="%s data" % title,
    #     #                 fontsize=12, opacity=.25, kind="img",
    #     #                 varnames=varnames)
    #     vine.plot_tplom()
    #     vine.plot_qqplom()
    #     # plotting.ccplom(data_residuals, k=1, title="data (t-1)", opacity=.25,
    #     #                 kind="img",
    #     #                 varnames=varnames)
    #     # fig, axs = vine.plot(edge_labels="copulas")
    #     # fig.suptitle(title)
    #     # vine.plot_tplom()
    #     # cop_sim = ar.VAR_LS_sim(B, sigma_u, T=T, u=residuals_sim)
    #     # var_sim = ar.VAR_LS_sim(B, sigma_u, T=T)
    #     # plotting.ccplom(cop_sim, k=0, x_bins=12, y_bins=12, cmap=cmap,
    #     #                 title="%s cop sim" % title, opacity=.25,
    #     #                 kind="img", varnames=varnames)
    #     # plotting.ccplom(var_sim, k=0, x_bins=12, y_bins=12, cmap=cmap,
    #     #                 title="%s var sim" % title, opacity=.25,
    #     #                 kind="img", varnames=varnames)
    #     # fig, axs = plt.subplots(K, sharex=True)
    #     # for i, ax in enumerate(axs):
    #     #     ax.plot(data[i], label="data")
    #     #     ax.plot(cop_sim[i], label="cop sim")
    #     #     ax.plot(var_sim[i], label="VAR")
    #     #     ax.set_title(varnames[i])
    #     #     ax.grid(True)
    #     #     if i == 0:
    #     #         ax.legend()
    #     # fig.suptitle(title)

        break
    plt.show()

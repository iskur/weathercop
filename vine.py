import itertools
import os
import re
from collections import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy import stats as spstats
from tqdm import tqdm

from weathercop import cop_conf, copulae as cops, find_copula, stats


def flat_set(*args):
    """Returns a flatted set of all possibly nested iterables containing
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
    n1, n2 = sorted(list(node1 ^ node2))
    return "Copula_%d_%d" % (n1, n2)


def get_rank_label(node1, node2):
    return "ranks_%s" % get_clabel(node1, node2)


def get_cond_labels(node1, node2, prefix=""):
    if isinstance(node1, int):
        return prefix + str(node1), prefix + str(node2)
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


def set_edge_copulae(tree, tau_min):
    """Fits a copula to the ranks at all nodes and sets conditional
    ranks and copula methods as edge attributes.
    """
    for node1, node2 in sorted(tree.edges_iter()):
        edge = tree[node1][node2]
        ranks1_key, ranks2_key = get_cond_labels(node1, node2)
        ranks1 = edge["ranks_" + ranks1_key]
        ranks2 = edge["ranks_" + ranks2_key]
        if abs(edge["tau"]) > tau_min:
            copula = find_copula.mml_serial(ranks1, ranks2)
            # copula = find_copula.mml(ranks1, ranks2)
        else:
            copula = cop.independence.generate_fitted(None, None)
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
        # we depend on these keys to start with a capital "C" when
        # relabeling later on
        edge["Copula_%s_%s" % (clabel1[0], clabel2[0])] = copula
        edge["C^_%s" % clabel1] = copula.inv_cdf_given_v
        edge["C^_%s" % clabel2] = copula.inv_cdf_given_u
        edge["C_%s" % clabel1] = copula.cdf_given_v
        edge["C_%s" % clabel2] = copula.cdf_given_u


class Vine:

    def __init__(self, ranks, k=0, varnames=None, verbose=True):
        """Vine copula.

        Parameter
        ---------
        ranks : (d, T) array
            d: number of variables
            T: number of time steps
        k : int, optional
        time shift to insert between u and v. (u is shifted
            backwards)
        """
        self.ranks = ranks
        self.k = k
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
        self.trees = self._gen_trees(ranks)
        # property cache
        self._A = self._edge_map = None
        # this relabels nodes to have a natural-order vine array
        self.A

    def _gen_trees(self, ranks):
        """Generate the vine trees."""
        # minimum absolute tau to reject dependence at 5% significance
        # level -> use the independence copula, then (see Genest and
        # Favre, 2007)
        n = self.T - self.k
        tau_min = 1.96 / np.sqrt((9 * n * (n - 1)) /
                                 (2 * (2 * n + 5)))
        if self.verbose:
            print("Minimum tau for dependence: %.4f" % tau_min)

        # first tree
        full_graph = nx.complete_graph(self.d)
        for i, ui in enumerate(ranks):
            ranks1 = ui[:-self.k] if self.k > 0 else ui
            for j, vj in enumerate(ranks[i:], start=i):
                if i != j:
                    ranks2 = vj[self.k:] if self.k > 0 else vj
                    tau = spstats.kendalltau(ranks1, ranks2).correlation
                    # as networkx minimizes the spanning tree, we have
                    # to invert the weights
                    ranks1_key, ranks2_key = get_cond_labels(i, j,
                                                             "ranks_")
                    ranks_dict = {ranks1_key: ranks1,
                                  ranks2_key: ranks2}
                    full_graph.add_edge(i, j,
                                        weight=(1 - abs(tau)), tau=tau,
                                        **ranks_dict)
        tree = self._best_tree(full_graph)
        set_edge_copulae(tree, tau_min)
        trees = [tree]

        # 2nd to (d-1)th tree
        for l in range(1, self.d - 1):
            full_graph = nx.Graph()
            # edges from above or last iteration are now nodes
            last_tree = trees[-1]
            last_edges = sorted(last_tree.edges())
            full_graph.add_nodes_from(last_edges)
            for node1, node2 in itertools.combinations(last_edges, 2):
                # do these edges share a node?
                proximity_condition = set(node1) & set(node2)
                if len(proximity_condition) == 1:
                    edge1 = last_tree[node1[0]][node1[1]]
                    edge2 = last_tree[node2[0]][node2[1]]
                    ranks1_key, ranks2_key = get_cond_labels(node1, node2,
                                                             "ranks_")
                    ranks1 = edge1[ranks1_key]
                    ranks2 = edge2[ranks2_key]
                    tau = spstats.kendalltau(ranks1, ranks2).correlation
                    ranks_dict = {ranks1_key: ranks1,
                                  ranks2_key: ranks2}
                    full_graph.add_edge(node1, node2,
                                        weight=(1 - abs(tau)), tau=tau,
                                        **ranks_dict)
            tree = self._best_tree(full_graph)
            set_edge_copulae(tree, tau_min)
            trees += [tree]
        return trees

    def simulate(self, *args, **kwds):
        sim = self._simulate(*args, **kwds)
        # reorder variables according to input order
        return np.array([sim[self.varnames.index(varname)]
                         for varname in self.varnames_old])

    def __getitem__(self, key):
        """Access the vine tree nodes by row and column index of Vine.A."""
        try:
            row, col = key
        except ValueError:
            raise TypeError("Key must contain a row and column number.")
        if row >= col:
            raise IndexError("First index must be >= than second index.")
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
        self.varnames_old = self.varnames
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
                      for item in A.ravel()]).reshape((self.d, self.d))
        return A

    @property
    def A(self):
        if self._A is None:
            self._A = self._gen_A()
            # in case an edge map was built before
            self._edge_map = None
        return self._A

    def plot(self, edge_labels="nodes"):
        nrows = int(len(self.trees) / 2)
        if nrows * 2 < len(self.trees):
            nrows += 1
        fig, axs = plt.subplots(nrows, 2)
        axs = np.ravel(axs)
        for tree_i, (ax, tree) in enumerate(zip(axs, self.trees)):
            ax.set_title(r"$T_%d$" % tree_i, fontsize=20)
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
            nx.draw_networkx(tree, ax=ax, pos=pos, node_size=2000,
                             font_size=20, labels=labels)

            if edge_labels == "nodes":
                if tree_i == 0:
                    elabels = {(node1, node2): "%d%d" % (node1, node2)
                               for node1, node2 in tree.edges()}
                else:
                    elabels = {(node1, node2): get_label(node1, node2)
                               for node1, node2 in tree.edges()}
            elif edge_labels == "copulas":
                elabels = {}
                for node1, node2 in tree.edges():
                    edge = tree[node1][node2]
                    cop_key = get_cop_key(node1, node2)
                    cop_name = edge[cop_key].name[len("fitted "):]
                    tau = edge["tau"]
                    label = ("%s\n" % cop_name) + (r"$\tau=%.2f$" % tau)
                    elabels[(node1, node2)] = label
            else:
                elabels = None
            nx.draw_networkx_edge_labels(tree, ax=ax, pos=pos,
                                         edge_labels=elabels,
                                         font_size=15)
        for ax in axs:
            ax.set_axis_off()
        return fig, axs

    def plot_tplom(self, opacity=.25, s_kwds=None):
        """Plots all bivariate copulae with scattered (conditioned) ranks.
        """
        s_kwds = dict() if s_kwds is None else s_kwds
        x_slice = slice(None, None if self.k == 0 else -self.k)
        y_slice = slice(self.k, None)

        fig, axs = plt.subplots(len(self.trees), self.d - 1,
                                subplot_kw=dict(aspect="equal"))
        # first tree, showing actual observations
        tree = self.trees[0]
        for ax_i, (node1, node2) in enumerate(tree.edges()):
            ax = axs[0, ax_i]
            edge = tree[node1][node2]
            # ranks1 = edge["ranks_u"]
            # ranks2 = edge["ranks_v"]
            ranks1_key, ranks2_key = get_cond_labels(node1, node2)
            ranks1 = edge["ranks_" + ranks1_key]
            ranks2 = edge["ranks_" + ranks2_key]

            cop_key = get_cop_key(node1, node2)
            cop = edge[cop_key]
            cop.plot_density(ax=ax, kind="contour", scatter=False)
            ax.set_title("%s (%.2f)" % (cop.name[len("fitted "):],
                                        cop.likelihood))
            ax.scatter(ranks1[x_slice], ranks2[y_slice],
                       marker="o",
                       facecolors=(0, 0, 0, 0),
                       edgecolors=(0, 0, 0, opacity),
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
                cop.plot_density(ax=ax, kind="contour", scatter=False)
                ax.set_title("%s (%.2f)" % (cop.name[len("fitted "):],
                                            cop.likelihood))
                ax.scatter(ranks1[x_slice], ranks2[y_slice],
                           marker="o",
                           facecolors=(0, 0, 0, 0),
                           edgecolors=(0, 0, 0, opacity),
                           **s_kwds)
                ax.set_xlabel(ranks1_key)
                ax.set_ylabel(ranks2_key)
            for ax in axs[tree_i, len(edges):]:
                ax.set_axis_off()
        return fig, axs


class CVine(Vine):

    def _best_tree(self, full_graph):
        nodes = full_graph.nodes()
        taus = np.array([[1
                          if node1 == node2 else
                          full_graph[node1][node2]["tau"]
                          for node1 in nodes]
                         for node2 in nodes])
        central_node_i = np.argmax(np.sum(np.abs(taus), axis=0))
        central_node = nodes[central_node_i]
        other_nodes = sorted(list(set(nodes) - set([central_node])))
        new_graph = nx.Graph()
        for other_node in other_nodes:
            new_graph.add_edge(central_node, other_node,
                               full_graph[central_node][other_node])
        return new_graph

    def _simulate(self, T=None):
        """Simulate a sample of size T.

        Notes
        -----
        See Algorithm 15 on p. 291.
        """
        T_sim = self.T if T is None else T

        zero = 1e-15
        one = 1 - zero
        Ps = np.random.rand(self.d, T_sim)
        Us = np.empty_like(Ps)
        Us[0] = Ps[0]
        for t in tqdm(range(T_sim)):
            U, P = Us[:, t], Ps[:, t]
            U[1] = self[0, 1]["C^_1|0"](conditioned=P[1],
                                        condition=P[0])
            for j in range(2, self.d):
                q = P[j]
                for l in range(j - 1, -1, -1):
                    cop = self[l, j]["C^_%d|%d" % (j, l)]
                    q = cop(conditioned=q, condition=P[l])
                    q = max(zero, min(one, q))
                U[j] = q
            Us[:, t] = U
        return Us


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

    def _simulate(self, T=None):
        """Simulate a sample of size T.

        Notes
        -----
        See Algorithm 17 on p. 292.
        """
        T_sim = self.T if T is None else T

        zero = 1e-15
        one = 1 - zero
        A, M, I = self.A, self.M, self.I
        Ps = np.random.rand(self.d, T_sim)
        Us = np.empty_like(Ps)
        Us[0] = Ps[0]
        Q, V, Z = [np.empty_like(A, dtype=float) for _ in range(3)]
        for t in tqdm(range(T_sim)):
            Q[:] = V[:] = Z[:] = 0
            # Q[:] = V[:] = Z[:] = zero
            U, P = Us[:, t], Ps[:, t]
            U[1] = self[0, 1]["C^_1|0"](conditioned=P[1],
                                        condition=P[0])
            Q[1, 1] = P[1]
            if I[0, 1]:
                V[0, 1] = self[0, 1]["C_0|1"](conditioned=U[0],
                                              condition=U[1])
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
                                  condition=s)
                    Q[l, j] = min(one, max(zero, Q[l, j]))
                cop = self[0, j]["C^_%d|%d" % (j, A[0, j])]
                U[j] = Q[0, j] = max(zero,
                                     min(one,
                                         cop(conditioned=Q[1, j],
                                             condition=U[A[0, j]])))
                cop = self[0, j]["C_%d|%d" % (A[0, j], j)]
                V[0, j] = cop(conditioned=U[A[0, j]],
                              condition=U[j])
                for l in range(1, j):
                    if I[l, j]:
                        cop = self[l, j]["C_%d|%d" % (A[l, j], j)]
                        V[l, j] = cop(conditioned=Z[l, j],
                                      condition=Q[l, j])
            Us[:, t] = U
        return Us

if __name__ == '__main__':
    # for deterministic networkx graphs!
    np.random.seed(2)

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

    import ar_models as ar
    data_filepath = os.path.join(cop_conf.weathercop_dir, "code",
                                 "vg_data.npz")
    # varnames = "R theta ILWR Qsw".split()
    varnames = "R theta ILWR Qsw rh v u".split()
    data_varnames = "R theta Qsw ILWR rh u v".split()
    # varnames = cop_conf.varnames
    with np.load(data_filepath) as saved:
        data_summer = saved["summer"]
        data_winter = saved["winter"]
        data_all = saved["all"]
    for data, title in zip((data_winter, data_summer),
                           ("winter", "summer")):
        # data = np.hstack((data_summer, data_winter))
        data = np.array([data[data_varnames.index(varname)]
                         for varname in varnames])
        p = 3
        K, T = data.shape
        B, sigma_u = ar.VAR_LS(data, p=p)
        data_residuals = ar.VAR_residuals(data, B, p=p)
        data_ranks = np.array([stats.rel_ranks(row) for row in data_residuals])
        vine = RVine(data_ranks, k=0, varnames=varnames,
                     verbose=True)
        # fig, axs = vine.plot(edge_labels="copulas")
        # fig.suptitle(title)
        # vine.plot_tplom()
        residuals_sim_ranks = vine.simulate()
        residuals_sim = np.array([spstats.distributions.norm.ppf(ranks)
                                  for ranks in residuals_sim_ranks])
        plotting.ccplom(residuals_sim, k=0, title="simulated", opacity=.25,
                        kind="contourf",
                        varnames=varnames)
        plotting.ccplom(data_residuals, k=0, title="data", opacity=.25,
                        kind="contourf",
                        varnames=varnames)
        # plotting.ccplom(data_residuals, k=1, title="data (t-1)", opacity=.25,
        #                 kind="contourf",
        #                 varnames=varnames)
        cop_sim = ar.VAR_LS_sim(B, sigma_u, T=T, u=residuals_sim)
        var_sim = ar.VAR_LS_sim(B, sigma_u, T=T)
        plotting.ccplom(cop_sim, k=0, title="cop sim", opacity=.25,
                        kind="contourf",
                        varnames=varnames)
        plotting.ccplom(var_sim, k=0, title="var sim", opacity=.25,
                        kind="contourf",
                        varnames=varnames)
        fig, axs = plt.subplots(K, sharex=True)
        for i, ax in enumerate(axs):
            ax.plot(data[i], label="data")
            ax.plot(cop_sim[i], label="cop sim")
            ax.plot(var_sim[i], label="VAR")
            ax.set_title(varnames[i])
            ax.grid(True)
            if i == 0:
                ax.legend()
        break
    plt.show()

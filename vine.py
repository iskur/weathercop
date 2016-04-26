import os
import itertools
import re
from collections import Iterable
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as spstats
import networkx as nx
from weathercop import find_copula, cop_conf, stats, plotting
from weathercop import copulae as cop


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


def get_rank_label(node1, node2):
    return "ranks_%s" % get_clabel(node1, node2)


def get_cond_labels(node1, node2, prefix=""):
    condition1 = flat_set(node1[0]) & flat_set(node1[1])
    condition2 = flat_set(node2[0]) & flat_set(node2[1])
    conditioned1 = flat_set(node1[0]) ^ flat_set(node1[1])
    conditioned2 = flat_set(node2[0]) ^ flat_set(node2[1])
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


class RVine:

    def __init__(self, ranks, k=0, varnames=None, verbose=True):
        """Regular vine copula.

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
            self.varnames = varnames
        self.verbose = verbose
        self.trees = self._gen_trees(ranks)
        # property cache
        self._A = self._edge_map = None
        # this relabels nodes to have a natural-order vine array
        self.A

    def _gen_trees(self, ranks):
        """Generate the vine trees.

        Notes
        -----
        Implements Algorithm 30, p. 304., which is a greedy algorithm
        that maximizes the sum of kendall's tau for each tree.
        """
        # minimum absolute tau to reject dependence at 5% significance
        # level -> use the independence copula, then (see Genest and
        # Favre 2007)
        n = self.T - self.k
        tau_min = 1.96 / np.sqrt((9 * n * (n - 1)) /
                                 (2 * (2 * n + 5)))
        if self.verbose:
            print("Minimum tau for dependence: %.4f" % tau_min)

        # first tree
        full_graph = nx.complete_graph(self.d)
        for i, ui in enumerate(ranks):
            ranks_u = ui[:-self.k] if self.k > 0 else ui
            for j, vj in enumerate(ranks[i:], start=i):
                if i != j:
                    ranks_v = vj[self.k:] if self.k > 0 else vj
                    tau = spstats.kendalltau(ranks_u, ranks_v).correlation
                    # as networkx minimizes the spanning tree, we have
                    # to invert the weights
                    full_graph.add_edge(i, j, weight=(1 - abs(tau)),
                                        ranks_u=ranks_u,
                                        ranks_v=ranks_v,
                                        tau=tau)
        tree = nx.minimum_spanning_tree(full_graph)
        for node1, node2 in tree.edges_iter():
            edge = tree[node1][node2]
            ranks_u = ranks[node1, :-self.k] if self.k > 0 else ranks[node1]
            ranks_v = ranks[node2, self.k:] if self.k > 0 else ranks[node2]
            if abs(edge["tau"]) > tau_min:
                copula = find_copula.mml_serial(ranks_u, ranks_v)
            else:
                copula = cop.independence.generate_fitted(None, None)
            edge["copula"] = copula
            clabel1 = get_clabel(node1, node2)
            clabel2 = get_clabel(node2, node1)
            edge["ranks_%s" % clabel1] = \
                copula.cdf_given_v(conditioned=ranks_u, condition=ranks_v)
            edge["ranks_%s" % clabel2] = \
                copula.cdf_given_u(conditioned=ranks_v, condition=ranks_u)
            edge["C^_%s" % clabel1] = copula.inv_cdf_given_v
            edge["C^_%s" % clabel2] = copula.inv_cdf_given_u
            edge["C_%s" % clabel1] = copula.cdf_given_v
            edge["C_%s" % clabel2] = copula.cdf_given_u
        trees = [tree]

        # 2nd to (d-1)th tree
        for l in range(1, self.d - 1):
            full_graph = nx.Graph()
            # edges from above or last iteration are now nodes
            last_tree = trees[-1]
            last_edges = last_tree.edges()
            full_graph.add_nodes_from(last_edges)
            for node1, node2 in itertools.combinations(last_edges, 2):
                proximity_condition = set(node1) & set(node2)
                if len(proximity_condition) == 1:
                    edge1 = last_tree[node1[0]][node1[1]]
                    edge2 = last_tree[node2[0]][node2[1]]
                    ranks1_key, ranks2_key = get_cond_labels(node1, node2)
                    ranks1 = edge1["ranks_%s" % ranks1_key]
                    ranks2 = edge2["ranks_%s" % ranks2_key]
                    tau = spstats.kendalltau(ranks1, ranks2).correlation
                    ranks_dict = {ranks1_key: ranks1,
                                  ranks2_key: ranks2}
                    full_graph.add_edge(node1, node2,
                                        weight=(1 - abs(tau)), tau=tau,
                                        **ranks_dict)
            tree = nx.minimum_spanning_tree(full_graph)
            for node1, node2 in tree.edges_iter():
                edge = tree[node1][node2]
                ranks1_key, ranks2_key = get_cond_labels(node1, node2)
                ranks1 = edge[ranks1_key]
                ranks2 = edge[ranks2_key]

                if abs(edge["tau"]) > tau_min:
                    copula = find_copula.mml_serial(ranks1, ranks2)
                else:
                    copula = cop.independence.generate_fitted(None, None)
                edge["copula"] = copula
                clabel1 = get_clabel(node1, node2)
                clabel2 = get_clabel(node2, node1)
                edge["ranks_%s" % clabel1] = \
                    copula.cdf_given_v(conditioned=ranks2, condition=ranks1)
                edge["ranks_%s" % clabel2] = \
                    copula.cdf_given_u(conditioned=ranks1, condition=ranks2)
                # the preconditioned set would make retrieving too complicated
                clabel1 = clabel1[:clabel1.index(";")]
                clabel2 = clabel2[:clabel2.index(";")]
                edge["C^_%s" % clabel1] = copula.inv_cdf_given_v
                edge["C^_%s" % clabel2] = copula.inv_cdf_given_u
                edge["C_%s" % clabel1] = copula.cdf_given_v
                edge["C_%s" % clabel2] = copula.cdf_given_u
            trees += [tree]
        return trees

    def _gen_A(self):
        """Vine array."""
        A = -np.ones((self.d, self.d), dtype=int)
        for tree_i, tree in enumerate(self.trees[::-1]):
            row = self.d - tree_i - 1
            if row < self.d - 1:
                A[row, row] = A[row, row + 1]
            if row == 1:
                A[0, 0] = (set(range(self.d)) - set(np.diag(A))).pop()
            for node1, node2 in tree.edges():
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
        # and don't forget the variable names!
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

        # relabel nodes (edges are relabeled implicitely)
        new_trees = []
        for tree in self.trees:
            new_tree = nx.relabel_nodes(tree, relabel_func)
            # relabel edge and node attribute keys
            for i, j in new_tree.edges():
                old_dict = {key: val for key, val in new_tree[i][j].items()}
                new_dict = {}
                for key, val in old_dict.items():
                    new_key = key.translate(relable_table)
                    # we expect to have an ordered preconditioned set!
                    part1, *part2 = new_key.split(";")
                    if part2:
                        part2 = "".join(sorted(*part2))
                        new_key = ";".join((part1, part2))
                    new_dict[new_key] = val
                new_tree[i][j] = new_dict
            new_trees.append(new_tree)
        self.trees = new_trees

        # in case an edge map was built before
        self._edge_map = None

        A = np.array([-1 if item == -1 else relabel_mapping[item]
                      for item in A.ravel()]).reshape((self.d, self.d))
        return A

    @property
    def edge_map(self):
        if self._edge_map is None:
            _edge_map = [{(tuple(sorted(flat_set(node1) ^ flat_set(node2))),
                           tuple(sorted(flat_set(node1) & flat_set(node2)))):
                          tree[node1][node2]
                          for node1, node2 in tree.edges()}
                         for tree in self.trees]
            self._edge_map = _edge_map
        return self._edge_map

    def __getitem__(self, key):
        """Access the vine tree nodes by row and column index of RVine.A."""
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
    def A(self):
        if self._A is None:
            self._A = self._gen_A()
        return self._A

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

    def simulate(self, T=None):
        """Simulate a sample of size T.

        Notes
        -----
        See Algorithm 17 on p. 292.
        """
        T_sim = self.T if T is None else T

        zero = 1e-15
        A, M, I = self.A, self.M, self.I
        Ps = np.random.rand(self.d, T_sim)
        Us = np.empty_like(Ps)
        Us[0] = Ps[0]
        Q, V, Z = [np.empty_like(A, dtype=float) for _ in range(3)]
        for t in tqdm(range(T_sim)):
            Q[:] = V[:] = Z[:] = 0
            U, P = Us[:, t], Ps[:, t]
            # U[1] = self[0, 1]["C^_0|1"](P[0], P[1])
            U[1] = self[0, 1]["C^_0|1"](conditioned=P[0],
                                        condition=P[1])
            Q[1, 1] = P[1]
            if I[0, 1]:
                # V[0, 1] = self[0, 1]["C_1|0"](U[0], U[1])
                V[0, 1] = self[0, 1]["C_1|0"](conditioned=U[1],
                                              condition=U[0])
            for j in range(2, self.d):
                Q[j, j] = P[j]
                for l in range(j - 1, 0, -1):
                    if A[l, j] == M[l, j]:
                        s = Q[l, A[l, j]]
                    else:
                        s = V[l - 1, M[l, j]]
                    Z[l, j] = s
                    # Q[l + 1, j] = max(zero, Q[l + 1, j])
                    cop = self[l, j]["C^_%d|%d" % (j, A[l, j])]
                    # Q[l, j] = cop(Q[l + 1, j], s)
                    Q[l, j] = cop(conditioned=Q[l + 1, j],
                                  condition=s)
                    # Q[l, j] = max(zero, Q[l, j])
                cop = self[0, j]["C^_%d|%d" % (j, A[0, j])]
                # U[j] = Q[0, j] = cop(Q[1, j], U[A[0, j]])
                U[j] = Q[0, j] = cop(conditioned=Q[1, j],
                                     condition=U[A[0, j]])
                cop = self[0, j]["C_%d|%d" % (A[0, j], j)]
                # V[0, j] = cop(U[A[0, j]], U[j])
                V[0, j] = cop(conditioned=U[A[0, j]],
                              condition=U[j])
                for l in range(1, j - 1):
                    if I[l, j]:
                        cop = self[l, j]["C_%d|%d" % (A[l, j], j)]
                        # V[l, j] = cop(Z[l, j], Q[l, j])
                        V[l, j] = cop(conditioned=Z[l, j],
                                      condition=Q[l, j])

            Us[:, t] = U
        return Us

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
            ranks1 = edge["ranks_u"]
            ranks2 = edge["ranks_v"]
            cop = edge["copula"]
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
            edges = tree.edges()
            for ax_i, (node1, node2) in enumerate(edges):
                ax = axs[tree_i, ax_i]
                edge = tree[node1][node2]
                ranks1_key, ranks2_key = get_cond_labels(node1, node2)
                ranks1 = edge[ranks1_key]
                ranks2 = edge[ranks2_key]
                cop = edge["copula"]
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
                elabels = {(i, j):
                           (("%s\n" %
                             tree[i][j]["copula"].name[len("fitted "):]) +
                            (r"$\tau=%.2f$" % tree[i][j]["tau"]))
                           for i, j in tree.edges()}
            else:
                elabels = None
            nx.draw_networkx_edge_labels(tree, ax=ax, pos=pos,
                                         edge_labels=elabels,
                                         font_size=15)
        for ax in axs:
            ax.set_axis_off()
        return fig, axs

if __name__ == '__main__':
    # for deterministic networkx graphs!
    np.random.seed(2)
    data_filepath = os.path.join(cop_conf.weathercop_dir, "code",
                                 "vg_data.npz")
    varnames = "R theta Qsw ILWR rh u v".split()
    data_varnames = "R theta Qsw ILWR rh u v".split()
    # varnames = cop_conf.varnames
    with np.load(data_filepath) as saved:
        data_summer = saved["summer"]
        data_winter = saved["winter"]
    for data, title in zip((data_summer, data_winter),
                           ("summer", "winter")):
        # data = np.hstack((data_summer, data_winter))
        data = np.array([data[data_varnames.index(varname)]
                         for varname in varnames])
        data_ranks = np.array([stats.rel_ranks(row) for row in data])
        rvine = RVine(data_ranks, k=0, varnames=varnames,
                      verbose=True)
        # rvine.plot_tplom()
        sim = rvine.simulate()
        sim = np.array([sim[rvine.varnames.index(name)]
                        for name in varnames])
        # plotting.ccplom(sim, k=0, title="simulated", opacity=.25,
        #                 kind="contour",
        #                 varnames=varnames)
        # plotting.ccplom(data_summer, k=0, title="data", opacity=.25,
        #                 kind="contour",
        #                 varnames=varnames)
        # fig, axs = rvine.plot(edge_labels="copulas")
        # fig.suptitle(title)
        break
    # plt.show()

import os
import itertools
import re
from collections import Iterable
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as spstats
import networkx as nx
from weathercop import find_copula, cop_conf, stats
from weathercop import copulae as cop


def flat_set(*args):
    """Returns a flatted set of all possibly nested iterables containing
    integers in args.
    """
    # muhahaha!
    return set(int(match) for match
               in re.findall(r"[0-9]+", repr(args)))


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
        self.k = k
        self.d, self.T = ranks.shape
        if varnames is None:
            self.varnames = list(range(self.d))
        else:
            self.varnames = varnames
        self.verbose = verbose
        self.trees = self._gen_trees(ranks)
        # this relabels nodes to have a natural-order vine array
        self.A

    def _gen_trees(self, ranks):
        """Generate the vine trees.

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
            for j, vj in enumerate(ranks):
                if i != j:
                    ranks_v = vj[self.k:] if self.k > 0 else vj
                    tau = spstats.kendalltau(ranks_u, ranks_v).correlation
                    # as networkx minimizes the spanning tree, we have
                    # to invert the weights
                    full_graph.add_edge(i, j, weight=(1 - abs(tau)),
                                        tau=tau)
        tree = nx.minimum_spanning_tree(full_graph)
        for i, j in tree.edges_iter():
            edge = tree[i][j]
            ranks_u = ranks[i, :-self.k] if self.k > 0 else ranks[i]
            ranks_v = ranks[j, self.k:] if self.k > 0 else ranks[j]
            if abs(edge["tau"]) > tau_min:
                copula = find_copula.mml_serial(ranks_u, ranks_v)
            else:
                copula = cop.independence.generate_fitted(None, None)
            edge["copula"] = copula
            edge["ranks_given_u"] = copula.cdf_given_u(ranks_u, ranks_v)
            edge["ranks_given_v"] = copula.cdf_given_v(ranks_u, ranks_v)
        trees = [tree]

        # 2nd to (d-1)th tree
        for l in range(2, self.d):
            full_graph = nx.Graph()
            # last edges are now nodes
            last_tree = trees[-1]
            last_edges = last_tree.edges()
            full_graph.add_nodes_from(last_edges)
            for node1, node2 in itertools.combinations(last_edges, 2):
                # proximity condition
                if len(set(node1) & set(node2)) == 1:
                    ranks1_u = last_tree[node1[0]][node1[1]]["ranks_given_u"]
                    ranks2_v = last_tree[node2[0]][node2[1]]["ranks_given_v"]
                    tau = spstats.kendalltau(ranks1_u, ranks2_v).correlation
                    full_graph.add_edge(node1, node2,
                                        weight=(1 - abs(tau)), tau=tau,
                                        ranks1_u=ranks1_u, ranks2_v=ranks2_v)
            tree = nx.minimum_spanning_tree(full_graph)
            for i, j in tree.edges_iter():
                edge = tree[i][j]
                if abs(edge["tau"]) > tau_min:
                    copula = find_copula.mml_serial(edge["ranks1_u"],
                                                    edge["ranks2_v"])
                else:
                    copula = cop.independence.generate_fitted(None, None)
                edge["copula"] = copula
                edge["ranks_given_u"] = copula.cdf_given_u(edge["ranks1_u"],
                                                           edge["ranks2_v"])
                edge["ranks_given_v"] = copula.cdf_given_v(edge["ranks1_u"],
                                                           edge["ranks2_v"])
            trees += [tree]
        return trees

    @property
    def A(self):
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
            return tuple(container)

        new_trees = []
        for tree in self.trees:
            new_trees += [nx.relabel_nodes(tree, relabel_func)]
        self.trees = new_trees

        A = np.array([-1 if item == -1 else relabel_mapping[item]
                      for item in A.ravel()]).reshape((self.d, self.d))
        return A

    def plot(self, edge_kind="nodes"):
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
                labels = {}
                for node1, node2 in tree.nodes():
                    key = node1, node2
                    conditioned = "".join(["%d" % no for no in
                                           flat_set(node1) ^
                                           flat_set(node2)])
                    condition = "".join(["%d" % no for no in
                                         flat_set(node1) &
                                         flat_set(node2)])
                    val = "%s|%s" % (conditioned, condition)
                    labels[key] = val
            pos = nx.spring_layout(tree)
            nx.draw_networkx(tree, ax=ax, pos=pos, node_size=2000,
                             font_size=20, labels=labels)

            if edge_kind == "nodes":
                if tree_i == 0:
                    edge_labels = {(node1, node2): "%d%d" % (node1, node2)
                                   for node1, node2 in tree.edges()}
                else:
                    edge_labels = {}
                    for node1, node2 in tree.edges():
                        key = node1, node2
                        conditioned = "".join(["%d" % no for no in
                                               flat_set(node1) ^
                                               flat_set(node2)])
                        condition = "".join(["%d" % no for no in
                                             flat_set(node1) &
                                             flat_set(node2)])
                        val = "%s|%s" % (conditioned, condition)
                        edge_labels[key] = val
            elif edge_kind == "copula":
                edge_labels = {(i, j):
                               (("%s\n" %
                                 tree[i][j]["copula"].name[len("fitted "):]) +
                                (r"$\tau=%.2f$" % tree[i][j]["tau"]))
                               for i, j in tree.edges()}
            nx.draw_networkx_edge_labels(tree, ax=ax, pos=pos,
                                         edge_labels=edge_labels,
                                         font_size=15)
            # from networkx.drawing.nx_agraph import graphviz_layout
            # nx.drawing.graphviz_layout = graphviz_layout
            # # pos = graphviz_layout(tree)
            # nx.draw_graphviz(tree, ax=ax,  # pos=pos,
            #                  node_size=2000,
            #                  font_size=20, labels=labels)
        for ax in axs:
            ax.set_axis_off()
        return fig, axs

if __name__ == '__main__':
    # for deterministic networkx graphs!
    np.random.seed(2)
    data_filepath = os.path.join(cop_conf.weathercop_dir, "code",
                                 "vg_data.npz")
    with np.load(data_filepath) as saved:
        data_summer = saved["summer"]
        data_winter = saved["winter"]
    # data = np.hstack((data_summer, data_winter))
    for data, title in zip((data_summer, data_winter),
                           ("summer", "winter")):
        data_ranks = np.array([stats.rel_ranks(row) for row in data])
        rvine = RVine(data_ranks, k=0, varnames=cop_conf.var_names,
                      verbose=True)
        fig, axs = rvine.plot(edge_kind="nodes")
        fig.suptitle(title)
        print(rvine.A)
    plt.show()

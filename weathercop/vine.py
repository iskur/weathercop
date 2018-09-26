import itertools
import multiprocessing
import re
from collections import Iterable
from contextlib import suppress
from itertools import repeat

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pyximport
import scipy.stats as spstats
from tqdm import tqdm
# import pyximport

from vg import helpers as my
from vg.time_series_analysis import distributions as dists
from weathercop import (cop_conf, copulae as cops, find_copula,
                        seasonal_cop, stats, tools)

# pyximport.install(setup_args={
#     # 'include_dirs': np.get_include(),
#     'include_dirs': [np.get_include(), "~/anaconda/envs/python3/include/"]
#     # 'gdb_debug': True
#     },
#                   reload_support=True)
from weathercop import normal_conditional


if cop_conf.PROFILE:
    def csim_one(args):
        # this is a function and not a method, so it can be pickled by
        # multiprocessing
        t, P, cvine, zero, one = args
        U = np.empty(len(P))
        U[0] = P[0]
        U[1] = cvine[0, 1]["C^_1|0"](conditioned=P[None, 1],
                                     condition=P[None, 0],
                                     t=t)
        for j in range(2, cvine.d):
            q = P[None, j]
            for l in range(j - 1, -1, -1):
                cop = cvine[l, j][f"C^_{j}|{l}"]
                q = cop(conditioned=q,
                        condition=P[None, l],
                        t=t)
            U[j] = q
        return U
else:
    pyximport.install(setup_args={'include_dirs': np.get_include(),
                                  # 'gdb_debug': True
                                  },
                      reload_support=True)
    from weathercop.cvine import csim_one


if cop_conf.PROFILE:
    def cquant_one(args):
        t, U, cvine, zero, one = args
        P = np.empty(len(U))
        P[0] = U[0]
        P[1] = cvine[0, 1]["C_1|0"](conditioned=U[None, 1],
                                    condition=P[None, 0],
                                    t=t)
        P[1] = max(zero, min(one, P[1]))
        for j in range(2, cvine.d):
            q = U[None, j]
            for l in range(j):
                cop = cvine[l, j][f"C_{j}|{l}"]
                # q_old = q[0]
                q = cop(conditioned=q,
                        condition=P[None, l],
                        t=t)
            P[j] = q
        return P
else:
    pyximport.install(setup_args={'include_dirs': np.get_include(),
                                  # 'gdb_debug': True
                                  },
                      reload_support=True)
    from weathercop.cvine import cquant_one


n_nodes = multiprocessing.cpu_count() - 1


def clear_vine_cache():
    for suffix in ("bak dat dir".split()):
        with suppress(FileNotFoundError):
            cop_conf.vine_cache.with_suffix(f".she.{suffix}").unlink()


def flat_set(*args):
    """Returns a flattened set of all possibly nested iterables containing
    integers in args.

    """
    # muhahaha!
    return set(int(match) for match
               in re.findall(r"[0-9]+", repr(args)))


def get_label(node1, node2):
    node1, node2 = map(flat_set, (node1, node2))
    conditioned = "".join(["%d" % no for no in (node1 ^ node2)])
    condition = "".join(["%d" % no for no in (node1 & node2)])
    return f"{conditioned}|{condition}"


def get_clabel(node1, node2, prefix=""):
    node1, node2 = map(flat_set, (node1, node2))
    c1 = tuple(node2 - node1)[0]
    c2 = tuple(node1 - node2)[0]
    condition = "".join([f"{no}" for no in sorted(node1 & node2)])
    clabel = f"{c1}|{c2}"
    if condition:
        clabel = f"{clabel};{condition}"
    return prefix + clabel


def get_last_nodes(node1, node2):
    node1, node2 = map(flat_set, (node1, node2))
    n1 = (node1 - node2).pop()
    n2 = (node2 - node1).pop()
    return n1, n2


def get_cop_key(node1, node2):
    n1, n2 = get_last_nodes(node1, node2)
    return f"Copula_{n1}_{n2}"


def get_cond_labels(node1, node2, prefix=""):
    if isinstance(node1, int):
        key1 = f"{prefix}{node1}|{node2}"
        key2 = f"{prefix}{node2}|{node1}"
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
             "".join([f"{no}" for no in sorted(condition1)])))
    key2 = ("%s%s|%s;%s" %
            (prefix,
             tuple(q)[0],
             tuple(conditioned2 - q)[0],
             "".join([f"{no}" for no in sorted(condition2)])))
    return key1, key2


def set_edge_copulae(tree, *, tau_min=None, names=None, verbose=True,
                     **kwds):
    """Fits a copula to the ranks at all nodes and sets conditional
    ranks and copula methods as edge attributes.
    """
    for node1, node2 in tree.edges():
        node1, node2 = sorted((node1, node2))
        edge = tree[node1][node2]
        ranks1_key, ranks2_key = get_cond_labels(node1, node2)
        ranks1 = edge[f"ranks_{ranks1_key}"]
        ranks2 = edge[f"ranks_{ranks2_key}"]
        cop_key = get_cop_key(node1, node2)
        if tau_min is None:
            try:
                copula = edge[cop_key]
            except KeyError:
                print("switched %s -> %s" % (cop_key,
                                             get_cop_key(node2, node1)))
                node1, node2 = node2, node1
                ranks1_key, ranks2_key = get_cond_labels(node1, node2)
                ranks1 = edge[f"ranks_{ranks1_key}"]
                ranks2 = edge[f"ranks_{ranks2_key}"]
                cop_key = get_cop_key(node1, node2)
                copula = edge[cop_key]
        else:
            if abs(edge["tau"]) > tau_min:
                if names:
                    print(f"{names[node1]} - {names[node2]}")
                copula = find_copula.mml(ranks1, ranks2,
                                         plot=False,
                                         verbose=verbose, **kwds)
            else:
                if verbose:
                    print("chose independence")
                copula = cops.independence.generate_fitted(ranks1, ranks2)
        clabel1 = get_clabel(node2, node1)  # u|v
        clabel2 = get_clabel(node1, node2)  # v|u
        edge[f"ranks_{clabel1}"] = copula.cdf_given_v(conditioned=ranks1,
                                                      condition=ranks2)
        edge[f"ranks_{clabel2}"] = copula.cdf_given_u(conditioned=ranks2,
                                                      condition=ranks1)

        if ";" in clabel1:
            # the preconditioned set would make retrieving overly complicated
            clabel1 = clabel1[:clabel1.index(";")]
            clabel2 = clabel2[:clabel2.index(";")]
        if tau_min is not None:
            edge[cop_key] = copula
        # we depend on these keys to start with a capital "C" when
        # relabeling later on
        edge[f"C^_{clabel1}"] = copula.inv_cdf_given_v
        edge[f"C^_{clabel2}"] = copula.inv_cdf_given_u
        edge[f"C_{clabel1}"] = copula.cdf_given_v
        edge[f"C_{clabel2}"] = copula.cdf_given_u


class Vine:

    def __init__(self, ranks, k=0, varnames=None, verbose=True,
                 build_trees=True, dtimes=None, weights="tau",
                 tau_min=None, debug=False, **kwds):
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
        tau_min : None or float, optional
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
        self.debug = debug
        self.weights = weights
        self.tau_min = tau_min
        if build_trees:
            self.trees = self._gen_trees(ranks)
            # property cache
            self._A = self._edge_map = None
            # this relabels nodes to have a natural-order vine array
            self.A

    @property
    def tau_min(self):
        if self._tau_min is None:
            if self.weights != "tau":
                return None
            # minimum absolute tau to reject dependence at 5% significance
            # level -> use the independence copula, then (see Genest and
            # Favre, 2007)
            n = self.T - self.k
            return 1.96 / np.sqrt((9 * n * (n - 1)) /
                                  (2 * (2 * n + 5)))
        return self._tau_min

    @tau_min.setter
    def tau_min(self, value):
        self._tau_min = value

    @tau_min.deleter
    def tau_min(self):
        self._tau_min = None

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
                                 "ranks_v": ranks2,
                                 # some additional information for
                                 # later introspection
                                 "tree": 0,
                                 "node_u": node1,
                                 "node_v": node2}
                    finite_mask = np.isfinite(ranks1) & np.isfinite(ranks2)
                    tau = spstats.kendalltau(ranks1[finite_mask],
                                             ranks2[finite_mask]).correlation
                    assert np.isfinite(tau)
                    if self.weights == "tau":
                        # as networkx minimizes the spanning tree, we have
                        # to invert the weights
                        weight = 1 - abs(tau)
                    elif self.weights == "likelihood":
                        cop_key = get_cop_key(node1, node2)
                        copula = find_copula.mml(ranks1,
                                                 ranks2,
                                                 verbose=self.verbose,
                                                 dtimes=self.dtimes,
                                                 plot=False
                                                 )
                        weight = -copula.likelihood
                        edge_dict[cop_key] = copula
                    full_graph.add_edge(node1, node2,
                                        weight=weight, tau=tau,
                                        **edge_dict)
        tree = self._best_tree(full_graph, tree_i=0)
        return tree

    def _gen_trees(self, ranks):
        """Generate the vine trees."""
        if self.verbose and self.weights == "tau":
            print(f"Minimum tau for dependence: {self.tau_min:.4f}")

        if self.verbose:
            print(f"Building tree 1 of {self.d - 1}")
        tree = self._gen_first_tree(ranks)
        set_edge_copulae(tree, tau_min=self.tau_min,
                         dtimes=self.dtimes, names=self.varnames)
        trees = [tree]

        # 2nd to (d-1)th tree
        for l in range(1, self.d - 1):
            if self.verbose:
                print(f"Building tree {l + 1} of {self.d - 1}")
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
                    finite_mask = np.isfinite(ranks1) & np.isfinite(ranks2)
                    tau = spstats.kendalltau(ranks1[finite_mask],
                                             ranks2[finite_mask]).correlation
                    edge_dict = {ranks1_key: ranks1,
                                 ranks2_key: ranks2,
                                 # some additional information for
                                 # later introspection
                                 "tree": l,
                                 "node_u": get_last_nodes(node1, node2)[0],
                                 "node_v": get_last_nodes(node2, node1)[0],
                                 "node_u_full": node1,
                                 "node_v_full": node2}
                    if self.weights == "tau":
                        weight = 1 - abs(tau)
                    elif self.weights == "likelihood":
                        copula = find_copula.mml(ranks1,
                                                 ranks2,
                                                 verbose=self.verbose,
                                                 dtimes=self.dtimes,
                                                 plot=False
                                                 )
                        clabel1 = get_clabel(node2, node1)[0]  # u|v
                        clabel2 = get_clabel(node1, node2)[0]  # v|u
                        cop_key = f"Copula_{clabel1}_{clabel2}"
                        edge_dict[cop_key] = copula
                        weight = -copula.likelihood
                    full_graph.add_edge(node1, node2,
                                        weight=weight, tau=tau,
                                        **edge_dict)
            tree = self._best_tree(full_graph, tree_i=l)
            set_edge_copulae(tree, tau_min=self.tau_min, dtimes=self.dtimes)
            trees += [tree]
        return trees

    def simulate(self, randomness=None, *args, **kwds):
        if randomness is not None:
            randomness = np.array([randomness[self.varnames_old.index(name)]
                                   for name in self.varnames])
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
        # we assume the variables were given in the old/outside order
        ranks = np.array([ranks[self.varnames_old.index(name)]
                          for name in self.varnames])
        Ps = self._quantiles(ranks, *args, **kwds)
        return np.array([Ps[self.varnames.index(name_old)]
                         for name_old in self.varnames_old])

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

    def __iter__(self):
        ii, jj = np.triu_indices_from(self.A, k=1)
        return iter(self[i, j] for i, j in zip(ii, jj))

    def __str__(self):
        return "\n".join("tree {tree}: {node_u} <-> {node_v}"
                         .format(**edge)
                         for edge in self)

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
        relabel_table = "".maketrans(olds, news)
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
            return tuple(container)

        # relabel nodes and edges
        if self.debug:
            print(f"\nRelabeling nodes: {relabel_mapping}")
        new_trees = []
        for tree_i, tree in enumerate(self.trees):
            new_tree = nx.Graph()
            if self.debug:
                print(f"Tree: {tree_i}")
            for node1, node2 in tree.edges():
                edge = tree[node1][node2]
                node1_new, node2_new = map(relabel_func,
                                           (edge["node_u"],
                                            edge["node_v"]))
                if self.debug and (node1_new > node2_new):
                    print(f"{node1} -> {node1_new},", end=" ")
                    print(f"{node2} -> {node2_new}")
                    print(f"({node1}, {node2}) -> ({node1_new}, {node2_new})")
                new_dict = {}
                for key, val in tree[node1][node2].items():
                    new_key = key.translate(relabel_table)
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
                    if (self.debug and
                            (node1_new > node2_new) and
                            (new_key != key)):
                        print(f"{key} -> {new_key}")
                    if new_key == "node_u":
                        val = node1_new
                    elif new_key == "node_v":
                        val = node2_new
                    new_dict[new_key] = val
                node1_new, node2_new = map(relabel_func, (node1, node2))
                new_tree.add_edge(node1_new, node2_new, **new_dict)
            if self.debug:
                print()
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

    def plot(self, edge_labels="copulas", fig=None, axs=None, figsize=None,
           *args, **kwds):
        """Plot vine structure.

        Parameter
        ---------
        edge_labels : "nodes" or "copulas", optional
        """
        np.random.seed(0)
        nrows = int(len(self.trees) / 2)
        if nrows * 2 < len(self.trees):
            nrows += 1
        node_fontsize = kwds.get("node_fontsize",
                                 mpl.rcParams["legend.fontsize"])
        edge_fontsize = kwds.get("edge_fontsize",
                                 mpl.rcParams["xtick.labelsize"])
        node_size = 75 * node_fontsize
        if fig is None and axs is None:
            fig, axs = plt.subplots(nrows, 2, figsize=figsize,
                                    constrained_layout=True)
        axs = np.ravel(axs)
        for tree_i, (ax, tree) in enumerate(zip(axs, self.trees)):
            ax.set_title(rf"$\mathcal{{T}}_{tree_i}$")
            if tree_i == 0:
                labels = {i: f"{i}: {self.varnames[i]}"
                          for i, varname in enumerate(self.varnames)}
            elif tree_i == 1:
                labels = {(node1, node2): f"{node1}{node2}"
                          for node1, node2 in tree.nodes()}
            else:
                labels = {(node1, node2): get_label(node1, node2)
                          for node1, node2 in tree.nodes()}
            node_dist = .4
            if len(tree.nodes()) > 3:
                pos = nx.spring_layout(tree)
            elif len(tree.nodes()) == 3:
                # draw the nodes on a line and make sure the central node
                # is in the middle
                central_node = max(tree, key=dict(nx.degree(tree)).get)
                other_nodes = [node for node in tree.nodes()
                               if node != central_node]
                pos = {central_node: np.array([0, 0]),
                       other_nodes[0]: np.array([-node_dist, 0]),
                       other_nodes[1]: np.array([node_dist, 0]),
                       }
            else:
                pos = {node: np.array([node_i, 0])
                       for node_i, node in enumerate(tree.nodes())}
            nx.draw_networkx_nodes(tree, ax=ax, pos=pos,
                                   node_size=node_size,
                                   node_color="w",
                                   edgecolors="k",
                                   font_size=node_fontsize,
                                   *args, **kwds)
            nx.draw_networkx_labels(tree, ax=ax, pos=pos,
                                    labels=labels,
                                    font_size=node_fontsize)

            if edge_labels == "nodes":
                if tree_i == 0:
                    elabels = {(node1, node2): f"{node1}{node2}"
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
                    cop_name = copula.name.replace("fitted ", "")
                    if cop_name.startswith("seasonal"):
                        cop_name = "\n".join(cop_name.split())
                    tau = edge["tau"]
                    label = (f"{cop_name}\n") + (rf"$\tau={tau:.2f}$")
                    elabels[(node1, node2)] = label
            else:
                elabels = None
            nx.draw_networkx_edge_labels(tree, ax=ax, pos=pos,
                                         edge_labels=elabels,
                                         font_size=edge_fontsize,
                                         # alpha=.8,
                                         bbox=dict(alpha=.5,
                                                   color="w",
                                                   edgecolor="w"),
                                         *args, **kwds)
            nx.draw_networkx_edges(tree, ax=ax, pos=pos, style="--")

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
            node1, node2 = sorted((node1, node2))
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

    def plot_seasonal(self):
        if self.dtimes is None:
            print("We are not seasonal (hint:pass dtimes to init)")
        figs, axs = [], []
        for edge in self:
            cop_key, copula = [(key, value)
                               for key, value in edge.items()
                               if key.startswith("Copula_")][0]
            if not isinstance(copula, seasonal_cop.SeasonalCop):
                continue
            fig, axs_ = plt.subplots(nrows=3, ncols=2,
                                     # subplot_kw=dict(aspect="equal")
                                     )
            node1 = flat_set(edge["node_u"])
            node2 = flat_set(edge["node_v"])
            condition = node1 & node2
            if condition:
                condition = " ".join(self.varnames[i] for i in condition)
                condition = " (given %s)" % condition
            else:
                condition = ""
            var1 = node1 - node2
            var2 = node2 - node1
            varname1 = self.varnames[list(var1)[0]]
            varname2 = self.varnames[list(var2)[0]]
            plot_doys = np.linspace(0, 366, 5)[:-1]
            copula.plot_corr(fig=fig, ax=axs_[0, 0],
                             title="Correlation")
            for plot_doy in plot_doys:
                axs_[0, 0].axvline(plot_doy, color="gray")
            copula.plot_fourier_fit(fig=fig, ax=axs_[0, 1],
                                    title=r"Fourier approximation of $\theta$")
            for plot_doy in plot_doys:
                axs_[0, 1].axvline(plot_doy, color="gray")
            copula.plot_seasonal_densities(fig=fig, axs=axs_[1:],
                                           plot_doys=plot_doys)
            title = ("tree:{tree}, {varname1}-{varname2}{condition}, {copula}"
                     .format(tree=edge["tree"], varname1=varname1,
                             varname2=varname2, condition=condition,
                             copula=copula.name[len("seasonal "):]))
            fig.suptitle(title)
            fig.tight_layout(rect=(0, 0, 1, .95))
            figs += [fig]
            axs += [axs_]
        return figs, axs


class CVine(Vine):

    def __init__(self, *args, central_node=None, **kwds):
        self.central_node_name = central_node
        super().__init__(*args, **kwds)

    def _best_tree(self, full_graph, tree_i):
        nodes = list(full_graph.nodes())
        weights = np.array([[1
                             if node1 == node2 else
                             full_graph[node1][node2]["weight"]
                             for node1 in nodes]
                            for node2 in nodes])
        if self.central_node_name is None or tree_i > 0:
            central_node_i = np.argmin(np.sum(weights, axis=0))
        else:
            central_node_i = self.varnames.index(self.central_node_name)
        central_node = nodes[central_node_i]
        other_nodes = sorted(list(set(nodes) - set([central_node])))
        new_graph = nx.Graph()
        for other_node in other_nodes:
            edge_dict = full_graph[central_node][other_node]
            new_graph.add_edge(central_node, other_node,
                               **edge_dict)
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

        if cop_conf.PROFILE:
            Us = [csim_one((t, Ps[:, t], self, zero, one))
                  for t in range(T_sim)]
        else:
            with multiprocessing.Pool(n_nodes) as pool:
                Us = pool.map(csim_one,
                              zip(range(T_sim),
                                  Ps.T,
                                  repeat(self),
                                  repeat(zero),
                                  repeat(one)),
                              chunksize=int(T_sim / n_nodes))
        return np.array(Us).T

    def _quantiles(self, ranks, **tqdm_kwds):
        """Returns the 'quantiles' (in the sense that if they would be used as
        random numbers in `simulate`, the input data would be
        reproduced).

        """
        if self.verbose:
            print("Obtaining quantiles in CVine")
        T = ranks.shape[1]
        Us = ranks
        zero = 1e-6
        one = 1 - zero

        if cop_conf.PROFILE:
            Ps = [cquant_one((t, Us[:, t], self, zero, one))
                  for t in range(T)]
        else:
            with multiprocessing.Pool(n_nodes) as pool:
                Ps = pool.map(cquant_one,
                              zip(range(T),
                                  Us.T,
                                  repeat(self),
                                  repeat(zero),
                                  repeat(one)),
                              chunksize=(T // n_nodes))
        return np.array(Ps).T


class DVine(Vine):
    pass


class RVine(Vine):

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)

    def _best_tree(self, full_graph, tree_i):
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
            P[1] = self[0, 1]["C_1|0"](conditioned=U[None, 1],
                                       condition=P[None, 0], t=t)
            Q[1, 1] = P[1]
            if I[0, 1]:
                V[0, 1] = minmax(self[0, 1]["C_0|1"](conditioned=U[None, 0],
                                                     condition=U[None, 1],
                                                     t=t))
            for j in range(2, self.d):
                Q[0, j] = U[j]
                cop = self[0, j]["C_%d|%d" % (j, A[0, j])]
                Q[1, j] = cop(conditioned=U[None, j],
                              condition=U[None, A[0, j]], t=t)
                cop = self[0, j]["C_%d|%d" % (A[0, j], j)]
                V[0, j] = minmax(cop(conditioned=U[None, A[0, j]],
                                     condition=U[None, j], t=t))
                for l in range(1, j):
                    if A[l, j] == M[l, j]:
                        s = Q[None, l, A[l, j]]
                    else:
                        s = V[None, l - 1, M[l, j]]
                    Z[l, j] = s
                    cop = self[l, j]["C_%d|%d" % (j, A[l, j])]
                    Q[l + 1, j] = minmax(cop(conditioned=Q[None, l, j],
                                             condition=s, t=t))
                P[j] = Q[j, j]
                for l in range(1, j):
                    if I[l, j]:
                        cop = self[l, j]["C_%d|%d" % (A[l, j], j)]
                        V[l, j] = minmax(cop(conditioned=Z[None, l, j],
                                             condition=Q[None, l, j],
                                             t=t))
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

        if randomness is None:
            Ps = np.random.rand(self.d, T_sim)
        else:
            Ps = randomness

        Us = np.empty_like(Ps)
        Us[0] = Ps[0]
        Q, V, Z = np.empty((3,) + self.A.shape)

        if cop_conf.PROFILE:
            Us = [rsim_one((t, Ps[:, t], self, zero, Q, V, Z))
                  for t in range(T_sim)]

        with multiprocessing.Pool(n_nodes) as pool:
            Us = pool.map(rsim_one,
                          zip(range(T_sim), Ps.T, repeat(self),
                              repeat(zero), repeat(Q),
                              repeat(V), repeat(Z)),
                          chunksize=int(T_sim / n_nodes))
        return np.array(Us).T


def rsim_one(args):
    t, P, rvine, zero, Q, V, Z = args
    I, A, M = rvine.I, rvine.A, rvine.M
    Q[:] = V[:] = Z[:] = zero
    U = np.empty(len(P))
    U[0] = P[0]
    U[1] = rvine[0, 1]["C^_1|0"](conditioned=P[None, 1],
                                 condition=P[None, 0], t=t)
    Q[1, 1] = P[1]
    if I[0, 1]:
        V[0, 1] = rvine[0, 1]["C_0|1"](conditioned=U[None, 0],
                                       condition=U[None, 1],
                                       t=t)
    for j in range(2, rvine.d):
        Q[j, j] = P[j]
        for l in range(j - 1, 0, -1):
            if A[l, j] == M[l, j]:
                s = Q[None, l, A[l, j]]
            else:
                s = V[None, l - 1, M[l, j]]
            Z[l, j] = s
            cop = rvine[l, j][f"C^_{j}|{A[l, j]}"]
            Q[l, j] = cop(conditioned=Q[None, l + 1, j],
                          condition=s,
                          t=t)
        cop = rvine[0, j]["C^_%d|%d" % (j, A[0, j])]
        U[j] = Q[0, j] = cop(conditioned=Q[None, 1, j],
                             condition=U[None, A[0, j]],
                             t=t)
        cop = rvine[0, j]["C_%d|%d" % (A[0, j], j)]
        V[0, j] = cop(conditioned=U[None, A[0, j]],
                      condition=U[None, j],
                      t=t)
        for l in range(1, j):
            if I[l, j]:
                cop = rvine[l, j]["C_%d|%d" % (A[l, j], j)]
                V[l, j] = cop(conditioned=Z[None, l, j],
                              condition=Q[None, l, j],
                              t=t)
    return U


@my.cache("data_means", "data_std", "vine", "As", "phases", "cop_quantiles",
          "qq_std")
def vg_ph(vg_obj, sc_pars, refit=False):
    weights = "tau"
    if vg_ph.vine is None or refit:
        data_means = vg_obj.data_trans.mean(axis=1)[:, None]
        data_std = vg_obj.data_trans.std(axis=1)[:, None]
        ranks = dists.norm.cdf(vg_obj.data_trans)
        with tools.shelve_open(cop_conf.vine_cache) as sh:
            key = "_".join(("_".join(vg_obj.var_names),
                            vg_obj.met_file,
                            weights,
                            vg_obj.primary_var[0]))
            key = tools.gen_hash(key)
            try:
                if refit:
                    raise KeyError("Refitting vine")
                vine = sh[key]
            except KeyError:
                vine = CVine(ranks,
                             varnames=vg_obj.var_names,
                             dtimes=vg_obj.times,
                             weights=weights,
                             central_node=vg_obj.primary_var[0])
                sh[key] = vine
        vg_ph.data_means = data_means
        vg_ph.data_std = data_std
        vg_ph.vine = vine
        vg_ph.cop_quantiles = np.array(vg_ph.vine.quantiles())
        # attach SeasonalCop instance to vg so that does not get lost.
        vg_obj.vine = vg_ph.vine
    if vg_ph.phases is None:
        qq_std = dists.norm.ppf(vg_ph.cop_quantiles)
        qq_std[~np.isfinite(qq_std)] = np.nan
        vg_ph.qq_std = my.interp_nan(qq_std)
        vg_ph.As = np.fft.fft(vg_ph.qq_std)
        vg_ph.phases = np.angle(vg_ph.As)
    T = vg_obj.T
    # phase randomization with same random phases in all variables
    phases_lh = np.random.choice(vg_ph.phases[vg_obj.primary_var_ii[0]],
                                 T // 2 if T % 2 == 1 else T // 2 - 1)
    # phases_lh = np.random.uniform(-np.pi, np.pi,
    #                               T // 2 if T % 2 == 1 else T // 2 - 1)
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

    # adjust means
    fft_sim -= fft_sim.mean(axis=1)[:, None]
    # fft_sim += vg_ph.data_means

    # change in mean scenario
    prim_i = vg_obj.primary_var_ii
    fft_sim[prim_i] += sc_pars.m[prim_i]
    fft_sim[prim_i] += sc_pars.m_t[prim_i]
    qq = dists.norm.cdf(fft_sim)
    eps = 1e-12
    qq[qq > (1 - eps)] = 1 - eps
    qq[qq < eps] = eps
    # print(np.mean(qq, axis=1))
    ranks_sim = vg_ph.vine.simulate(randomness=qq)
    # print(np.mean(ranks_sim, axis=1))
    data_sim = dists.norm.ppf(ranks_sim)
    data_sim[~np.isfinite(data_sim)] = np.nan
    data_sim = my.interp_nan(data_sim)
    data_sim += vg_ph.data_means

    # from lhglib.contrib.time_series_analysis import time_series as ts
    # fig, axs = plt.subplots(nrows=vg_obj.K, ncols=1, sharex=True)
    # for i, ax in enumerate(axs):
    #     freqs = np.fft.fftfreq(T)
    #     ax.bar(freqs, phases[i], width=.5 / T, label="surrogate")
    #     ax.bar(freqs, vg_ph.phases[i], width=.5 / T, label="data")
    #     axs[0].legend(loc="best")

    # my.hist(vg_ph.phases.T, 20)

    # vg_ph.vine.plot(edge_labels="copulas")
    # # vg_ph.vine.plot_tplom()
    # vg_ph.vine.plot_qqplom()
    # fig, axs = ts.plot_cross_corr(vg_ph.qq_std)
    # fig, axs = ts.plot_cross_corr(fft_sim, linestyle="--", axs=axs, fig=fig)
    # fig, axs = plt.subplots(nrows=vg_obj.K, ncols=1, sharex=True)
    # for i, ax in enumerate(axs):
    #     ax.plot(vg_ph.qq_std[i])
    #     ax.plot(fft_sim[i], linestyle="--")
    #     plt.show()

    return data_sim


if __name__ == '__main__':
    # for deterministic networkx graphs!
    # np.random.seed(2)

    import vg
    import time
    from vg import vg_plotting, vg_base
    import config_konstanz_disag as vg_conf
    vg.conf = vg_plotting.conf = vg_base.conf = vg_conf
    met_vg = vg.VG(("theta", "ILWR", "rh", "R"), verbose=True,
                   dump_data=False)

    before = time.time()
    met_vg.simulate(sim_func=vg_ph, theta_incr=.0,
                    # sim_func_kwds=dict(refit=True)
                    )
    print(time.time() - before)

    ranks = np.array([stats.rel_ranks(values)
                      for values in met_vg.data_trans])
    cvine = CVine(ranks, varnames=met_vg.var_names,
                  dtimes=met_vg.times,
                  # weights="likelihood"
                  )
    # quantiles = cvine.quantiles()
    # assert np.all(np.isfinite(quantiles))
    before = time.time()
    for _ in range(2):
        sim = cvine.simulate()
    print(time.time() - before)

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


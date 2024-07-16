from itertools import combinations

import igraph as ig
import numpy as np


def is_dag(B):
    G = ig.Graph.Weighted_Adjacency(B.T.tolist())
    return G.is_dag()


def get_parents(B, i):
    return np.where(B[i, :] != 0)[0]


def get_vstructs(B):
    vstructs = set()
    nodes = list(range(len(B)))
    for node in nodes:
        parents = get_parents(B, node)
        for p1, p2 in combinations(parents, 2):
            if p1 not in get_parents(B, p2) and p2 not in get_parents(B, p1):
                vstructs.add((p1, node, p2))
    return vstructs


def get_moral_graph(B):
    B_support = (B != 0).astype(int)
    vstructs = get_vstructs(B)
    undirected_graph = ((B_support + B_support.T) != 0).astype(int)
    moral_graph = undirected_graph
    for i, j, k in vstructs:
        # i -> j <- k
        moral_graph[i, k] = 1
        moral_graph[k, i] = 1
    return moral_graph


def construct_local_pdag(num_vars, target, neighbors, vstructs):
    # Construct pdag
    pdag = np.zeros((num_vars, num_vars))
    for neighbor in neighbors:
        # Undirected edge
        pdag[target, neighbor] = 1    # neighbor -> target
        pdag[neighbor, target] = 1    # target -> neighbor
    for i, j, k in vstructs:
        # Remove i -- j
        pdag[i, j] = 0
        pdag[j, i] = 0
        # Remove j -- k
        pdag[k, j] = 0
        pdag[j, k] = 0
        # Add i -> j <- k
        pdag[j, i] = 1    # i -> j
        pdag[j, k] = 1    # k -> j

    return pdag


def get_local_dcg(dcg, target, include_spouses=False):
    # Return structure_local of the same dimension as structure
    # structure_local contains only directed/undirected edges
    # connecting to target node, as well as incoming edges
    # of each child
    num_vars = len(dcg)
    dcg_local = np.zeros((num_vars, num_vars))
    dcg_local[:, target] = dcg[:, target]
    dcg_local[target, :] = dcg[target, :]
    if include_spouses:
        # Include incoming edges for children
        children = np.where(dcg[:, target])[0]
        if len(children) > 0:
            for child in children:
                parents_of_child = np.where(dcg[child, :])[0]
                dcg_local[child, parents_of_child] = dcg[child, parents_of_child]
    return dcg_local


def get_local_pdag_from_dag(dag, target):
    num_vars = len(dag)
    # Get undirected edges connecting to T
    neighbors = set(np.where(dag[target, :])[0]) | set(np.where(dag[:, target])[0])
    # get_vstructs(dag_est) may return many v-structures, but we only
    # trust those v-structures with the form target -> child <- spouse
    vstructs = [(i, j, k) for (i, j, k) in get_vstructs(dag)
                if i == target or k == target]
    pdag = construct_local_pdag(num_vars, target, neighbors, vstructs)
    return pdag

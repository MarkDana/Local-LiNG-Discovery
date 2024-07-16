from itertools import combinations

from causallearn.search.FCMBased.lingam.hsic import hsic_test_gamma
import networkx as nx
import numpy as np

from local_isa_ling.acyclic_sink_processing import acyclic_sink_processing
from local_isa_ling.block_diagonal_processing import block_diagonal_processing
from local_isa_ling.isa import sparseica_W_adasize_Alasso_mask_regu


def two_step_ica(X, ica_lambda, ica_regu):
    num_vars = X.shape[1]
    mask = np.ones((num_vars, num_vars)) - np.eye(num_vars)
    _, W, _, _, _, _, _, _ = sparseica_W_adasize_Alasso_mask_regu(ica_lambda, mask, X.T, ica_regu)
    return W


def local_isa_ling(X, target, threshold=0.05, postprocess_type='acyclic_sink',
                   alpha=0.05, rank_tol=0.05):
    num_samples, num_vars = X.shape
    # Usually lamda is set to a constant times log(num_samples)
    ica_lambda = np.log(num_samples) * 4
    ica_regu = 0.05
    W = two_step_ica(X, ica_lambda, ica_regu)
    if postprocess_type == 'acyclic_sink':
        dag = acyclic_sink_processing(W, target, threshold)
        params = {'dag': dag}
        return params
    elif postprocess_type == 'block_diagonal':
        S = (W @ X.T).T
        subspaces, num_ind_test = get_subspaces_with_heuristics(W, S, target, threshold, alpha)
        subspaces = [list(subspace) for subspace in subspaces]
        all_possible_results, most_stable_idx \
            = block_diagonal_processing(W, subspaces, threshold, rank_tol)
        dcgs = []
        for possible_result in all_possible_results:
            dcg = np.zeros((num_vars, num_vars))
            for (parent, child), weight in possible_result.items():
                dcg[child, parent] = weight
            dcgs.append(dcg)
        params = {'dcgs': dcgs,
                  'most_stable_idx': most_stable_idx,
                  'dcg': dcgs[most_stable_idx]}
        return params
    else:
        raise ValueError('Unknown post-processing type')


def get_subspaces(S, alpha):
    num_vars = S.shape[1]
    num_ind_test = 0
    adj = np.zeros((num_vars, num_vars))
    for i, j in combinations(range(num_vars), 2):
        # Try to reduce some HSIC test, by finding the connected
        # components in each iteration
        G = nx.from_numpy_array(adj)
        connected_components = list(nx.connected_components(G))
        if is_same_subspace(connected_components, i, j):
            continue
        S_i = np.expand_dims(S[:, i], axis=1)
        S_j = np.expand_dims(S[:, j], axis=1)
        _, pval = hsic_test_gamma(S_i, S_j)
        num_ind_test += 1
        if pval <= alpha:
            adj[i, j] = 1
            adj[j, i] = 1
    G = nx.from_numpy_array(adj)
    connected_components = list(nx.connected_components(G))
    return connected_components, num_ind_test


def get_subspaces_with_heuristics(W, S, target, threshold, alpha):
    # Use some heuristic to make it more efficient and reliable
    T_index_in_M = target
    rows_with_nonzero_entries_on_T_column \
        = np.where(~np.isclose(W[:, T_index_in_M], 0, atol=threshold))[0]
    singletons = [i for i in rows_with_nonzero_entries_on_T_column]
    others = [i for i in range(len(W)) if i not in singletons]
    map_dict = {idx: idx_mapped for idx, idx_mapped in enumerate(others)}
    if len(others) > 1:
        subspaces_others, num_ind_test = get_subspaces(S[:, others], alpha)
        subspaces_others_mapped = []
        for subspace in subspaces_others:
            subspace_mapped = [map_dict[i] for i in subspace]
            subspaces_others_mapped.append(subspace_mapped)
        subspaces = [[i] for i in singletons] + subspaces_others_mapped
    else:
        subspaces = singletons + others
        subspaces = [[i] for i in subspaces]
        num_ind_test = 0
    return subspaces, num_ind_test


def is_same_subspace(subspaces, source_idx_1, source_idx_2):
    for subspace in subspaces:
        if source_idx_1 in subspace and source_idx_2 in subspace:
            return True
    return False
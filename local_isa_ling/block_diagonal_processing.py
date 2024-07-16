from itertools import permutations

import numpy as np


def block_diagonal_processing(W, subspaces, threshold=0.05, rank_tol=0.05):
    """
    :param W: an |M| x |M| demixing matrix, i.e., BMM_inv with permutation and general scalings.
              when M is an MB of T, the first column of W correspond to the central node T (i.e., T==M[0]).
    :param subspaces: list of lists, the connected components (independent subspaces) of W @ data (obtained by pairwise HSIC tests),
                                   e.g., when |M|=4, subspaces = [[0, 2], [1], [3]], meaning the
                                   first and third rows of W demix data and produce one subspace,
                                   while the second and fourth rows of W demix data and produce independent components (singletons).
    :param threshold: threshold for coefficient to be considered as zero (generalizable to matrix rank (singular values threshold)).
    :return: all_possible_results: a list of dictionaries; each dictionary is a possible result in form of `poped_out_edges` as in `inverse_direct_lingam`.

    """
    T_index_in_M = 0
    row_id_to_subspace_id = {row_id: subspace_id for subspace_id, subspace in enumerate(subspaces) for row_id in subspace}
    rows_with_nonzero_entries_on_T_column = np.where(~np.isclose(W[:, T_index_in_M], 0, atol=threshold))[0]
    for row_id in rows_with_nonzero_entries_on_T_column:
        # i.e., if T->Xi, Xi row must be demixed to singleton
        assert len(subspaces[row_id_to_subspace_id[row_id]]) == 1, \
            "Xi row must be demixed to singleton"

    # now we want to know which of the columns (variables) correspond to these several rows
    # note! we need to consider all permutations of range(M), not just those several subspaces (rows groups)
    # because two variables in a same group may not have contigent variable names (i.e., column names)
    good_permutations_regarding_rows_nonzero_on_T = set()
    for perm in permutations(range(len(W))):    # this step is ugly; we need some pruning to reduce the search space (permutations with constraints)
        perm = list(perm)
        original_rid_to_new_rid = {p: r for r, p in enumerate(perm)}
        permed_W = W[perm]
        # if np.any(np.diagonal(permed_W) == 0):
        #     continue
        permed_connected_components = [[original_rid_to_new_rid[rid] for rid in cc] for cc in subspaces]
        is_a_good_diagonal_perm = True
        for cc in permed_connected_components:
            if np.linalg.matrix_rank(permed_W[cc][:, cc], tol=rank_tol) != len(cc):
                is_a_good_diagonal_perm = False
                break
        if is_a_good_diagonal_perm:
            nonzero_rows_to_true_colids = tuple(original_rid_to_new_rid[rid] for rid in rows_with_nonzero_entries_on_T_column)
            good_permutations_regarding_rows_nonzero_on_T.add((nonzero_rows_to_true_colids, tuple(perm)))

    # if the graph is acyclic, at this step `good_permutations_regarding_rows_nonzero_on_T` should contain one and only one element,
    #   i.e., these several rows (T and T's children) are correctly identified and in their correct locations.
    # but if the graph is cyclic, good_permutations_regarding_rows_nonzero_on_T may not be unique,
    #   but all the them will be correct (possible), i.e., consistent with the local part of one of the correct DAG in the equivalence class.
    # in any case, we return all the possibilities.
    all_possible_results = []
    min_stability_val = float('inf')
    most_stable_idx = None
    for i, (nonzero_rows_to_true_colids, perm) in enumerate(good_permutations_regarding_rows_nonzero_on_T):
        poped_out_edges = dict()
        stability_val = compute_cycle_strength(W, perm)
        if stability_val < min_stability_val:
            min_stability_val = stability_val
            most_stable_idx = i
        for row_id_in_any_W, true_col_id in zip(rows_with_nonzero_entries_on_T_column, nonzero_rows_to_true_colids):
            estimated_incoming_weights = np.copy(W[row_id_in_any_W])
            estimated_incoming_weights /= estimated_incoming_weights[true_col_id]
            estimated_incoming_weights[true_col_id] = 0
            estimated_incoming_weights *= -1
            for parent, weight in enumerate(estimated_incoming_weights):
                if np.abs(weight) > threshold:
                    poped_out_edges[(parent, true_col_id)] = weight
        all_possible_results.append(poped_out_edges)

    return all_possible_results, most_stable_idx


def compute_cycle_strength(W, perm):
    perm = list(perm)
    W_prime = W[perm]
    W_prime = W_prime / np.diagonal(W_prime)[:, np.newaxis]
    B_prime = np.eye(len(W_prime)) - W_prime
    loop_strength = 0
    B_prod = B_prime
    for _ in range(len(W) - 1):
        B_prod = B_prod @ B_prime
        loop_strength += np.sum(np.abs(np.diagonal(B_prod)))
    return loop_strength
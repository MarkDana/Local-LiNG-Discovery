import numpy as np


def acyclic_sink_processing(W, target, threshold):
    W = W.copy()
    num_vars = len(W)
    W[np.abs(W) < threshold] = 0
    # assert is_singleton(subspaces, target)    # Must be a singleton
    dag = np.zeros((num_vars, num_vars))
    W_iter = W.copy()
    rows_mapping = list(range(num_vars))
    columns_mapping = list(range(num_vars))
    while len(W_iter) > 0:
        columns = np.where((W_iter != 0).sum(axis=0) == 1)[0]
        # assert len(columns) > 0
        # In case there is not any sink node
        # This indicates that the matrix W is not "acyclic"
        if len(columns) == 0:
            columns, W_iter, W = get_column_with_one_nonzero(W_iter, W,
                                                             rows_mapping, columns_mapping)
        rows = np.where(W_iter[:, columns].T)[1]
        columns_mapped = [columns_mapping[i] for i in columns]
        rows_mapped = [rows_mapping[i] for i in rows]
        for row_mapped, column_mapped in zip(rows_mapped, columns_mapped):
            if row_mapped == target:
                coef = compute_coefficients(W, row_mapped, column_mapped)
                dag[column_mapped] = coef
                break
            if W[row_mapped, target] != 0:
                coef = compute_coefficients(W, row_mapped, column_mapped)
                dag[column_mapped] = coef
        W_iter = np.delete(W_iter, columns, axis=1)
        W_iter = np.delete(W_iter, rows, axis=0)
        rows_mapping = [i for i in rows_mapping if i not in rows_mapped]
        columns_mapping = [i for i in columns_mapping if i not in columns_mapped]
    return dag


def get_column_with_one_nonzero(W_iter, W, rows_mapping, columns_mapping):
    nonzeros_counts = (W_iter != 0).sum(axis=0)
    min_nonzeros = nonzeros_counts.min()
    if (nonzeros_counts == min_nonzeros).sum() == 1:
        # 1st criterion: Find column with smallest nonzero count
        column = nonzeros_counts.argmin()
    else:
        # 2nd : Find column with smallest sum of absolute weights
        column = np.abs(W_iter).sum(axis=0).argmin()
    rows = np.where(W_iter[:, column])[0]
    for row in rows:
        if row != column:
            row_mapped = rows_mapping[row]
            column_mapped = columns_mapping[column]
            W[row_mapped, column_mapped] = 0
    W_iter[:, column] = 0
    W_iter[column, column] = 1
    return np.array([column]), W_iter, W


def is_singleton(subspaces, source_idx):
    for subspace in subspaces:
        if source_idx in subspace:
            if len(subspace) == 1:
                return True
            break
    return False


def compute_coefficients(W, row_idx, column_one_idx):
    coef = np.copy(W[row_idx])
    coef /= coef[column_one_idx]
    coef[column_one_idx] = 0
    coef = -coef
    return coef
import causaldag as cd
import numpy as np

from utils.graph import is_dag, get_local_pdag_from_dag, get_local_dcg


def count_accuracy_of_mb(mb_true, mb_est):
    """Compute various accuracy metrics for estimated Markov blanket."""
    true_pos = mb_true.intersection(mb_est)
    false_pos = mb_est - mb_true
    false_neg = mb_true - true_pos
    precision, recall, f1 = count_precision_recall_f1(tp=len(true_pos),
                                                      fp=len(false_pos),
                                                      fn=len(false_neg))
    return {'mb_precision': precision,
            'mb_recall': recall,
            'mb_f1': f1,
            'mb_true_size': len(mb_true),
            'mb_est_size': len(mb_est)}


def count_accuracy(dcg_true, dcg_est, target):
    """Compute various accuracy metrics for estimated local structure."""
    # assert is_dag(dag_true)
    # assert is_dag(dag_est)
    results = {}
    if is_dag(dcg_true) and is_dag(dcg_est):
        # Calculate performance metrics for PDAG
        results_pdag = compute_pdag_accuracy_from_dag(dcg_true, dcg_est, target)
        results.update(results_pdag)
    # Calculate performance metrics for local directed graphs
    results_dcg = count_dcg_accuracy(dcg_true, dcg_est, target)
    results.update(results_dcg)
    return results


def compute_pdag_accuracy_from_dag(dag_true, dag_est, target):
    assert is_dag(dag_true)
    assert is_dag(dag_est)
    pdag_true = get_local_pdag_from_dag(dag_true, target)
    pdag_est = get_local_pdag_from_dag(dag_est, target)
    pdag_true = (pdag_true != 0).astype(int)
    # causaldag package assumes that each column refers to parents of each variable
    # so we need to transpose here
    pdag_est = (pdag_est != 0).astype(int)
    pdag_shd = cd.PDAG.from_amat(pdag_true.T).shd(cd.PDAG.from_amat(pdag_est.T))
    return {'pdag_shd': pdag_shd,
            'pdag_nnz_true': pdag_true.sum(),
            'pdag_nnz_est': pdag_est.sum()}


def count_precision_recall_f1(tp, fp, fn):
    # Precision
    if tp + fp == 0:
        precision = None
    else:
        precision = float(tp) / (tp + fp)

    # Recall
    if tp + fn == 0:
        recall = None
    else:
        recall = float(tp) / (tp + fn)

    # F1 score
    if precision is None or recall is None:
        f1 = None
    elif precision == 0 or recall == 0:
        f1 = 0.0
    else:
        f1 = float(2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def count_dcg_accuracy(B_bin_true, B_bin_est, target):
    """Code modified from https://github.com/xunzheng/notears/blob/master/notears/utils.py"""
    B_bin_true = get_local_dcg(B_bin_true, target, include_spouses=True)
    B_bin_est = get_local_dcg(B_bin_est, target, include_spouses=True)
    B_bin_true = (B_bin_true != 0).astype(int)
    B_bin_est = (B_bin_est != 0).astype(int)
    d = B_bin_true.shape[0]
    # linear index of nonzeros
    pred = np.flatnonzero(B_bin_est)
    cond = np.flatnonzero(B_bin_true)
    cond_reversed = np.flatnonzero(B_bin_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    if pred_size == 0:
        fdr = None
    else:
        fdr = float(len(reverse) + len(false_pos)) / pred_size
    if len(cond) == 0:
        tpr = None
    else:
        tpr = float(len(true_pos)) / len(cond)
    if cond_neg_size == 0:
        fpr = None
    else:
        fpr = float(len(reverse) + len(false_pos)) / cond_neg_size
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_bin_est + B_bin_est.T))
    cond_lower = np.flatnonzero(np.tril(B_bin_true + B_bin_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    # false neg
    false_neg = np.setdiff1d(cond, true_pos, assume_unique=True)
    precision, recall, f1 = count_precision_recall_f1(tp=len(true_pos),
                                                      fp=len(reverse) + len(false_pos),
                                                      fn=len(false_neg))
    results = {'dcg_fdr': fdr,
               'dcg_tpr': tpr,
               'dcg_fpr': fpr,
               'dcg_shd': shd, 
               'dcg_precision': precision,
               'dcg_recall': recall,
               'dcg_f1': f1,
               'dcg_nnz_est': pred_size,
               'dcg_nnz_true': len(cond)}
    return results
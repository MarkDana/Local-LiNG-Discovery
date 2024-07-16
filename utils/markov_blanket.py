import numpy as np
from sklearn.linear_model import Lasso

from utils.graph import get_moral_graph


def estimate_markov_blanket(X, target):
    mb_lambda = 0.001
    mb_threshold = 0.01
    reg_coef = lasso(X, target, mb_lambda)
    mb = set(np.where(np.abs(reg_coef) > mb_threshold)[0])
    return mb

def get_true_markov_blanket(B, target):
    moral_graph = get_moral_graph(B)
    mb = set(np.where(moral_graph[target])[0])
    return mb


def lasso(X, target, l1_lambda=0.1):
    num_nodes = X.shape[1]
    neighbors = np.array([i for i in range(num_nodes) if i != target])
    y = X[:, target]
    x = X[:, neighbors]
    reg = Lasso(alpha=l1_lambda)
    reg.fit(x, y)
    coef = reg.coef_
    coef = np.insert(coef, target, 0)
    return coef
import random

import numpy as np


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def map_local_structure(structure_local, local_indices, num_vars):
    structure_mapped = np.zeros((num_vars, num_vars))
    structure_mapped[np.ix_(local_indices, local_indices)] = structure_local
    return structure_mapped

"""Code modified from https://github.com/syanga/dglearn/blob/master/dglearn/utils/sample_parameters.py"""
import numpy as np


def sample_data(B, noise_scales, num_samples):
    """
        Generate data given B matrix, variances
    """
    num_vars = len(noise_scales)
    noise = [np.random.uniform(-noise_scale, noise_scale, size=(num_samples))
             for noise_scale in noise_scales]
    noise = np.array(noise).T
    noise = noise**5
    return (np.linalg.inv(np.eye(num_vars) - B.T) @ noise.T).T


def sample_param_unif(B_support, B_low=0.5, B_high=0.9, noise_low=0.75, noise_high=1.25,
                      flip_sign=True, max_eig=1.0, max_cond_number=float('inf')):
    """
        Generate graph parameters given support matrix
        by sampling uniformly on a range

        returns B matrix with specified support and log variances.
        Accept-reject sampling: ensure eigenvalues of B all have norm < 1
        Note that for DAGs, eigenvalues are all zero.
    """
    assert B_support.shape[0] == B_support.shape[1]
    num_vars = B_support.shape[0]

    stable = False
    while not stable:
        B_sampled = B_support * np.random.uniform(B_low, B_high, size=B_support.shape)
        stable = np.max(np.absolute(np.linalg.eig(B_sampled)[0])) < max_eig
        if np.linalg.cond(np.eye(num_vars) - B_sampled) > max_cond_number: 
            stable = False

    noise_scales = np.random.uniform(noise_low, noise_high, size=num_vars)
    if flip_sign: 
        B_sampled *= (np.random.binomial(1, 0.5, size=B_sampled.shape)*2 - 1)

    return B_sampled, noise_scales
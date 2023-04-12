""" Evaluation metrics """

import numpy as np

def rdm(u1, u2):
    """Relative difference measure. Normalized l2 difference between u1 and u2
    indicating topography changes.

    u1 and u2 are leadfields of shape (n_sensors, n_sources[, n_orientations]).

    u1 : ndarray
    u2 : ndarray

    """
    assert u1.ndim == u2.ndim
    return np.linalg.norm(u1/np.linalg.norm(u1, axis=0, keepdims=True) - \
                          u2/np.linalg.norm(u2, axis=0, keepdims=True),
                          axis=0)

def lnmag(u1, u2):
    """Logarithmic magnitude difference measure. Ratio between overall
    magnitude of u1 and u2.
    """
    assert u1.ndim == u2.ndim
    return np.log(np.linalg.norm(u1, axis=0)/np.linalg.norm(u2, axis=0))
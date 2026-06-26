"""
Step 2 of Laplacian Eigenmaps (Belkin & Niyogi, 2003)

Given the adjacency graph, assign weights to edges
and compute the Graph Laplacian L = D - W.

Two weighting schemes:
  - Heat kernel : Wij = exp(-||xi - xj||^2 / t)
  - Binary      : Wij = 1 if connected, 0 otherwise
"""

import numpy as np


def heat_kernel_weights(X, adjacency, t):
    """
    Compute heat kernel weights.

    Wij = exp(-||xi - xj||^2 / t)  if i,j connected
    Wij = 0                          otherwise

    Parameters
    ----------
    X          : ndarray (n, d) — data points
    adjacency  : ndarray (n, n) — binary adjacency from graph.py
    t          : float — bandwidth parameter

    Returns
    -------
    W : ndarray (n, n) — symmetric weight matrix
    """
    n = X.shape[0]
    W = np.zeros((n, n))

    # Only compute weights for connected pairs
    rows, cols = np.where(adjacency > 0)

    for i, j in zip(rows, cols):
        diff = X[i] - X[j]
        squared_dist = np.dot(diff, diff)
        W[i, j] = np.exp(-squared_dist / t)

    return W


def binary_weights(adjacency):
    """
    Simple binary weight matrix.

    Wij = 1 if connected, 0 otherwise.
    Special case of heat kernel as t -> infinity.

    Parameters
    ----------
    adjacency : ndarray (n, n)

    Returns
    -------
    W : ndarray (n, n)
    """
    return adjacency.astype(float)


def compute_laplacian(W):
    """
    Compute Graph Laplacian L = D - W and degree matrix D.

    Parameters
    ----------
    W : ndarray (n, n) — weight matrix

    Returns
    -------
    L : ndarray (n, n) — Graph Laplacian
    D : ndarray (n, n) — Diagonal degree matrix
    """
    
    d = W.sum(axis=1)
    D = np.diag(d)

    # Graph Laplacian
    L = D - W

    return L, D


"""
Step 3 of Laplacian Eigenmaps

Solve the generalized eigenvalue problem:
    L f = lambda D f

The eigenvectors f1, f2, ..., fm corresponding to the
m smallest non-zero eigenvalues give the embedding.
"""

import numpy as np
from scipy.linalg import eigh


def compute_embedding(L, D, n_components=2):
    """
    Solve Lf = lambda Df and return embedding coordinates.

    Parameters
    ----------
    L           : ndarray (n, n) — Graph Laplacian
    D           : ndarray (n, n) — Degree matrix
    n_components: int — target embedding dimension

    Returns
    -------
    embedding   : ndarray (n, n_components)
        Each row is the embedding of one data point.
        embedding[i] = (f1(i), f2(i), ..., fm(i))

    eigenvalues : ndarray (n_components,)
        The n_components smallest non-zero eigenvalues.
    """
    # eigh solves generalized problem A v = lambda B v
    eigenvalues, eigenvectors = eigh(L, D)

    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Discard f0 (lambda0 = 0, constant vector)
    # Take f1, f2, ..., fm
    embedding = eigenvectors[:, 1:n_components + 1]
    eigenvalues = eigenvalues[1:n_components + 1]

    return embedding, eigenvalues


def eigenvalue_gap(L, D, k_max=10):
    """
    Compute first k_max eigenvalues to find cluster structure.

    A large gap after lambda_k means k natural clusters exist:
        lambda_1 ~ ... ~ lambda_k  <<  lambda_{k+1}

    Parameters
    ----------
    L     : ndarray (n, n)
    D     : ndarray (n, n)
    k_max : int

    Returns
    -------
    eigenvalues : ndarray (k_max,)
    """
    eigenvalues, _ = eigh(L, D)
    eigenvalues = np.sort(eigenvalues)
    return eigenvalues[:k_max]
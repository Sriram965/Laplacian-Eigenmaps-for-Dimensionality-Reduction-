"""
Step 1 of Laplacian Eigenmaps (Belkin & Niyogi, 2003)

Building a neighborhood graph from raw data points.
Two methods:
  - k-Nearest Neighbors (kNN)
  - epsilon-neighborhood
"""

import numpy as np
from scipy.spatial import KDTree


def knn_graph(X, k):
    """
    It Builds a  k-Nearest Neighbor adjacency matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    k : int — number of nearest neighbors

    Returns
    -------
    adjacency : ndarray of shape (n, n)
        Binary symmetric matrix.
        adjacency[i,j] = 1 if xj is among k nearest neighbors of xi
    """
    n = X.shape[0]
    adjacency = np.zeros((n, n))

    # KDTree for fast nearest neighbor search — O(n log n)
    tree = KDTree(X)

    # k+1 because query returns the point itself as first result
    distances, indices = tree.query(X, k=k + 1)

    for i in range(n):
        for j in indices[i][1:]:   
            adjacency[i, j] = 1
            adjacency[j, i] = 1 

    return adjacency


def epsilon_graph(X, epsilon):
    """
    It Builds an epsilon-neighborhood adjacency matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    epsilon : float — distance threshold

    Returns
    -------
    adjacency : ndarray of shape (n, n)
    """
    n = X.shape[0]
    adjacency = np.zeros((n, n))

    tree = KDTree(X)
    neighbors = tree.query_ball_point(X, r=epsilon)

    for i in range(n):
        for j in neighbors[i]:
            if i != j:
                adjacency[i, j] = 1
                adjacency[j, i] = 1

    return adjacency


def check_connectivity(adjacency):
    """
    Check if the graph is connected.

    If disconnected — increase k or epsilon.
    A disconnected graph means L has multiple zero eigenvalues
    which breaks the embedding.

    Returns
    -------
    is_connected : bool
    n_components : int
    """
    n = adjacency.shape[0]
    visited = np.zeros(n, dtype=bool)
    n_components = 0

    def dfs(node):
        stack = [node]
        while stack:
            v = stack.pop()
            if not visited[v]:
                visited[v] = True
                neighbors = np.where(adjacency[v] > 0)[0]
                stack.extend(neighbors)

    for i in range(n):
        if not visited[i]:
            dfs(i)
            n_components += 1

    return n_components == 1, n_components
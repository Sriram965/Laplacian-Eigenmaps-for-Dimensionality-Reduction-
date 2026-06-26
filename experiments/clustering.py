"""
Experiment 3 — Spectral Clustering

Shows that the same Laplacian machinery used for
dimensionality reduction also solves clustering.

Dataset: Two rings (non-convex shapes)
k-means fails — it assumes spherical clusters
Spectral clustering succeeds — uses graph structure
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph import knn_graph
from core.laplacian import heat_kernel_weights, compute_laplacian, binary_weights
from core.embedding import compute_embedding, eigenvalue_gap


def spectral_clustering(X, k=2, n_neighbors=10, t=1.0):
    """
    Spectral clustering using our Laplacian Eigenmaps core.

    Steps:
    1. Build kNN graph
    2. Compute L = D - W
    3. Solve Lf = lambda Df
    4. Run k-means on eigenvectors

    Parameters
    ----------
    X           : ndarray (n, d)
    k           : int — number of clusters
    n_neighbors : int — kNN parameter
    t           : float — heat kernel bandwidth

    Returns
    -------
    labels    : ndarray (n,) — cluster assignments
    embedding : ndarray (n, k) — eigenvector embedding
    """
    # Steps 1-3 — exactly same as Laplacian Eigenmaps
    adj = knn_graph(X, k=n_neighbors)
    W = binary_weights(adj)
    L, D = compute_laplacian(W)
    embedding, eigenvalues = compute_embedding(L, D, n_components=k)

    # Step 4 — k-means on eigenvectors (only difference from LE)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embedding)

    return labels, embedding, eigenvalues


def run():

    print("=" * 50)
    print("Experiment 3: Spectral Clustering")
    print("=" * 50)

    # 1. Generate two rings dataset
    print("\n[1/4] Generating two rings dataset...")
    X, y_true = make_circles(n_samples=300, noise=0.05,
                             factor=0.4, random_state=42)
    print(f"      Shape: {X.shape}")
    print(f"      Classes: {np.unique(y_true)}")

    # 2. Spectral Clustering
    print("\n[2/4] Running spectral clustering...")
    labels_spectral, embedding, eigenvalues = spectral_clustering(
        X, k=2, n_neighbors=20, t=1.0
    )
    score_spectral = adjusted_rand_score(y_true, labels_spectral)
    print(f"      Eigenvalues: {eigenvalues.round(4)}")
    print(f"      Adjusted Rand Score: {score_spectral:.4f}")

    # 3. k-means for comparison
    print("\n[3/4] Running k-means for comparison...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(X)
    score_kmeans = adjusted_rand_score(y_true, labels_kmeans)
    print(f"      Adjusted Rand Score: {score_kmeans:.4f}")

    # 4. Eigenvalue gap
    print("\n[4/4] Computing eigenvalue spectrum...")
    adj = knn_graph(X, k=20)
    W   = binary_weights(adj)
    L, D = compute_laplacian(W)
    evals = eigenvalue_gap(L, D, k_max=6)
    print(f"      Eigenvalues: {evals.round(4)}")
    print(f"      Gap after lambda2 reveals 2 clusters")

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # Ground truth
    axes[0].scatter(X[:,0], X[:,1], c=y_true,
                   cmap='Set1', s=15, alpha=0.8)
    axes[0].set_title('Ground Truth')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')

    # Spectral clustering
    axes[1].scatter(X[:,0], X[:,1], c=labels_spectral,
                   cmap='Set1', s=15, alpha=0.8)
    axes[1].set_title(f'Spectral Clustering\nARI: {score_spectral:.3f}')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')

    # k-means
    axes[2].scatter(X[:,0], X[:,1], c=labels_kmeans,
                   cmap='Set1', s=15, alpha=0.8)
    axes[2].set_title(f'k-means\nARI: {score_kmeans:.3f}')
    axes[2].set_xlabel('x'); axes[2].set_ylabel('y')

    # Eigenvalue spectrum
    axes[3].plot(range(len(evals)), evals, 'o-',
                color='steelblue', markersize=8)
    axes[3].set_title('Eigenvalue Spectrum\n(gap after λ₂ → 2 clusters)')
    axes[3].set_xlabel('Index')
    axes[3].set_ylabel('Eigenvalue')
    axes[3].grid(True, alpha=0.3)

    plt.suptitle('Spectral Clustering vs k-means on Two Rings',
                fontsize=13)
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    fig.savefig("results/clustering.png", dpi=150, bbox_inches='tight')
    print("\n      Saved: results/clustering.png")
    plt.show()

    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"  Spectral Clustering ARI : {score_spectral:.4f}")
    print(f"  k-means ARI             : {score_kmeans:.4f}")

    return labels_spectral, labels_kmeans, y_true


if __name__ == "__main__":
    run()
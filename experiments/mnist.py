"""
Experiment 2 — MNIST Dimensionality Reduction

Reduce MNIST digits from 784D to 2D using Laplacian Eigenmaps.
Same digit classes should cluster together in the embedding.

Evaluation:
  - Visual: color-coded 2D plot
  - KNN accuracy: are neighbors preserved?
  - Trustworthiness: formal locality measure (0 to 1)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph import knn_graph, check_connectivity
from core.laplacian import heat_kernel_weights, compute_laplacian
from core.embedding import compute_embedding


def trustworthiness(X_high, X_low, k=10):
    """
    Trustworthiness measures how well local structure
    is preserved from high-dimensional to low-dimensional.

    Score = 1.0 means perfect preservation.
    Score < 0.9 means significant structure loss.

    Parameters
    ----------
    X_high : ndarray (n, d) — original high-dim data
    X_low  : ndarray (n, m) — embedding
    k      : int — number of neighbors to check

    Returns
    -------
    score : float in [0, 1]
    """
    n = X_high.shape[0]

    # Rank neighbors in high-dim space
    from scipy.spatial.distance import cdist
    dist_high = cdist(X_high, X_high)
    dist_low  = cdist(X_low,  X_low)

    rank_high = np.argsort(np.argsort(dist_high, axis=1), axis=1)
    rank_low  = np.argsort(dist_low, axis=1)

    score = 0.0
    for i in range(n):
        # k nearest neighbors in low-dim space
        neighbors_low = rank_low[i, 1:k+1]
        for j in neighbors_low:
            # penalize if they were far away in high-dim space
            r = rank_high[i, j]
            if r > k:
                score += (r - k)

    norm = 2.0 / (n * k * (2 * n - 3 * k - 1))
    return 1.0 - norm * score


def run(n_samples=2000, k=10, t=10.0, n_components=2):

    print("=" * 50)
    print("Experiment 2: MNIST Dimensionality Reduction")
    print("=" * 50)


    # 1. Load MNIST
    print("\n[1/5] Loading MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X_full = mnist.data.astype(float)
    y_full = mnist.target.astype(int)

    # Use subset for speed
    np.random.seed(42)
    idx = np.random.choice(len(X_full), n_samples, replace=False)
    X = X_full[idx]
    y = y_full[idx]

    X = X / 255.0

    print(f"      Using {n_samples} samples")
    print(f"      Shape: {X.shape}  ({X.shape[1]}D per image)")
    print(f"      Classes: {np.unique(y)}")


    # 2. Build Graph
    print(f"\n[2/5] Building {k}-NN graph...")
    adj = knn_graph(X, k=k)
    connected, n_comp = check_connectivity(adj)
    print(f"      Connected: {connected}, Components: {n_comp}")

    # 3. Laplacian + Embedding
    print(f"\n[3/5] Computing Laplacian and embedding...")
    W = heat_kernel_weights(X, adj, t=t)
    L, D = compute_laplacian(W)
    embedding_le, eigenvalues = compute_embedding(L, D, n_components)
    print(f"      Eigenvalues: {eigenvalues.round(4)}")
    print(f"      Embedding shape: {embedding_le.shape}")

    # 4. Evaluation
    print(f"\n[4/5] Evaluating...")

    # KNN classification accuracy on embedding
    knn = KNeighborsClassifier(n_neighbors=5)
    scores_le = cross_val_score(knn, embedding_le, y, cv=5)
    print(f"      KNN accuracy (Laplacian Eigenmaps): {scores_le.mean():.3f}")

    # PCA for comparison
    pca = PCA(n_components=n_components)
    embedding_pca = pca.fit_transform(X)
    scores_pca = cross_val_score(knn, embedding_pca, y, cv=5)
    print(f"      KNN accuracy (PCA):                 {scores_pca.mean():.3f}")

    # Trustworthiness
    print(f"      Computing trustworthiness (this takes a moment)...")
    trust_le  = trustworthiness(X, embedding_le, k=10)
    trust_pca = trustworthiness(X, embedding_pca, k=10)
    print(f"      Trustworthiness (Laplacian Eigenmaps): {trust_le:.4f}")
    print(f"      Trustworthiness (PCA):                 {trust_pca:.4f}")


    # 5. Plot
    print(f"\n[5/5] Plotting...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Laplacian Eigenmaps
    for digit in range(10):
        mask = y == digit
        axes[0].scatter(embedding_le[mask, 0], embedding_le[mask, 1],
                       c=[colors[digit]], label=str(digit),
                       s=15, alpha=0.7)
    axes[0].set_title(f'Laplacian Eigenmaps (784D → 2D)\nKNN acc: {scores_le.mean():.3f}  Trust: {trust_le:.3f}')
    axes[0].set_xlabel('f1'); axes[0].set_ylabel('f2')
    axes[0].legend(title='Digit', bbox_to_anchor=(1.05, 1), loc='upper left')

    # PCA
    for digit in range(10):
        mask = y == digit
        axes[1].scatter(embedding_pca[mask, 0], embedding_pca[mask, 1],
                       c=[colors[digit]], label=str(digit),
                       s=15, alpha=0.7)
    axes[1].set_title(f'PCA (784D → 2D)\nKNN acc: {scores_pca.mean():.3f}  Trust: {trust_pca:.3f}')
    axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
    axes[1].legend(title='Digit', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.suptitle('MNIST: Laplacian Eigenmaps vs PCA', fontsize=13)
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    fig.savefig("results/mnist.png", dpi=150, bbox_inches='tight')
    print("      Saved: results/mnist.png")
    plt.show()

    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"  KNN accuracy — LE  : {scores_le.mean():.3f}")
    print(f"  KNN accuracy — PCA : {scores_pca.mean():.3f}")
    print(f"  Trustworthiness — LE  : {trust_le:.4f}")
    print(f"  Trustworthiness — PCA : {trust_pca:.4f}")

    return embedding_le, embedding_pca, y


if __name__ == "__main__":
    run()
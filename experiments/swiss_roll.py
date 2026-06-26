"""
Experiment 1 — Swiss Roll Unrolling

The Swiss Roll is a 2D manifold rolled up in 3D space.
PCA fails — it is linear and cannot handle curvature.
Laplacian Eigenmaps succeeds — it respects local geometry.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph import knn_graph, check_connectivity
from core.laplacian import heat_kernel_weights, compute_laplacian
from core.embedding import compute_embedding


def run(n_samples=1000, k=10, t=10.0):

    print("=" * 50)
    print("Experiment 1: Swiss Roll")
    print("=" * 50)


    # 1. Generating the  Data
    print("\n[1/4] Generating Swiss Roll...")
    X, color = make_swiss_roll(n_samples=n_samples, random_state=42)
    print(f"      Shape: {X.shape}  (3D ambient space)")
    print(f"      Intrinsic dimension: 2")

    # 2. Building  Graph
    print(f"\n[2/4] Building {k}-NN graph...")
    adj = knn_graph(X, k=k)
    connected, n_comp = check_connectivity(adj)
    print(f"      Connected: {connected}, Components: {n_comp}")

    # 3. Laplacian + Embedding
    print(f"\n[3/4] Computing Laplacian and embedding...")
    W = heat_kernel_weights(X, adj, t=t)
    L, D = compute_laplacian(W)
    embedding_le, eigenvalues = compute_embedding(L, D, n_components=2)
    print(f"      Eigenvalues: {eigenvalues.round(4)}")

    # 4. PCA for comparison
    print(f"\n[4/4] Running PCA for comparison...")
    pca = PCA(n_components=2)
    embedding_pca = pca.fit_transform(X)
    print(f"      Variance explained: {pca.explained_variance_ratio_.round(3)}")

    # Plot
    fig = plt.figure(figsize=(18, 5))

    # Original 3D
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X[:,0], X[:,1], X[:,2],
                c=color, cmap='Spectral', s=10, alpha=0.8)
    ax1.set_title('Swiss Roll (3D)', fontsize=12)
    ax1.set_xlabel('X'); 
    ax1.set_ylabel('Y'); 
    ax1.set_zlabel('Z')

    # Laplacian Eigenmaps
    ax2 = fig.add_subplot(132)
    sc2 = ax2.scatter(embedding_le[:,0], embedding_le[:,1],
                      c=color, cmap='Spectral', s=10, alpha=0.8)
    ax2.set_title('Laplacian Eigenmaps\n✓ Unrolled correctly',
                  fontsize=12, color='green')
    ax2.set_xlabel('f₁'); ax2.set_ylabel('f₂')
    plt.colorbar(sc2, ax=ax2)

    # PCA
    ax3 = fig.add_subplot(133)
    sc3 = ax3.scatter(embedding_pca[:,0], embedding_pca[:,1],
                      c=color, cmap='Spectral', s=10, alpha=0.8)
    ax3.set_title('PCA\n✗ Fails on curved manifold',
                  fontsize=12, color='red')
    ax3.set_xlabel('PC₁'); ax3.set_ylabel('PC₂')
    plt.colorbar(sc3, ax=ax3)

    plt.suptitle('Swiss Roll: Laplacian Eigenmaps vs PCA',
                 fontsize=14)
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    fig.savefig("results/swiss_roll.png", dpi=150, bbox_inches='tight')
    print("\nSaved: results/swiss_roll.png")
    plt.show()

    return embedding_le, embedding_pca, eigenvalues


if __name__ == "__main__":
    run()
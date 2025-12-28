import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pathlib import Path

def cluster_data(data: np.ndarray, n_clusters: int = 3) -> tuple[np.ndarray, float]:
    """Perform KMeans clustering and return labels with silhouette score."""
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels, silhouette_score(data, labels)

def save_cluster_visualization(X_original: np.ndarray, X_pca: np.ndarray, 
                                labels_3d: np.ndarray, labels_pca: np.ndarray) -> None:
    """Create and save clustering visualization plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original data clustering (first two dimensions)
    ax1.scatter(X_original[:, 0], X_original[:, 1], c=labels_3d, cmap='viridis')
    ax1.set_title("Clustering Results (Java & OS)")
    ax1.set_xlabel("Java Programming")
    ax1.set_ylabel("Operating Systems")
    
    # PCA reduced clustering
    ax2.scatter(X_pca, np.zeros_like(X_pca), c=labels_pca, cmap='plasma', alpha=0.6)
    ax2.set_title("PCA Reduced to 1D")
    ax2.set_xlabel("Principal Component")
    ax2.set_yticks([])
    
    plt.tight_layout()
    output_path = Path('./data/clustering_results.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {output_path}")

def main() -> None:
    # Load and prepare data
    df = pd.read_excel('./data/stu_scores.xlsx')
    X = df.iloc[:, -3:].values
    
    # Standardize data
    X_scaled = StandardScaler().fit_transform(X)
    
    # Clustering on original 3D data
    labels_3d, score_3d = cluster_data(X_scaled)
    print(f"Original data silhouette score: {score_3d:.4f}")
    
    # PCA reduction and clustering
    X_pca = PCA(n_components=1).fit_transform(X_scaled)
    labels_pca, score_pca = cluster_data(X_pca)
    print(f"PCA reduced data silhouette score: {score_pca:.4f}")
    
    # Save visualization
    save_cluster_visualization(X_scaled, X_pca, labels_3d, labels_pca)

if __name__ == "__main__":
    main()
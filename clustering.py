"""
Clustering Module for Hybrid LLM Optimization

Implements data and parameter clustering to partition the optimization problem
into manageable sub-problems, reducing computational complexity.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class ClusterInfo:
    """Information about a cluster."""
    cluster_id: int
    centroid: np.ndarray
    members: np.ndarray  # Indices of cluster members
    size: int
    mean_distance: float
    

class KMeansClustering:
    """
    K-Means clustering for data and parameter partitioning.
    """
    
    def __init__(self, n_clusters: int, max_iter: int = 100, tol: float = 1e-4, random_state: Optional[int] = None):
        """
        Initialize K-Means clustering.
        
        Args:
            n_clusters: Number of clusters to create
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.converged = False
        
    def fit(self, X: np.ndarray) -> 'KMeansClustering':
        """
        Fit K-Means to data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Self for chaining
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices].copy()
        
        for iteration in range(self.max_iter):
            # Assign points to nearest centroid
            distances = self._compute_distances(X, self.centroids)
            self.labels = np.argmin(distances, axis=1)
            
            # Store old centroids for convergence check
            old_centroids = self.centroids.copy()
            
            # Update centroids
            for k in range(self.n_clusters):
                mask = self.labels == k
                if mask.sum() > 0:
                    self.centroids[k] = X[mask].mean(axis=0)
                # If cluster is empty, reinitialize it randomly
                else:
                    self.centroids[k] = X[np.random.choice(n_samples)]
            
            # Check convergence
            centroid_shift = np.linalg.norm(self.centroids - old_centroids)
            if centroid_shift < self.tol:
                self.converged = True
                break
        
        # Calculate inertia (sum of squared distances to nearest centroid)
        distances = self._compute_distances(X, self.centroids)
        self.inertia = (distances.min(axis=1) ** 2).sum()
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Input data
            
        Returns:
            Cluster labels
        """
        distances = self._compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)
    
    @staticmethod
    def _compute_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distances between data points and centroids.
        
        Args:
            X: Data points (n_samples, n_features)
            centroids: Cluster centers (n_clusters, n_features)
            
        Returns:
            Distance matrix (n_samples, n_clusters)
        """
        # Using broadcasting for efficient computation
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x . c
        X_sqsum = (X ** 2).sum(axis=1, keepdims=True)  # (n_samples, 1)
        C_sqsum = (centroids ** 2).sum(axis=1, keepdims=True).T  # (1, n_clusters)
        dot_product = X @ centroids.T  # (n_samples, n_clusters)
        
        distances = np.sqrt(np.maximum(X_sqsum + C_sqsum - 2 * dot_product, 0))
        return distances
    
    def get_cluster_info(self, X: np.ndarray) -> List[ClusterInfo]:
        """
        Get detailed information about each cluster.
        
        Args:
            X: Original data
            
        Returns:
            List of ClusterInfo objects
        """
        cluster_infos = []
        distances = self._compute_distances(X, self.centroids)
        
        for k in range(self.n_clusters):
            mask = self.labels == k
            member_indices = np.where(mask)[0]
            
            if mask.sum() > 0:
                mean_dist = distances[mask, k].mean()
            else:
                mean_dist = np.inf
            
            info = ClusterInfo(
                cluster_id=k,
                centroid=self.centroids[k],
                members=member_indices,
                size=mask.sum(),
                mean_distance=mean_dist
            )
            cluster_infos.append(info)
        
        return cluster_infos


class ParameterClustering:
    """
    Clusters model parameters by layer and importance for adaptive optimization.
    """
    
    def __init__(self, n_clusters: int = 4):
        """
        Initialize parameter clustering.
        
        Args:
            n_clusters: Number of parameter clusters
        """
        self.n_clusters = n_clusters
        self.clustering = KMeansClustering(n_clusters=n_clusters, random_state=42)
        self.layer_clusters = {}
        self.layer_importance = {}
        
    def cluster_layer_parameters(self, layer_name: str, weights: np.ndarray) -> Dict:
        """
        Cluster parameters within a single layer.
        
        Args:
            layer_name: Name of the layer
            weights: Weight matrix of shape (out_features, in_features) or (features,)
            
        Returns:
            Dictionary with clustering information
        """
        # Flatten weights for clustering
        flat_weights = weights.flatten().reshape(-1, 1)
        
        # Fit clustering
        self.clustering.fit(flat_weights)
        
        # Store results
        cluster_info = self.clustering.get_cluster_info(flat_weights)
        self.layer_clusters[layer_name] = {
            'info': cluster_info,
            'labels': self.clustering.labels.reshape(weights.shape),
            'centroids': self.clustering.centroids.flatten()
        }
        
        return self.layer_clusters[layer_name]
    
    def get_cluster_importance(self, layer_name: str) -> np.ndarray:
        """
        Get importance scores for clusters in a layer.
        Based on cluster size and cohesion.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Importance scores for each cluster
        """
        if layer_name not in self.layer_clusters:
            return None
        
        cluster_info = self.layer_clusters[layer_name]['info']
        
        importances = []
        total_size = sum(info.size for info in cluster_info)
        
        for info in cluster_info:
            # Importance based on size (larger clusters) and cohesion (smaller mean distance)
            size_factor = info.size / max(total_size, 1)
            cohesion_factor = 1.0 / (1.0 + info.mean_distance)  # Normalized cohesion
            
            importance = size_factor * cohesion_factor
            importances.append(importance)
        
        importances = np.array(importances)
        self.layer_importance[layer_name] = importances / importances.sum()
        
        return self.layer_importance[layer_name]
    
    def suggest_quantization_precision(self, layer_name: str) -> np.ndarray:
        """
        Suggest per-cluster quantization precision based on importance.
        Higher importance clusters get higher precision (less aggressive quantization).
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Precision allocation for each cluster (normalized to sum to 1.58)
        """
        importances = self.get_cluster_importance(layer_name)
        if importances is None:
            return None
        
        # Allocate precision proportional to importance
        # Total budget is 1.58 bits
        precision = importances * 1.58
        
        return precision


class DataClustering:
    """
    Clusters training data for efficient mini-batch construction.
    """
    
    def __init__(self, n_clusters: int = 8):
        """
        Initialize data clustering.
        
        Args:
            n_clusters: Number of data clusters
        """
        self.n_clusters = n_clusters
        self.clustering = KMeansClustering(n_clusters=n_clusters, random_state=42)
        self.cluster_assignments = None
        
    def cluster_embeddings(self, embeddings: np.ndarray) -> Tuple[np.ndarray, List[ClusterInfo]]:
        """
        Cluster data based on embeddings (e.g., token embeddings).
        
        Args:
            embeddings: Input embeddings of shape (n_samples, embedding_dim)
            
        Returns:
            Tuple of (cluster_labels, cluster_infos)
        """
        self.clustering.fit(embeddings)
        self.cluster_assignments = self.clustering.labels
        
        cluster_infos = self.clustering.get_cluster_info(embeddings)
        return self.cluster_assignments, cluster_infos
    
    def get_balanced_clusters(self) -> List[np.ndarray]:
        """
        Get cluster indices ensuring balanced cluster sizes.
        
        Returns:
            List of arrays, each containing indices for one cluster
        """
        clusters = []
        for k in range(self.n_clusters):
            cluster_indices = np.where(self.cluster_assignments == k)[0]
            clusters.append(cluster_indices)
        
        return clusters
    
    def suggest_mini_batch_strategy(self) -> Dict:
        """
        Suggest mini-batch construction strategy based on clusters.
        
        Returns:
            Dictionary with suggested strategy
        """
        cluster_sizes = [np.sum(self.cluster_assignments == k) for k in range(self.n_clusters)]
        
        return {
            'cluster_sizes': cluster_sizes,
            'total_samples': sum(cluster_sizes),
            'balanced_batch_size': min(cluster_sizes),
            'strategy': 'sample_from_each_cluster'
        }


if __name__ == "__main__":
    # Test K-Means clustering
    print("=== K-Means Clustering Test ===")
    
    # Create synthetic data
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(100, 2) + [0, 0],
        np.random.randn(100, 2) + [5, 5],
        np.random.randn(100, 2) + [-5, 5],
    ])
    
    kmeans = KMeansClustering(n_clusters=3, random_state=42)
    kmeans.fit(X)
    
    print(f"Converged: {kmeans.converged}")
    print(f"Inertia: {kmeans.inertia:.4f}")
    print(f"Cluster labels shape: {kmeans.labels.shape}")
    
    # Test parameter clustering
    print("\n=== Parameter Clustering Test ===")
    param_clusterer = ParameterClustering(n_clusters=4)
    
    layer_weights = np.random.randn(64, 64)
    cluster_info = param_clusterer.cluster_layer_parameters("test_layer", layer_weights)
    
    precision = param_clusterer.suggest_quantization_precision("test_layer")
    print(f"Precision allocation: {precision}")
    print(f"Total precision: {precision.sum():.2f} bits")
    
    # Test data clustering
    print("\n=== Data Clustering Test ===")
    data_clusterer = DataClustering(n_clusters=4)
    embeddings = np.random.randn(1000, 128)
    
    labels, infos = data_clusterer.cluster_embeddings(embeddings)
    strategy = data_clusterer.suggest_mini_batch_strategy()
    
    print(f"Cluster sizes: {strategy['cluster_sizes']}")
    print(f"Suggested balanced batch size: {strategy['balanced_batch_size']}")

import numpy as np
from sklearn.datasets import make_blobs

class CorrelatedClusterGenerator:
    def __init__(
        self, n_samples, n_features, n_clusters, cluster_std, correlation=0.5, random_state=None
    ):
        """
        Initialize the parameters for generating correlated clusters.
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.correlation = correlation
        self.random_state = random_state
        self.cluster_std = cluster_std

    def generate_clusters(self):
        """
        Generate correlated clusters.

        Returns:
        - X: Feature matrix with samples for each cluster.
        - y: Cluster labels for each sample.
        - centers: True centers of the blobs
        """
        np.random.seed(self.random_state)

        # Create initial blobs and cluster centers
        X, y, blob_centers = make_blobs(
            n_samples=self.n_samples,
            centers=self.n_clusters,
            n_features=self.n_features,
            random_state=self.random_state,
            cluster_std=self.cluster_std,
            return_centers=True   
        )

        centers = blob_centers

        X, y = [], []

        # Generate correlated samples for each cluster
        for i, center in enumerate(centers):
            std = self.cluster_std[i] if isinstance(self.cluster_std, (list, np.ndarray)) else self.cluster_std

            cov_matrix = np.full((self.n_features, self.n_features), self.correlation * std**2)
            np.fill_diagonal(cov_matrix, std**2)

            cluster_samples = np.random.multivariate_normal(
                mean=center, cov=cov_matrix, size=self.n_samples // self.n_clusters
            )

            X.append(cluster_samples)
            y.append(np.full(self.n_samples // self.n_clusters, i))

        X = np.vstack(X)
        y = np.hstack(y)

        return X, y, centers

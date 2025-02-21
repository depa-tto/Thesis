import numpy as np
from sklearn.datasets import make_blobs

class CorrelatedClusterGenerator:
    def __init__(self, n_samples, n_features, n_clusters, correlation=0.5, random_state=None):
        """
        Initialize the parameters for generating correlated clusters.

        Parameters:
        - n_samples: Total number of samples.
        - n_features: Number of features.
        - n_clusters: Number of clusters.
        - correlation: The correlation between features within a cluster (default is 0.5).
        - random_state: The seed for the random number generator (default is None).
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.correlation = correlation
        self.random_state = random_state

    def generate_clusters(self):
        """
        Generate correlated clusters.

        Returns:
        - X: Feature matrix with samples for each cluster.
        - y: Cluster labels for each sample.
        """
        np.random.seed(self.random_state)

        # Create initial blobs and cluster centers
        X_blobs, y_blobs = make_blobs(n_samples=self.n_samples, centers=self.n_clusters, 
                                       n_features=self.n_features, random_state=self.random_state)

        # Calculate the centers of the clusters
        centers = np.array([X_blobs[y_blobs == i].mean(axis=0) for i in range(self.n_clusters)])

        # Initialize empty lists to hold the samples and labels
        X, y = [], []

        # Generate correlated samples for each cluster
        for i, center in enumerate(centers):
            cov_matrix = np.full((self.n_features, self.n_features), self.correlation)
            np.fill_diagonal(cov_matrix, 1)

            cluster_samples = np.random.multivariate_normal(mean=center, cov=cov_matrix, size=self.n_samples // self.n_clusters)

            X.append(cluster_samples)
            y.append(np.full(self.n_samples // self.n_clusters, i))

        # Stack the cluster samples and labels
        X = np.vstack(X)
        y = np.hstack(y)

        return X, y

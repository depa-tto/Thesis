import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.datasets import make_spd_matrix
class CorrelatedClusterGenerator:
    """
    A generator for synthetic datasets with correlated features and well-separated clusters.
    
    This class creates synthetic datasets specifically designed for testing dimensionality 
    reduction algorithms followed by clustering algorithms. It generates clusters with:
    - Controllable correlations between features
    - Variable standard deviations per cluster
    - Guaranteed separation between cluster centers
    - Realistic covariance structures
    
    Attributes:
        n_samples (int): Total number of samples to generate
        n_features (int): Number of features (dimensions) in the dataset
        n_clusters (int): Number of clusters to generate
        cluster_std (float or list): Standard deviation(s) for clusters
        correlation (float): Level of correlation between features (0-1)
        separation_factor (float): Minimum separation between clusters as multiple of std
        random_state (int): Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_samples,
        n_features,
        n_clusters,
        cluster_std=1.0,
        correlation=0.5,
        separation_factor=3.0,
        random_state=None,
    ):
        """
        Initialize the correlated cluster generator with specified parameters.
        
        Args:
            n_samples (int): Total number of data points to generate
            n_features (int): Dimensionality of the feature space
            n_clusters (int): Number of distinct clusters to create
            cluster_std (float or list): Standard deviation for each cluster.
                                       If float, same std for all clusters.
                                       If list, different std for each cluster.
            correlation (float): Base correlation level between features (0.0 to 1.0)
            separation_factor (float): Minimum distance between cluster centers
                                     expressed as multiple of standard deviation
            random_state (int): Seed for random number generator for reproducibility
        """
        self.n_samples = n_samples                  # Store total number of samples
        self.n_features = n_features                # Store feature dimensionality
        self.n_clusters = n_clusters                # Store number of clusters
        self.cluster_std = cluster_std              # Store standard deviation configuration
        self.correlation = correlation              # Store correlation level
        self.separation_factor = separation_factor  # Store separation factor
        self.random_state = random_state            # Store random seed

    def generate_clusters(self):
        """
        Generate the complete synthetic dataset with correlated clusters.
        
        This is the main method that orchestrates the entire generation process:
        1. Sets random seed for reproducibility
        2. Generates well-separated cluster centers
        3. Distributes samples among clusters
        4. Creates correlated samples for each cluster
        5. Combines and shuffles all data
        
        Returns:
            tuple: (X, y, centers) where:
                - X (ndarray): Feature matrix of shape (n_samples, n_features)
                - y (ndarray): Cluster labels of shape (n_samples,)
                - centers (ndarray): True cluster centers of shape (n_clusters, n_features)
        """
        # Set random seed to ensure reproducible results
        np.random.seed(self.random_state)

        # Generate cluster centers with controlled separation
        centers = self._generate_separated_centers()

        # Initialize lists to store samples and labels for all clusters
        X, y = [], []
        
        # Calculate how many samples each cluster should have
        samples_per_cluster = self._distribute_samples()

        # Generate samples for each cluster individually
        for i, (center, n_samples_cluster) in enumerate(
            zip(centers, samples_per_cluster)
        ):
            # Get the standard deviation for this specific cluster
            std = self._get_cluster_std(i)

            # Create covariance matrix with desired correlations for this cluster
            cov_matrix = self._generate_covariance_matrix(std)

            # Generate multivariate normal samples around the cluster center
            cluster_samples = np.random.multivariate_normal(
                mean=center,                   # Center point for this cluster
                cov=cov_matrix,                # Covariance matrix with correlations
                size=n_samples_cluster         # Number of samples for this cluster
            )

            # Add generated samples to the overall dataset
            X.append(cluster_samples)
            # Create labels for this cluster (all samples get label 'i')
            y.append(np.full(n_samples_cluster, i))

        # Combine all cluster samples into single arrays
        X = np.vstack(X)  # Stack vertically to create (n_samples, n_features) matrix
        y = np.hstack(y)  # Stack horizontally to create (n_samples,) label vector

        # Shuffle the data to avoid any ordering bias in the dataset
        shuffle_idx = np.random.permutation(len(X))

        # Return shuffled data, labels, and original cluster centers
        return X[shuffle_idx], y[shuffle_idx], centers

    def _get_cluster_std(self, cluster_idx):
        """
        Retrieve the standard deviation for a specific cluster.
        
        This method handles both uniform standard deviation (single float value)
        and variable standard deviations (list/array with different values per cluster).
        
        Args:
            cluster_idx (int): Index of the cluster (0 to n_clusters-1)
            
        Returns:
            float: Standard deviation value for the specified cluster
        """
        # Check if cluster_std is a list or array (different std per cluster)
        if isinstance(self.cluster_std, (list, np.ndarray)):
            return self.cluster_std[cluster_idx]  # Return specific std for this cluster
        # If cluster_std is a single value, use it for all clusters
        return self.cluster_std

    def _generate_separated_centers(self):
        """
        Generate cluster centers with guaranteed minimum separation.
        
        This method ensures that clusters are well-separated in the feature space
        by enforcing a minimum distance between any two cluster centers. The minimum
        distance is calculated as separation_factor * maximum_standard_deviation.
        If cluster centers were generated too close to each other (especially 
        if the standard deviation was large), the clusters would randomly overlap. 
        In this case, any clustering algorithm would fail, and you wouldn't know whether
        the failure was due to the algorithm itself or poor data quality.

        With Guarantee: By ensuring a minimum distance we are defining a clear ground truth. 
        ensuring thar the clusters are separable in high dimension
        
        Returns:
            ndarray: Array of cluster centers with shape (n_clusters, n_features)
        """
        # Calculate the maximum standard deviation among all clusters
        if isinstance(self.cluster_std, (list, np.ndarray)):
            max_std = max(self.cluster_std)  # Find maximum if multiple std values
        else:
            max_std = self.cluster_std       # Use single value if uniform std

        # Calculate minimum required distance between cluster centers
        min_distance = self.separation_factor * max_std
        centers = []  # List to store generated centers

        # Generate centers one by one, ensuring proper separation
        for _ in range(self.n_clusters):
            attempts = 0  # Counter to avoid infinite loops
            
            # Try to find a valid center position
            while attempts < 1000:  # Maximum attempts to prevent infinite loops
                # Generate random center coordinates in range [-10, 10]
                center = np.random.uniform(-10, 10, self.n_features)

                # If this is the first center, accept it immediately
                if len(centers) == 0:
                    centers.append(center)
                    break

                # Calculate distances from this center to all existing centers
                distances = [np.linalg.norm(center - c) for c in centers]
                
                # Check if minimum distance requirement is satisfied
                if min(distances) >= min_distance:
                    centers.append(center)  # Accept this center
                    break
                    
                attempts += 1  # Increment attempt counter

        # Convert list to numpy array and return
        return np.array(centers)

    def _generate_covariance_matrix(self, std):
        """
        Generate a valid covariance matrix with controlled correlations.
        
        This method creates a covariance matrix that:
        1. Is guaranteed to be positive semi-definite (valid for multivariate normal)
        2. Has the specified standard deviation on the diagonal
        3. Contains realistic correlation structure between features
        
        The method uses sklearn's make_spd_matrix to ensure mathematical validity,
        then scales it to achieve the desired correlation level and standard deviation.
        
        Args:
            std (float): Standard deviation for this cluster
            
        Returns:
            ndarray: Covariance matrix of shape (n_features, n_features)
        """

        try:
            # Generate a random positive semi-definite matrix
            # This guarantees the matrix will be mathematically valid
            base_cov = make_spd_matrix(
                n_dim=self.n_features,          # Dimensionality of the matrix
                random_state=self.random_state  # Use same random seed for reproducibility
            )

            # Normalize the matrix to extract correlation structure
            # First, get the square root of diagonal elements (standard deviations)
            diag_sqrt = np.sqrt(np.diag(base_cov))
            
            # Convert covariance matrix to correlation matrix
            # by dividing by outer product of standard deviations
            corr_matrix = base_cov / np.outer(diag_sqrt, diag_sqrt)

            # Determine the correlation scaling factor
            if isinstance(self.correlation, (int, float)):
                corr_factor = self.correlation           # Use single correlation value
            else:
                corr_factor = np.mean(self.correlation)  # Use average if range given

            # Scale the off-diagonal correlations to desired level
            # Formula: scaled_corr = I + corr_factor * (original_corr - I)
            # This preserves the identity matrix (diagonal = 1) while scaling correlations
            scaled_corr = (
                corr_matrix - np.eye(self.n_features)  # Remove identity matrix
            ) * corr_factor + np.eye(self.n_features)   # Scale and add back identity

            # Convert correlation matrix back to covariance matrix
            # by scaling with the desired standard deviation
            final_cov = scaled_corr * (std**2)

            return final_cov

        except (ImportError, ValueError, np.linalg.LinAlgError) as e:
            # Fallback strategy if the sophisticated method fails
            # Print warning message with error details
            print(f"Warning: Using diagonal covariance matrix for std={std}. Error: {e}")
            
            # Return simple diagonal covariance matrix (no correlations)
            return np.eye(self.n_features) * (std ** 2)

    def _distribute_samples(self):
        """
        Distribute the total number of samples among clusters as evenly as possible.
        
        This method handles the case where n_samples is not perfectly divisible
        by n_clusters. It ensures that:
        1. All samples are assigned to clusters
        2. Distribution is as even as possible
        3. Any remaining samples are distributed to the first clusters
        
        Returns:
            list: Number of samples for each cluster
        """
        # Calculate base number of samples per cluster (integer division)
        base_samples = self.n_samples // self.n_clusters
        
        # Calculate remaining samples that couldn't be evenly distributed
        remainder = self.n_samples % self.n_clusters

        # Start with base number of samples for each cluster
        samples_per_cluster = [base_samples] * self.n_clusters

        # Distribute remaining samples to the first 'remainder' clusters
        # This ensures total samples equals n_samples exactly
        for i in range(remainder):
            samples_per_cluster[i] += 1  # Add one extra sample to cluster i

        return samples_per_cluster


def trimmed_clustering(X, n_clusters, trim_fraction=0.1, max_iter=100, tol=1e-4, random_state=42):
    """
    Implementation of trimmed k-means clustering.
    
    This algorithm iteratively assigns points to the nearest cluster centers,
    discards a fraction of farthest points as outliers, and updates the centroids.
    """
    rng = np.random.default_rng(random_state)

    # Random initialization of cluster centers
    init_idx = rng.choice(len(X), size=n_clusters, replace=False)
    centroids = X[init_idx]
    prev_centroids = None

    for _ in range(max_iter):
        # Compute distances to centroids
        distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)
        min_distances = distances[np.arange(len(X)), labels]

        # Identify points to trim (outliers)
        threshold = np.percentile(min_distances, 100 * (1 - trim_fraction))
        mask = min_distances <= threshold

        # Update centroids using only trimmed points
        new_centroids = []
        for k in range(n_clusters):
            cluster_points = X[mask & (labels == k)]
            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                new_centroids.append(X[rng.choice(len(X))])
        new_centroids = np.vstack(new_centroids)

        # Check convergence
        if prev_centroids is not None and np.allclose(new_centroids, prev_centroids, atol=tol):
            break

        prev_centroids = centroids
        centroids = new_centroids

    # Fit final KMeans on trimmed data
    trimmed_X = X[mask]
    trimmed_indices = np.where(mask)[0]
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(trimmed_X)

    return kmeans, trimmed_indices, np.where(~mask)[0]



def evaluate_clustering(X, y_true, y_pred):
    """
    Evaluate clustering performance using ARI, NMI, and Silhouette score.
    
    Args:
        X (ndarray): Feature matrix
        y_true (array): Ground-truth labels
        y_pred (array): Predicted cluster labels
    
    Returns:
        dict: Scores for ARI, NMI, and Silhouette
    """
    scores = {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "Silhouette": silhouette_score(X, y_pred),
    }
    return scores



def create_cluster_dataset(
    n_clusters=3,
    n_samples=10000,
    n_features=20,
    cluster_std=1.0,
    correlation=0.2,
    separation_factor=3.0,
    random_state=42,
    noise_std=0.0,
    sample_limits=None
):
    """
    Generate a synthetic dataset with correlated clusters and return it as a DataFrame.

    Args:
        n_clusters (int): number of clusters
        n_samples (int): total number of data points to generate
        n_features (int): number of features (dimensions)
        cluster_std (float or list): standard deviation for clusters (single value or per-cluster list)
        correlation (float): correlation level between features 
        separation_factor (float): minimum separation between cluster centers
        random_state (int): random seed for reproducibility
        noise_std (float): additional Gaussian noise to add to the dataset
        sample_limits (dict): subsample size per cluster, 
                              e.g., {cluster_idx: n_samples_to_keep}

    Returns:
        pd.DataFrame: dataset with features and a "target" column containing cluster labels
    """

    # Initialize the generator with the desired parameters
    gen = CorrelatedClusterGenerator(
        n_samples=n_samples,
        n_features=n_features,
        n_clusters=n_clusters,
        cluster_std=cluster_std,
        correlation=correlation,
        separation_factor=separation_factor,
        random_state=random_state
    )

    # Generate samples, labels, and true cluster centers
    X, y, _ = gen.generate_clusters()

    # Optionally add Gaussian noise to the data
    if noise_std > 0:
        X = X + np.random.normal(scale=noise_std, size=X.shape)

    # Optionally subsample specific clusters
    if sample_limits:
        X_new, y_new = [], []
        for cluster_idx, n_keep in sample_limits.items():
            mask = np.array(y) == cluster_idx
            X_cluster = X[mask][:n_keep]   # Keep only the first n_keep samples
            y_cluster = [cluster_idx] * len(X_cluster)
            X_new.append(X_cluster)
            y_new.extend(y_cluster)
        X = np.vstack(X_new)
        y = np.array(y_new)

    # Build DataFrame with features + target labels
    df = pd.DataFrame(X, columns=[f"var_{i+1}" for i in range(X.shape[1])])
    df["target"] = y

    return df
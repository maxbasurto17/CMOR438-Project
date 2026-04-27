import numpy as np

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        """
        Initializes the DBSCAN clustering model.

        Parameters
        ----------
        eps : float, default=0.5
            The maximum distance between two samples for one to be considered 
            as in the neighborhood of the other.
        min_samples : int, default=5
            The number of samples in a neighborhood for a point to be 
            considered as a core point.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def _get_neighbors(self, X, sample_idx):
        """Returns the indices of all points within eps distance of X[sample_idx]."""
        distances = np.linalg.norm(X - X[sample_idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def train(self, X):
        """
        Performs DBSCAN clustering on the dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to cluster.

        Returns
        -------
        self : object
        """
        n_samples = X.shape[0]
        # Initialize labels as -1 (Noise/Unvisited)
        self.labels_ = np.full(n_samples, -1)
        cluster_id = 0

        for i in range(n_samples):
            # Skip if already visited
            if self.labels_[i] != -1:
                continue

            neighbors = self._get_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                # Label as noise for now, but could be changed later if reachable
                self.labels_[i] = -1
            else:
                # Found a core point, start a new cluster
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1
        
        return self

    def _expand_cluster(self, X, sample_idx, neighbors, cluster_id):
        """Recursively expands the cluster using density-reachable points."""
        self.labels_[sample_idx] = cluster_id
        
        # We use a queue-like approach to explore neighbors
        queue = list(neighbors)
        
        while queue:
            neighbor_idx = queue.pop(0)
            
            # If previously marked as noise, it's a border point
            if self.labels_[neighbor_idx] == -1:
                self.labels_[neighbor_idx] = cluster_id
            
            # If not visited, investigate its neighborhood
            elif self.labels_[neighbor_idx] == -1: # unvisited
                self.labels_[neighbor_idx] = cluster_id
                
                new_neighbors = self._get_neighbors(X, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    queue.extend(new_neighbors)

    def predict(self, X):
        """
        DBSCAN is not traditionally used for predicting new points, but 
        we return the labels assigned during training.
        """
        return self.labels_
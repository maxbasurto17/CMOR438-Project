import numpy as np
from rice_ml.measures.validation import euclidean_distance

class KMeans:
    """
    k-Means clustering algorithm implementation.
    
    Parameters
    ----------
    k : int
        The number of clusters to form as well as the number of centroids to generate.
    max_iter : int, optional (default=100)
        Maximum number of iterations of the k-means algorithm for a single run.
        
    Attributes
    ----------
    centers_ : numpy.ndarray
        Coordinates of cluster centers.
    labels_ : numpy.ndarray
        Labels of each point.
    """
    
    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centers_ = None
        self.labels_ = None
        
    def _assign_labels(self, X, centers):
        """
        Assigns each data point to the closest cluster center.
        
        Parameters
        ----------
        X : numpy.ndarray
            The dataset of shape (n_samples, n_features).
        centers : numpy.ndarray
            The current cluster centers of shape (k, n_features).
            
        Returns
        -------
        numpy.ndarray
            Array of assigned cluster indices for each point.
        """
        labels = np.zeros(X.shape[0], dtype=int)
        
        for i in range(X.shape[0]):
            distances = [euclidean_distance(X[i], center) for center in centers]
            labels[i] = np.argmin(distances)
            
        return labels
    
    def _update_centers(self, X, labels):
        """
        Updates the cluster centers by calculating the mean of the assigned points.
        
        Parameters
        ----------
        X : numpy.ndarray
            The dataset of shape (n_samples, n_features).
        labels : numpy.ndarray
            The current cluster assignments for each point.
            
        Returns
        -------
        numpy.ndarray
            The newly calculated cluster centers.
        """
        new_centers = np.zeros((self.k, X.shape[1]))
        
        for j in range(self.k):
            # Extract points assigned to cluster j
            cluster_points = X[labels == j]
            
            if len(cluster_points) > 0:
                new_centers[j] = np.mean(cluster_points, axis=0)
            else:
                # Fallback if a cluster ends up empty: re-initialize with a random point
                new_centers[j] = X[np.random.choice(X.shape[0])]
                
        return new_centers

    def fit(self, X):
        """
        Compute k-means clustering.
        
        Parameters
        ----------
        X : numpy.ndarray
            The dataset to cluster, of shape (n_samples, n_features).
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Initialize centers randomly from the dataset
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centers_ = X[random_indices]
        
        for _ in range(self.max_iter):
            # Assign clusters based on current centers
            self.labels_ = self._assign_labels(X, self.centers_)
            
            # Update centers based on current assignments
            new_centers = self._update_centers(X, self.labels_)
            
            # Check for convergence
            if np.allclose(self.centers_, new_centers):
                break
                
            self.centers_ = new_centers
            
        return self

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
        
        Parameters
        ----------
        X : numpy.ndarray
            New data to predict.
            
        Returns
        -------
        numpy.ndarray
            Index of the cluster each sample belongs to.
        """
        if self.centers_ is None:
            raise RuntimeError("The model has not been fitted yet. Call 'fit' first.")
        return self._assign_labels(X, self.centers_)
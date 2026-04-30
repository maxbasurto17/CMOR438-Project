import numpy as np

class PCA:
    def __init__(self, n_components=2):
        """
        Initializes the Principal Component Analysis (PCA) model.

        PCA reduces the dimensionality of data by projecting it onto the 
        directions of maximum variance.

        Parameters
        ----------
        n_components : int, default=2
            The number of principal components to keep.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
        self.explained_variance_ratio = None 

    def train(self, X):
        """
        Fits the PCA model by calculating the principal components of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
-------
        self : object
        """
        # Center the data (Zero-mean)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute the Covariance Matrix
        # rowvar=False because columns are features
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Calculate Eigenvalues and Eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvectors by eigenvalues in descending order
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[:, idxs]

        # Calculate the ratio before slicing eigenvalues
        total_variance = np.sum(eigenvalues)
        
        # Store the top n_components
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        
        # Calculate and store the ratio
        self.explained_variance_ratio = self.explained_variance / total_variance
        
        return self

    def transform(self, X):
        """
        Projects the data into the reduced dimensional space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.

        Returns
-------
        X_projected : ndarray of shape (n_samples, n_components)
            The data projected onto the principal components.
        """
        # Center the data using the mean learned during training
        X_centered = X - self.mean
        # Project data: X_new = X_centered @ W
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        """Fits the model and returns the projected data in one step."""
        return self.train(X).transform(X)
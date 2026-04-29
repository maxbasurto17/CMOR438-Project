import numpy as np

class StandardScaler:
    def __init__(self):
        """
        Standardize features by removing the mean and scaling to unit variance.
        
        The standard score of a sample `x` is calculated as:
     from .standard_scaler import StandardScaler
from .one_hot_encoder import OneHotEncoder   $$z = \frac{x - u}{s}$$
        where `u` is the mean and `s` is the standard deviation.
        """
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """
        Compute the mean and standard deviation to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation.
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero by setting std to 1 where it is 0
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        """
        Perform standardization by centering and scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.
        """
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)
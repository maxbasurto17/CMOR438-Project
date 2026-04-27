import numpy as np

class OneHotEncoder:
    def __init__(self):
        """
        Encode categorical features as a one-hot numeric array.
        """
        self.categories_ = None

    def fit(self, y):
        """
        Identify unique categories in the input.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            The categorical labels to encode.
        """
        self.categories_ = np.unique(y)
        return self

    def transform(self, y):
        """
        Transform labels into a one-hot encoded matrix.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            The labels to transform.
        """
        one_hot = np.zeros((len(y), len(self.categories_)))
        cat_to_idx = {cat: i for i, cat in enumerate(self.categories_)}
        
        for i, val in enumerate(y):
            if val in cat_to_idx:
                one_hot[i, cat_to_idx[val]] = 1.0
                
        return one_hot

    def fit_transform(self, y):
        """Fit to labels, then transform them."""
        return self.fit(y).transform(y)
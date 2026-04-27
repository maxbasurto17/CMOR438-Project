import numpy as np

class RegressionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        """
        Initializes the Regression Tree.

        Parameters
        ----------
        max_depth : int, default=5
            The maximum depth of the tree. Limits the number of splits to prevent overfitting.
        min_samples_split : int, default=2
            The minimum number of samples required to split an internal node.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _calculate_variance(self, y):
        """Computes the variance of the target labels for a given set."""
        if len(y) == 0:
            return 0
        return np.var(y)

    def _best_split(self, X, y):
        """
        Finds the feature and threshold that maximize variance reduction.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The training data features.
        y : ndarray of shape (n_samples,)
            The continuous target values.

        Returns
        -------
        best_idx : int or None
            Index of the feature to split on.
        best_thr : float or None
            Threshold value for the split.
        """
        m, n = X.shape
        if m < self.min_samples_split:
            return None, None

        best_var_reduction = -1
        best_idx, best_thr = None, None
        current_var = self._calculate_variance(y)

        for idx in range(n):
            thresholds = np.unique(X[:, idx])
            for thr in thresholds:
                left_indices = X[:, idx] <= thr
                y_left, y_right = y[left_indices], y[~left_indices]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                n_l, n_r = len(y_left), len(y_right)
                weighted_var = (n_l / m) * self._calculate_variance(y_left) + \
                               (n_r / m) * self._calculate_variance(y_right)
                
                var_reduction = current_var - weighted_var

                if var_reduction > best_var_reduction:
                    best_var_reduction = var_reduction
                    best_idx = idx
                    best_thr = thr

        return best_idx, best_thr

    def _build_tree(self, X, y, depth=0):
        """Recursively builds the regression tree by splitting nodes."""
        value = np.mean(y)
        node = {"value": value}

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] <= thr
                node["feature_index"] = idx
                node["threshold"] = thr
                node["left"] = self._build_tree(X[indices_left], y[indices_left], depth + 1)
                node["right"] = self._build_tree(X[~indices_left], y[~indices_left], depth + 1)
        return node

    def train(self, X, y):
        """
        Builds the regression tree using the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Continuous target values.

        Returns
        -------
        self : object
        """
        self.tree = self._build_tree(X, y)
        return self

    def _predict_one(self, inputs, node):
        """Recursively traverses the tree to predict a value for one sample."""
        if "feature_index" not in node:
            return node["value"]
        if inputs[node["feature_index"]] <= node["threshold"]:
            return self._predict_one(inputs, node["left"])
        return self._predict_one(inputs, node["right"])

    def predict(self, X):
        """
        Predicts regression values for the provided samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            The predicted continuous values.
        """
        return np.array([self._predict_one(inputs, self.tree) for inputs in X])
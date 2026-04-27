import numpy as np

class DecisionTree:
    def __init__(self, max_depth=10):
        """
        Initializes the Decision Tree classifier.
        
        Parameters
        ----------
        max_depth : int, default=10
            The maximum depth of the tree to prevent overfitting.
        """
        self.max_depth = max_depth
        self.tree = None

    def _gini(self, y):
        """Computes the Gini Impurity of a label set."""
        m = len(y)
        if m == 0: return 0
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _best_split(self, X, y):
        """
        Iterates over features and thresholds to find the best Gini-based split.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.
        y : ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        best_idx : int or None
            Feature index of the optimal split.
        best_thr : float or None
            Threshold value for the optimal split.
        """
        m, n = X.shape
        if m <= 1: return None, None
        
        best_gini = self._gini(y)
        best_idx, best_thr = None, None
        
        for idx in range(n):
            thresholds = np.unique(X[:, idx])
            for thr in thresholds:
                left_indices = X[:, idx] <= thr
                y_left = y[left_indices]
                y_right = y[~left_indices]
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                gini = (len(y_left) * self._gini(y_left) + 
                        len(y_right) * self._gini(y_right)) / m
                
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = thr
                    
        return best_idx, best_thr

    def _build_tree(self, X, y, depth=0):
        """
        Recursively builds the decision tree nodes until depth or purity is reached.
        """
        unique_classes = np.unique(y)
        num_samples_per_class = [np.sum(y == i) for i in unique_classes]
        predicted_class = unique_classes[np.argmax(num_samples_per_class)]
        
        node = {"class": predicted_class}

        if len(unique_classes) == 1:
            return node

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
        Fits the Decision Tree to the classification data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input features.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : object
        """
        self.tree = self._build_tree(X, y)
        return self

    def _predict_one(self, inputs, node):
        """Traverses the internal dictionary to find the leaf class for one sample."""
        if "feature_index" not in node:
            return node["class"]
        if inputs[node["feature_index"]] <= node["threshold"]:
            return self._predict_one(inputs, node["left"])
        else:
            return self._predict_one(inputs, node["right"])

    def predict(self, X):
        """
        Predicts class labels for the provided samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
-------
        predictions : ndarray of shape (n_samples,)
            The predicted class labels.
        """
        return np.array([self._predict_one(inputs, self.tree) for inputs in X])
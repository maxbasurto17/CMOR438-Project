import numpy as np

class DecisionTree:
    def __init__(self, max_depth=10, max_features=None):
        """
        Initializes the Decision Tree classifier.
        
        Parameters
        ----------
        max_depth : int, default=10
            The maximum depth of the tree to prevent overfitting.
        max_features : int, optional
            The number of random features to consider when looking for the best 
            split. If None, all features are considered. This is primarily 
            used by the RandomForest class.
        """
        self.max_depth = max_depth
        self.max_features = max_features
        self.tree = None

    def _gini(self, y):
        """
        Computes the Gini Impurity of a set of labels.
        
        Gini impurity is a measure of how often a randomly chosen element from 
        the set would be incorrectly labeled if it was randomly labeled 
        according to the distribution of labels in the subset.
        """
        m = len(y)
        if m == 0:
            return 0
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _best_split(self, X, y, max_features):
        """
        Finds the best feature and threshold to split the data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input feature matrix.
        y : ndarray of shape (n_samples,)
            The target labels.
        max_features : int or None
            The number of random features to evaluate for the split.

        Returns
        -------
        best_idx : int or None
            The index of the feature used for the best split.
        best_thr : float or None
            The threshold value used for the best split.
        """
        m, n = X.shape
        if m <= 1:
            return None, None
        
        best_gini = self._gini(y)
        best_idx, best_thr = None, None
        
        # --- FEATURE SUBSAMPLING ---
        # Select a random subset of feature indices if max_features is specified
        if max_features is not None and max_features < n:
            feature_indices = np.random.choice(n, max_features, replace=False)
        else:
            feature_indices = range(n)
        
        for idx in feature_indices:
            thresholds = np.unique(X[:, idx])
            for thr in thresholds:
                left_indices = X[:, idx] <= thr
                y_left, y_right = y[left_indices], y[~left_indices]
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                # Weighted average of Gini Impurity for the children nodes
                gini = (len(y_left) * self._gini(y_left) + 
                        len(y_right) * self._gini(y_right)) / m
                
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = thr
                    
        return best_idx, best_thr

    def _build_tree(self, X, y, depth=0):
        """
        Recursive function to build the tree nodes.
        
        At each node, we determine the most frequent class and decide whether
        to split further based on depth and purity.
        """
        unique_classes = np.unique(y)
        # Calculate the distribution to determine the majority class
        num_samples_per_class = [np.sum(y == i) for i in unique_classes]
        
        # Mapping index of max count back to the actual class label
        predicted_class = unique_classes[np.argmax(num_samples_per_class)]
        
        node = {"class": predicted_class}

        # Stop if the node is pure (only one class remains)
        if len(unique_classes) == 1:
            return node

        # Recursive split if max depth hasn't been reached
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y, self.max_features)
            if idx is not None:
                indices_left = X[:, idx] <= thr
                node["feature_index"] = idx
                node["threshold"] = thr
                node["left"] = self._build_tree(X[indices_left], y[indices_left], depth + 1)
                node["right"] = self._build_tree(X[~indices_left], y[~indices_left], depth + 1)
        return node

    def train(self, X, y):
        """
        Builds the decision tree from the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input features.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.tree = self._build_tree(X, y)
        return self

    def _predict_one(self, inputs, node):
        """Traverse the tree recursively for a single input sample."""
        if "feature_index" not in node:
            return node["class"]
        
        if inputs[node["feature_index"]] <= node["threshold"]:
            return self._predict_one(inputs, node["left"])
        else:
            return self._predict_one(inputs, node["right"])

    def predict(self, X):
        """
        Predicts class labels for a set of samples.

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
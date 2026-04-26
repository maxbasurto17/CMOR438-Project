import numpy as np

class DecisionTree:
    def __init__(self, max_depth=10):
        """
        Initializes the Decision Tree classifier.
        
        Parameters
        ----------
        max_depth : int
            The maximum depth of the tree to prevent overfitting.
        """
        self.max_depth = max_depth
        self.tree = None

    def _gini(self, y):
        """Calculate Gini Impurity for a list of labels."""
        m = len(y)
        if m == 0: return 0
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _best_split(self, X, y):
        """Find the best feature and threshold to split the data."""
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
                
                # Weighted average of Gini Impurity
                gini = (len(y_left) * self._gini(y_left) + 
                        len(y_right) * self._gini(y_right)) / m
                
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = thr
                    
        return best_idx, best_thr

    def _build_tree(self, X, y, depth=0):
        """Recursive function to build the tree nodes."""
        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
        predicted_class = np.argmax(num_samples_per_class)
        node = {"class": predicted_class}

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
        """Builds the decision tree from the training data."""
        self.tree = self._build_tree(X, y)
        return self

    def _predict_one(self, inputs, node):
        """Traverse the tree for a single input sample."""
        if "feature_index" not in node:
            return node["class"]
        if inputs[node["feature_index"]] <= node["threshold"]:
            return self._predict_one(inputs, node["left"])
        else:
            return self._predict_one(inputs, node["right"])

    def predict(self, X):
        """Predict classes for a set of samples."""
        return np.array([self._predict_one(inputs, self.tree) for inputs in X])
import numpy as np

class RegressionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _calculate_variance(self, y):
        if len(y) == 0:
            return 0
        return np.var(y)

    def _best_split(self, X, y):
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

                # Calculate weighted variance of children
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
        # Leaf node stores the AVERAGE value of targets
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
        self.tree = self._build_tree(X, y)
        return self

    def _predict_one(self, inputs, node):
        if "feature_index" not in node:
            return node["value"]
        if inputs[node["feature_index"]] <= node["threshold"]:
            return self._predict_one(inputs, node["left"])
        return self._predict_one(inputs, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(inputs, self.tree) for inputs in X])
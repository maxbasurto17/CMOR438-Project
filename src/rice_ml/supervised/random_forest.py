import numpy as np
from .decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, max_features=None):
        """
        Initializes the Random Forest classifier.

        A Random Forest is an ensemble of Decision Trees, trained with 
        bootstrapping and feature randomness to reduce variance.

        Parameters
        ----------
        n_estimators : int, default=10
            The number of trees in the forest.
        max_depth : int, default=10
            The maximum depth of each individual tree.
        min_samples_split : int, default=2
            The minimum samples required to split a node (passed to trees).
        max_features : int, optional
            The number of random features to consider when looking for the best 
            split at each node. If None, defaults to sqrt(n_features).
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        """
        Creates a random sample of the data with replacement (bootstrapping).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data features.
        y : ndarray of shape (n_samples,)
            Target labels.

        Returns
-------
        X_sample : ndarray
            The bootstrapped feature matrix.
        y_sample : ndarray
            The bootstrapped target vector.
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def train(self, X, y):
        """
        Builds the forest by training multiple decision trees on bootstrapped data.

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
        self.trees = []
        n_features = X.shape[1]

        # Default max_features to sqrt(n_features) if not specified
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        for _ in range(self.n_estimators):
            # Create bootstrap sample (Bagging)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # Create and train a tree with feature subsampling
            tree = DecisionTree(
                max_depth=self.max_depth, 
                max_features=self.max_features
            )
            tree.train(X_sample, y_sample)
            self.trees.append(tree)
        return self

    def predict(self, X):
        """
        Predicts class labels by taking a majority vote from all trees in the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples for which to predict labels.

        Returns
-------
        predictions : ndarray of shape (n_samples,)
            The predicted class labels based on majority consensus.
        """
        # Collect predictions from every tree in the ensemble
        # Resulting shape: (n_estimators, n_samples)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        predictions = []
        # Iterate through each sample to find the majority class
        for i in range(X.shape[0]):
            sample_preds = tree_preds[:, i]
            unique, counts = np.unique(sample_preds, return_counts=True)
            majority_class = unique[np.argmax(counts)]
            predictions.append(majority_class)
            
        return np.array(predictions)
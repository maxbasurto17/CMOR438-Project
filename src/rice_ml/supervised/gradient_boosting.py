import numpy as np
from .regression_tree import RegressionTree

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        """
        Initializes the Gradient Boosting Regressor.

        Gradient Boosting fits trees sequentially to the residuals of 
        previous predictions.

        Parameters
        ----------
        n_estimators : int, default=100
            The number of boosting stages to perform.
        learning_rate : float, default=0.1
            Shrinks the contribution of each tree by learning_rate.
        max_depth : int, default=3
            Maximum depth of the individual regression trees.
        min_samples_split : int, default=2
            The minimum samples required to split a node.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.initial_prediction = None

    def train(self, X, y):
        """
        Fits the gradient boosting model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
        """
        self.trees = []
        
        # Start with an initial constant prediction (the mean of y)
        self.initial_prediction = np.mean(y)
        current_predictions = np.full(len(y), self.initial_prediction)

        for _ in range(self.n_estimators):
            # Calculate residuals
            # In regression with MSE loss, the gradient is just (y - y_hat)
            residuals = y - current_predictions

            # Fit a regression tree to the residuals
            tree = RegressionTree(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split
            )
            tree.train(X, residuals)
            
            # Update current predictions
            # We add a small fraction (learning_rate) of the new tree's output
            predictions_from_tree = tree.predict(X)
            current_predictions += self.learning_rate * predictions_from_tree
            
            self.trees.append(tree)
            
        return self

    def predict(self, X):
        """
        Predicts regression value for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted values.
        """
        # Start with the initial constant prediction
        y_pred = np.full(len(X), self.initial_prediction)

        # Add the weighted contribution of each tree
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
            
        return y_pred
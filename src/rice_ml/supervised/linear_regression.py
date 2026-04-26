import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        """
        Initializes the Linear Regression model.
        
        Parameters
        ----------
        learning_rate : float, optional
            The step size for the gradient descent update (default is 0.01).
        epochs : int, optional
            The number of iterations over the entire training dataset (default is 1000).
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.cost_history = []

    def train(self, X, y):
        """
        Fits a linear regression model using batch gradient descent.
        
        Parameters
        ----------
        X : numpy.ndarray
            The input feature matrix of shape (n_samples, n_features).
        y : numpy.ndarray
            The target values (labels) of shape (n_samples,).
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.cost_history = []
        
        for epoch in range(self.epochs):
            # Compute the model's predictions: y = Xw + b
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute residuals
            error = y_pred - y
            
            # Compute gradients for weights and bias
            # Partial derivative of the Mean Squared Error with respect to weights
            dw = (1 / n_samples) * np.dot(X.T, error)
            # Partial derivative of the Mean Squared Error with respect to bias
            db = (1 / n_samples) * np.sum(error)
            
            # Update the weights and bias
            # Move in the opposite direction of the gradient
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute the cost (Mean Squared Error) and save it for tracking
            cost = (1 / (2 * n_samples)) * np.sum(error ** 2)
            self.cost_history.append(cost)
            
        return self

    def predict(self, X):
        """
        Calculates predictions for a trained linear regression model.
        
        Parameters
        ----------
        X : numpy.ndarray
            The input feature matrix of shape (n_samples, n_features).
            
        Returns
        -------
        y_pred : numpy.ndarray
            The predicted target values of shape (n_samples,).
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model must be trained before making predictions.")
            
        # y = Xw + b
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
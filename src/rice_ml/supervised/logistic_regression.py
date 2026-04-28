import numpy as np
from rice_ml.measures.validation import *

class LogisticRegression:
    def __init__(self, eta=0.01, epochs=1000):
        """
        Initializes the Logistic Regression model.
        
        Parameters:
        eta (float): Learning rate.
        epochs (int): Number of passes over the training dataset.
        """
        self.eta = eta
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []  # Tracks the loss over time for plotting/debugging

    def train(self, X, y):
        """
        Trains the model using gradient descent.
        
        Parameters:
        X (numpy.ndarray): Training data of shape (n_samples, n_features).
        y (numpy.ndarray): Target labels (0 or 1) of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent loop
        for _ in range(self.epochs):
            # Calculate predictions (Forward pass)
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.eta * dw
            self.bias -= self.eta * db

            # Calculate Binary Cross-Entropy Loss to track performance
            # Epsilon prevents log(0) errors
            epsilon = 1e-9
            y_pred_safe = np.clip(y_predicted, epsilon, 1 - epsilon)
            loss = - (1 / n_samples) * np.sum(y * np.log(y_pred_safe) + (1 - y) * np.log(1 - y_pred_safe))
            self.losses.append(loss)

    def predict_proba(self, X):
        """Returns the probability that the samples belong to class 1."""
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)

    def predict(self, X):
        """
        Predicts class labels for samples in X.
        Uses a standard threshold of 0.5.
        """
        probabilities = self.predict_proba(X)
        # Convert probabilities to 0 or 1
        predicted_classes = [1 if p >= 0.5 else 0 for p in probabilities]
        return np.array(predicted_classes)
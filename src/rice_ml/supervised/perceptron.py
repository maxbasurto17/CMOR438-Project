import numpy as np

class Perceptron:
    def __init__(self, eta=0.5, epochs=100):
        """
        Initializes the Perceptron classifier.

        Parameters
        ----------
        eta : float, default=0.5
            The learning rate, controlling the step size during weight updates.
        epochs : int, default=100
            The number of complete passes over the training dataset.
        """
        self.eta = eta
        self.epochs = epochs
        self.bias = 0
        self.weights = None
        self.errors = []

    def train(self, x, y):
        """
        Fits the Perceptron model to the training data.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target labels, typically 1 and -1.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.weights = np.zeros(len(x[0]))
        self.errors = []
        
        for i in range(self.epochs):
            error_count = 0
            for xi, target in zip(x, y):
                y_hat = self.predict(xi)
                self.weights = self.weights - self.eta * (y_hat - target) * xi
                self.bias = self.bias - self.eta * (y_hat - target)
                if y_hat != target:
                    error_count += 1
            self.errors.append(error_count)
                
    def predict(self, xi):
        """
        Predicts the class label for a single input sample.

        Parameters
        ----------
        xi : array-like of shape (n_features,)
            The input vector for which to predict the label.

        Returns
        -------
        label : int
            The predicted class label (1 or -1).
        """
        z = np.dot(xi, self.weights) + self.bias
        return np.where(z >= 0, 1, -1)
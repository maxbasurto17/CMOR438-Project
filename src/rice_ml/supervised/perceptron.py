import numpy as np

class Perceptron:
    # initialize function
    def __init__(self, eta=0.5, epochs=100):
        # learning rate
        self.eta = eta
        # epochs
        self.epochs = epochs
        # bias
        self.bias = 0

    # Train model
    def train(self, x, y):
        # initialize weights 
        self.weights = np.zeros(len(x[0]))
        # initialize errors
        self.errors = []
        
        # loop through for number of epochs
        for i in range(self.epochs):
            # reset errors for each epoch
            error_count = 0

            # then loop through each x,y instance
            for xi, target in zip(x, y):
                # make prediction
                y_hat = self.predict(xi)
                # update weights
                self.weights = self.weights - self.eta * (y_hat - target) * xi
                # update bias
                self.bias = self.bias - self.eta * (y_hat - target)

                # check for error
                if y_hat != target:
                    error_count += 1
            
            # appends error count after each epoch to evaluate training effectiveness over time
            self.errors.append(error_count)
                
    # Use to make predictions
    def predict(self, xi):
        z = np.dot(xi, self.weights) + self.bias
        return np.where(z >= 0, 1, -1)
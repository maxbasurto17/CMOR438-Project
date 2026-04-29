import numpy as np
from rice_ml.measures.validation import mean_squared_error, calculate_total_mse, sigmoid, d_sigmoid


class MultiLayerPerceptron(object):
    """
    A Multi-Layer Perceptron model.
    
    Parameters
    ----------
    layers : list of int, optional
        A list detailing the number of nodes in each layer, starting with the 
        input layer and ending with the output layer (default is [784, 60, 60, 10]).
        
    Attributes
    ----------
    layers : list of int
        The node counts for each layer in the network.
    W : list of numpy.ndarray
        Weight matrices for each layer, 1-indexed.
    B : list of numpy.ndarray
        Bias vectors for each layer, 1-indexed.
    errors_ : list of float
        The mean squared error evaluated after each epoch during training.
    """
    
    def __init__(self, layers=[784, 60, 60, 10]):
        self.layers = layers
        self.W, self.B = self._initialize_weights(self.layers)
        self.errors_ = []

    def _initialize_weights(self, layers):
        """
        Initializes the weights and biases using Gaussian distributions 
        scaled by a specific factor.
        
        Parameters
        ----------
        layers : list of int
            The node counts for each layer in the network.
            
        Returns
        -------
        tuple of (list, list)
            A tuple containing the initialized lists of weight matrices (W) 
            and bias vectors (B), padded at the 0th index with [[0.0]].
        """
        W = [[0.0]]
        B = [[0.0]]
        for i in range(1, len(layers)):
            w_temp = np.random.randn(layers[i], layers[i-1]) * np.sqrt(2 / layers[i-1])
            b_temp = np.random.randn(layers[i], 1) * np.sqrt(2 / layers[i-1])
        
            W.append(w_temp)
            B.append(b_temp)
        return W, B

    def _forward_pass(self, xi):
        """
        Performs the forward pass of the network for a single observation.
        
        Parameters
        ----------
        xi : numpy.ndarray
            The flattened input feature column vector.
            
        Returns
        -------
        tuple of (list, list)
            A tuple containing the pre-activation lists (Z) and 
            post-activation lists (A) for each layer.
        """
        # Ensure xi is a column vector
        xi_col = np.array(xi).reshape(-1, 1)

        Z = [[0.0]]
        A = [xi_col]
        L = len(self.W) - 1
        
        for i in range(1, L + 1):
            z = self.W[i] @ A[i-1] + self.B[i]
            Z.append(z)
            
            a = sigmoid(z)
            A.append(a)
            
        return Z, A

    def train(self, X_train, y_train, alpha=0.046, epochs=4):
        """
        Trains the neural network using stochastic gradient descent.
        
        Parameters
        ----------
        X_train : list of numpy.ndarray
            The training data, formatted as a list of column vectors.
        y_train : list of numpy.ndarray
            The training labels, formatted as a list of one-hot encoded column vectors.
        alpha : float, optional
            The learning rate (default is 0.046).
        epochs : int, optional
            The number of passes over the entire training dataset (default is 4).
            
        Returns
        -------
        None
        """
        # Print the initial mean squared error
        self.errors_ = [calculate_total_mse(self.W, self.B, X_train, y_train)]
        print(f"Starting Cost = {self.errors_[0]}")

        # Find the number of non-input layers.
        L = len(self.layers) - 1

        # For each epoch perform stochastic gradient descent. 
        for k in range(epochs):
            # Loop over each (xi, yi) training pair of data.
            for xi, yi in zip(X_train, y_train):
                
                # Forward pass to find the preactivation and postactivation values
                Z, A = self._forward_pass(xi)

                # Store the errors in a dictionary for clear interpretation
                deltas = dict()

                # Compute the output error 
                output_error = (A[L] - yi) * d_sigmoid(Z[L])
                deltas[L] = output_error

                # Loop from L-1 to 1 to compute node errors at each hidden layer
                for i in range(L-1, 0, -1):
                    deltas[i] = (self.W[i+1].T @ deltas[i+1]) * d_sigmoid(Z[i])

                # Loop over each hidden layer and the output layer to perform gradient descent
                for i in range(1, L+1):
                    self.W[i] -= alpha * deltas[i] @ A[i-1].T
                    self.B[i] -= alpha * deltas[i]

            # Show the user the cost over all training examples
            self.errors_.append(calculate_total_mse(self.W, self.B, X_train, y_train))   
            print(f"{k + 1}-Epoch Cost = {self.errors_[-1]}")

    def predict(self, xi):
        """
        Predicts the class label for a single given observation.
        
        Parameters
        ----------
        xi : numpy.ndarray
            The input feature column vector.
            
        Returns
        -------
        int
            The predicted class index (determined via argmax).
        """
        _, A = self._forward_pass(xi)
        return int(np.argmax(A[-1]))
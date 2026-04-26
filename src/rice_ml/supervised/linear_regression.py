import numpy as np

def linear_regression_gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    """
    Fits a linear regression model using batch gradient descent.
    
    Parameters
    ----------
    X : numpy.ndarray
        The input feature matrix of shape (n_samples, n_features).
    y : numpy.ndarray
        The target values (labels) of shape (n_samples,).
    learning_rate : float, optional
        The step size for the gradient descent update (default is 0.01).
    epochs : int, optional
        The number of iterations over the entire training dataset (default is 1000).
        
    Returns
    -------
    weights : numpy.ndarray
        The learned coefficients for the linear regression model, shape (n_features,).
    bias : float
        The learned intercept/bias term.
    cost_history : list of float
        The history of the cost function (Mean Squared Error) computed at each epoch. 
        Useful for plotting to ensure the algorithm converges.
    """
    # Get the number of samples (rows) and features (columns)
    n_samples, n_features = X.shape
    
    # Initialize weights and bias
    weights = np.zeros(n_features)
    bias = 0.0
    
    # Keep track of the cost at each epoch to monitor convergence
    cost_history = []
    
    for epoch in range(epochs):
        # Compute the model's predictions
        y_pred = np.dot(X, weights) + bias
        
        # Compute residuals
        error = y_pred - y
        
        # Compute gradients for weights and bias
        # Partial derivative of the Mean Squared Error with respect to weights
        dw = (1 / n_samples) * np.dot(X.T, error)
        # Partial derivative of the Mean Squared Error with respect to bias
        db = (1 / n_samples) * np.sum(error)
        
        # Update the weights and bias
        # Move in the opposite direction of the gradient
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db
        
        # Compute the cost (Mean Squared Error) and save it for tracking
        cost = (1 / (2 * n_samples)) * np.sum(error ** 2)
        cost_history.append(cost)
        
    return weights, bias, cost_history

def predict_linear_regression(X, weights, bias):
    """
    Calculates predictions for a trained linear regression model.
    
    Parameters
    ----------
    X : numpy.ndarray
        The input feature matrix of shape (n_samples, n_features).
    weights : numpy.ndarray
        The learned coefficients for the linear regression model, shape (n_features,).
    bias : float
        The learned intercept/bias term.
        
    Returns
    -------
    y_pred : numpy.ndarray
        The predicted target values of shape (n_samples,).
    """
    # y = Xw + b
    y_pred = np.dot(X, weights) + bias
    
    return y_pred
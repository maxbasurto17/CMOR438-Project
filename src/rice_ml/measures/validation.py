import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Calculates the Mean Squared Error (MSE) between true and predicted values.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        The actual target values.
    y_pred : numpy.ndarray
        The predicted target values.
        
    Returns
    -------
    float
        The mean squared error.
    """
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        The actual target values.
    y_pred : numpy.ndarray
        The predicted target values.
        
    Returns
    -------
    float
        The root mean squared error.
    """
    # Helper function: reusing the MSE function we already defined
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)

def mean_absolute_error(y_true, y_pred):
    """
    Calculates the Mean Absolute Error (MAE) between true and predicted values.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        The actual target values.
    y_pred : numpy.ndarray
        The predicted target values.
        
    Returns
    -------
    float
        The mean absolute error.
    """
    return np.mean(np.abs(y_true - y_pred))

def r_squared_score(y_true, y_pred):
    """
    Calculates the R-squared (Coefficient of Determination) score.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        The actual target values.
    y_pred : numpy.ndarray
        The predicted target values.
        
    Returns
    -------
    float
        The R-squared score. A score of 1.0 indicates perfect predictions.
        A score of 0.0 indicates the model performs no better than just 
        guessing the mean of the true values.
    """
    # Calculate the mean of the true values
    y_mean = np.mean(y_true)
    
    # Sum of Squares of Residuals (Errors)
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Total Sum of Squares (Variance of the data)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    
    # Handle the edge case where the variance of the data is 0
    if ss_tot == 0:
        return 0.0
        
    return 1 - (ss_res / ss_tot)

def accuracy_score(y_true, y_pred):
    """
    Calculates the Accuracy of the predictions.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        The actual target values (0 or 1).
    y_pred : numpy.ndarray
        The predicted target values (0 or 1).
        
    Returns
    -------
    float
        The ratio of correctly predicted observations to the total observations.
    """
    # np.mean on a boolean array converts True to 1 and False to 0, 
    # cleanly giving us the percentage of exact matches.
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred):
    """
    Calculates the Precision of the predictions.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        The actual target values (0 or 1).
    y_pred : numpy.ndarray
        The predicted target values (0 or 1).
        
    Returns
    -------
    float
        The precision score.
    """
    # True Positives: Model predicted 1, and it actually is 1
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    
    # False Positives: Model predicted 1, but it is actually 0
    false_positives = np.sum((y_pred == 1) & (y_true == 0))
    
    # Handle edge case where model never predicted 1 (prevents dividing by zero)
    if (true_positives + false_positives) == 0:
        return 0.0
        
    return true_positives / (true_positives + false_positives)

def recall_score(y_true, y_pred):
    """
    Calculates the Recall (Sensitivity) of the predictions.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        The actual target values (0 or 1).
    y_pred : numpy.ndarray
        The predicted target values (0 or 1).
        
    Returns
    -------
    float
        The recall score.
    """
    # True Positives: Model predicted 1, and it actually is 1
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    
    # False Negatives: Model predicted 0, but it is actually 1
    false_negatives = np.sum((y_pred == 0) & (y_true == 1))
    
    # Handle edge case where there are no actual 1s in the data
    if (true_positives + false_negatives) == 0:
        return 0.0
        
    return true_positives / (true_positives + false_negatives)

def f1_score(y_true, y_pred):
    """
    Calculates the F1-Score of the predictions.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        The actual target values (0 or 1).
    y_pred : numpy.ndarray
        The predicted target values (0 or 1).
        
    Returns
    -------
    float
        The F1-score, which is the harmonic mean of precision and recall.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    # Handle edge case to prevent division by zero
    if (precision + recall) == 0:
        return 0.0
        
    return 2 * (precision * recall) / (precision + recall)

def sigmoid(z):
    """
    Computes the sigmoid activation function.
    
    Parameters
    ----------
    z : numpy.ndarray
        The pre-activation linear combination vector.
        
    Returns
    -------
    numpy.ndarray
        The post-activation values.
    """
    return 1.0 / (1.0 + np.exp(-z))

def d_sigmoid(z):
    """
    Computes the derivative of the sigmoid activation function.
    
    Parameters
    ----------
    z : numpy.ndarray
        The pre-activation linear combination vector.
        
    Returns
    -------
    numpy.ndarray
        The derivative of the sigmoid function evaluated at z.
    """
    return sigmoid(z) * (1.0 - sigmoid(z))

def calculate_total_mse(W, B, X, y):
    """
    Calculates the Mean Squared Error over an entire dataset.
    Used as a cost function for training the Multi-Layer Perceptron.

    Parameters
    ----------
    W : list of numpy.ndarray
        The list of weight matrices.
    B : list of numpy.ndarray
        The list of bias vectors.
    X : list of numpy.ndarray
        The list of input feature column vectors.
    y : list of numpy.ndarray
        The list of one-hot encoded target vectors.
        
    Returns
    -------
    float
        The average mean squared error across all samples.
    """
    cost = 0.0
    m = len(X)
    for xi, yi in zip(X, y):
        # Forward pass to get predictions
        Z = [[0.0]]
        A = [xi]
        L = len(W) - 1
        for i in range(1, L + 1):
            z = W[i] @ A[i-1] + B[i]
            Z.append(z)
            a = sigmoid(z)
            A.append(a)
        
        # Using the mean_squared_error function
        cost += mean_squared_error(yi, A[-1])
        
    return cost / m if m > 0 else 0.0
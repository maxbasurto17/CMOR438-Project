import numpy as np

from rice_ml.supervised.linear_regression import linear_regression_gradient_descent, predict_linear_regression

def generate_dummy_data():
    """
    Helper function to generate a synthetic dataset where we know 
    the exact 'true' weights and bias.
    Equation: y = 3*x1 + 2*x2 + 1
    """
    np.random.seed(42)
    X = np.random.rand(100, 2) # 100 samples, 2 features
    
    true_weights = np.array([3.0, 2.0])
    true_bias = 1.0
    
    y = np.dot(X, true_weights) + true_bias
    
    return X, y, true_weights, true_bias

def test_predict_linear_regression():
    """
    Tests if the prediction function computes the dot product and addition correctly.
    """
    X_test = np.array([[1, 2], 
                       [3, 4]])
    weights = np.array([1, 1])
    bias = 0.5
    
    expected_predictions = np.array([3.5, 7.5])
    
    predictions = predict_linear_regression(X_test, weights, bias)
    
    # Assert arrays are equal up to a certain decimal point
    np.testing.assert_array_almost_equal(predictions, expected_predictions)

def test_gradient_descent_output_shapes():
    """
    Tests if the training function returns objects of the correct data types and shapes.
    """
    X, y, _, _ = generate_dummy_data()
    epochs = 50
    
    weights, bias, cost_history = linear_regression_gradient_descent(
        X, y, learning_rate=0.01, epochs=epochs
    )
    
    # Standard Python asserts with custom error messages
    assert weights.shape == (2,), f"Expected shape (2,), got {weights.shape}"
    assert isinstance(bias, float), f"Expected bias to be float, got {type(bias)}"
    assert len(cost_history) == epochs, f"Expected {epochs} costs, got {len(cost_history)}"

def test_gradient_descent_convergence():
    """
    Tests if the algorithm can successfully recover the hidden weights and bias 
    from the synthetic dataset.
    """
    X, y, true_weights, true_bias = generate_dummy_data()
    
    weights, bias, cost_history = linear_regression_gradient_descent(
        X, y, learning_rate=0.1, epochs=1000
    )
    
    # Check if the learned weights are very close to [3.0, 2.0]
    np.testing.assert_array_almost_equal(weights, true_weights, decimal=1)
    
    # Check if the learned bias is close to 1.0 using numpy.isclose
    assert np.isclose(bias, true_bias, atol=0.1), f"Expected bias ~{true_bias}, got {bias}"
    
    # Verify that gradient descent actually reduced the cost
    assert cost_history[0] > cost_history[-1], "Cost did not decrease over training!"
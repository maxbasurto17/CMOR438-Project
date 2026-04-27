import numpy as np
from rice_ml.supervised.multi_layer_perceptron import MultiLayerPerceptron
from rice_ml.measures.validation import *

def test_initialization():
    """
    Tests the MultiLayerPerceptron initialization for correct shapes 
    and non-zero initial values across multiple layers.
    
    Raises
    ------
    AssertionError
        If the weight/bias matrices have incorrect shapes or are initialized to zero.
    """
    np.random.seed(42)
    # Using 2 hidden layers [4, 5] to ensure dynamic loops work correctly
    layers = [3, 4, 5, 2]  
    net = MultiLayerPerceptron(layers=layers)
    
    # Check total lengths (accounting for index 0 padding)
    assert len(net.W) == len(layers), "Incorrect number of weight matrices."
    assert len(net.B) == len(layers), "Incorrect number of bias matrices."
    
    # Check shapes and non-zero initialization for every layer dynamically
    for i in range(1, len(layers)):
        expected_w_shape = (layers[i], layers[i-1])
        expected_b_shape = (layers[i], 1)
        
        # Shape checks
        assert net.W[i].shape == expected_w_shape, f"W[{i}] expected {expected_w_shape}, got {net.W[i].shape}"
        assert net.B[i].shape == expected_b_shape, f"B[{i}] expected {expected_b_shape}, got {net.B[i].shape}"
        
        # Non-zero checks
        assert np.any(net.W[i] != 0), f"Weights in layer {i} should not be all zeros."
        assert np.any(net.B[i] != 0), f"Biases in layer {i} should not be all zeros."


def test_forward_pass_and_predict():
    """
    Tests the _forward_pass and predict methods to ensure intermediate 
    matrices have expected dimensions and the final prediction is valid.
    
    Raises
    ------
    AssertionError
        If intermediate arrays have the incorrect shape or the prediction is invalid.
    """
    layers = [3, 4, 2]
    net = MultiLayerPerceptron(layers=layers)
    xi = np.array([[1.0], [2.0], [3.0]]) 
    
    # --- Test Forward Pass Shapes ---
    Z, A = net._forward_pass(xi)
    
    assert Z[0] == [0.0], "Z[0] should be initialized to [0.0]."
    assert A[0].shape == (3, 1), f"A[0] should be input shape (3, 1), got {A[0].shape}"
    
    for i in range(1, len(layers)):
        expected_shape = (layers[i], 1)
        assert Z[i].shape == expected_shape, f"Z[{i}] expected {expected_shape}, got {Z[i].shape}"
        assert A[i].shape == expected_shape, f"A[{i}] expected {expected_shape}, got {A[i].shape}"
        
    # --- Test Prediction Output ---
    prediction = net.predict(xi)
    
    assert isinstance(prediction, int), "Prediction must return an integer index."
    assert 0 <= prediction < layers[-1], "Prediction should be bounded by the output layer size."


def test_train_backpropagation():
    """
    Tests the train method to ensure backpropagation correctly calculates 
    gradients and reduces the loss over a few epochs.
    
    Raises
    ------
    AssertionError
        If the network loss does not strictly decrease over the training process.
    """
    np.random.seed(42)
    net = MultiLayerPerceptron(layers=[2, 4, 2])
    
    # Extremely simple toy dataset (linearly separable mapping)
    X_train = [np.array([[0.0], [0.0]]), np.array([[1.0], [1.0]])]
    y_train = [np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])]
    
    epochs = 10
    net.train(X_train, y_train, alpha=0.5, epochs=epochs)
    
    # Check that error array collected the right number of records (initial + 1 per epoch)
    assert len(net.errors_) == epochs + 1, "Errors list does not match the number of epochs + 1."
    
    # Check if loss decreased
    initial_loss = net.errors_[0]
    final_loss = net.errors_[-1]
    
    assert final_loss < initial_loss, f"Training failed to reduce loss. Started at {initial_loss}, ended at {final_loss}."
    assert net.errors_[1] < initial_loss, "Loss should immediately decrease after the first epoch."
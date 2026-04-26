import numpy as np
from rice_ml.measures.validation import *

def test_mean_squared_error():
    """
    Tests MSE calculation with a known manual example.
    """
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    
    # Manual check:
    # Errors: [0.5, -0.5, 0.0, -1.0]
    # Squared errors: [0.25, 0.25, 0.0, 1.0]
    # Mean of squared errors = 1.5 / 4 = 0.375
    expected_mse = 0.375
    
    mse = mean_squared_error(y_true, y_pred)
    assert np.isclose(mse, expected_mse), f"Expected {expected_mse}, got {mse}"

def test_root_mean_squared_error():
    """
    Tests RMSE calculation.
    """
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    
    # From previous test, MSE is 0.375
    expected_rmse = np.sqrt(0.375)
    
    rmse = root_mean_squared_error(y_true, y_pred)
    assert np.isclose(rmse, expected_rmse), f"Expected {expected_rmse}, got {rmse}"

def test_mean_absolute_error():
    """
    Tests MAE calculation with a known manual example.
    """
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    
    # Manual check:
    # Absolute Errors: [0.5, 0.5, 0.0, 1.0]
    # Mean of absolute errors = 2.0 / 4 = 0.5
    expected_mae = 0.5
    
    mae = mean_absolute_error(y_true, y_pred)
    assert np.isclose(mae, expected_mae), f"Expected {expected_mae}, got {mae}"

def test_r_squared_perfect_score():
    """
    Tests R-squared score when predictions are perfectly accurate.
    """
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    
    # Perfect predictions should yield an R2 score of 1.0
    r2 = r_squared_score(y_true, y_pred)
    assert np.isclose(r2, 1.0), f"Expected R2 of 1.0 for perfect predictions, got {r2}"

def test_r_squared_mean_guessing():
    """
    Tests R-squared score when the model just predicts the mean of the data.
    """
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    # The mean of y_true is 2.5
    y_pred = np.array([2.5, 2.5, 2.5, 2.5])
    
    # Predicting the mean should yield an R2 score of 0.0
    r2 = r_squared_score(y_true, y_pred)
    assert np.isclose(r2, 0.0), f"Expected R2 of 0.0 for predicting the mean, got {r2}"

def get_dummy_class_data():
    """
    Returns a consistent set of true and predicted labels for testing.
    """
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 0])
    return y_true, y_pred

def test_accuracy_score():
    """
    Tests Accuracy score function against the dummy data.
    """
    y_true, y_pred = get_dummy_class_data()
    
    expected_accuracy = 5 / 6
    
    acc = accuracy_score(y_true, y_pred)
    assert np.isclose(acc, expected_accuracy)

def test_precision_score():
    """
    Tests Precision score function against the dummy data.
    """
    y_true, y_pred = get_dummy_class_data()
    
    # We predicted '1' twice, and both times we were correct.
    # Precision = TP / (TP + FP) = 2 / (2 + 0) = 1.0
    expected_precision = 1.0
    
    prec = precision_score(y_true, y_pred)
    assert np.isclose(prec, expected_precision)

def test_recall_score():
    """
    Tests Recall score function against the dummy data.
    """
    y_true, y_pred = get_dummy_class_data()
    
    # There were three actual '1's, but we only found two of them.
    # Recall = TP / (TP + FN) = 2 / (2 + 1) = 2/3
    expected_recall = 2 / 3
    
    rec = recall_score(y_true, y_pred)
    assert np.isclose(rec, expected_recall)

def test_f1_score():
    """
    Tests F1 score function against the dummy data.
    """
    y_true, y_pred = get_dummy_class_data()
    
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    # F1 = 2 * (1.0 * (2/3)) / (1.0 + (2/3)) = (4/3) / (5/3) = 0.8
    expected_f1 = 0.8
    
    f1 = f1_score(y_true, y_pred)
    assert np.isclose(f1, expected_f1)

def test_zero_division_handling():
    """
    Tests edge cases where the denominator could be zero to ensure
    the functions don't crash.
    """
    # A model that only ever guesses 0
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    
    # Precision denominator (TP + FP) will be 0 here
    prec = precision_score(y_true, y_pred)
    assert prec == 0.0
    
    # Recall denominator (TP + FN) is safely > 0, but TP is 0, so recall is 0.
    # Therefore, F1 denominator (Precision + Recall) will be 0.
    f1 = f1_score(y_true, y_pred)
    assert f1 == 0.0
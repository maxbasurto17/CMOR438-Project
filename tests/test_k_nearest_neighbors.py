import numpy as np
from rice_ml.measures.distances import *
from rice_ml.supervised.k_nearest_neighbors import *

def test_get_neighbors():
    """
    Test the `_get_neighbors` method of the KNearestNeighbors class.
    """
    # Dummy training data with two classes (A and B)
    X_train = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    y_train = np.array(['A', 'A', 'B', 'B', 'A', 'B'])
    
    # Initialize KNN with k=3
    knn = KNearestNeighbors(k=3)
    
    # Test point that is closer to class 'A'
    test_point = np.array([1.2, 1.5])
    neighbors = knn._get_neighbors(X_train, y_train, test_point)
    
    assert len(neighbors) == 3, "Test Failed: Did not return exactly k neighbors."
    assert neighbors[0][0] <= neighbors[1][0] <= neighbors[2][0], "Test Failed: Neighbors are not sorted by distance."

def test_predict_point():
    """
    Test the `predict_point` method of the KNearestNeighbors class.
    """
    # Dummy training data with two classes (A and B)
    X_train = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    y_train = np.array(['A', 'A', 'B', 'B', 'A', 'B'])
    
    # Initialize KNN with k=3
    knn = KNearestNeighbors(k=3)
    
    # Test points for both classes
    test_point_A = np.array([1.2, 1.5]) # Closer to class 'A'
    test_point_B = np.array([7, 9])     # Closer to class 'B'
    
    assert knn.predict_point(X_train, y_train, test_point_A) == 'A', "Test Failed: Incorrect prediction for Class A."
    assert knn.predict_point(X_train, y_train, test_point_B) == 'B', "Test Failed: Incorrect prediction for Class B."

def test_predict():
    """
    Test the `predict` batch method of the KNearestNeighbors class.
    """
    # Dummy training data with two classes (A and B)
    X_train = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    y_train = np.array(['A', 'A', 'B', 'B', 'A', 'B'])
    
    # Initialize KNN with k=3
    knn = KNearestNeighbors(k=3)
    
    # Test multiple points at once
    X_test = np.array([[1.2, 1.5], [7, 9]])
    preds = knn.predict(X_train, y_train, X_test)
    
    assert np.array_equal(preds, np.array(['A', 'B'])), "Test Failed: Batch prediction is incorrect."
    
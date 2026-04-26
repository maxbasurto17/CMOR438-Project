import numpy as np
from rice_ml.measures.distances import *

def test_euclidean_distance():
    """
    Tests the Euclidean distance function with a known example.
    """
    point1 = np.array([1, 2])
    point2 = np.array([4, 6])
    
    # Manual calculation:
    # Distance = sqrt((4-1)^2 + (6-2)^2) = sqrt(3^2 + 4^2) = sqrt(25) = 5
    expected_distance = 5.0
    
    distance = euclidean_distance(point1, point2)
    assert np.isclose(distance, expected_distance), f"Expected {expected_distance}, got {distance}"

    
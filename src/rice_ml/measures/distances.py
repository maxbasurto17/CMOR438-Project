import numpy as np

def euclidean_distance(point1, point2):
    """
    Calculates the Euclidean distance between two numpy arrays.
    
    Parameters:
    point1 (numpy.ndarray): The first data point.
    point2 (numpy.ndarray): The second data point.
    
    Returns:
    float: The Euclidean distance between the two points.
    """
    # Using numpy's norm for efficient distance calculation
    return np.linalg.norm(point1 - point2)
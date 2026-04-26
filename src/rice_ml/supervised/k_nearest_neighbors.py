import numpy as np
from rice_ml.measures.distances import *

class KNearestNeighbors:
    """
    k-Nearest Neighbors (KNN) classifier.

    Parameters
    ----------
    k : int, default=3
        The number of nearest neighbors to consider for classification.

    Attributes
    ----------
    k : int
        The number of nearest neighbors utilized.
    """

    def __init__(self, k=3):
        """
        Initialize the KNN classifier.
        """
        self.k = k
        
    def _get_neighbors(self, X_train, y_train, target_point):
        """
        Find the k-nearest neighbors of a target point from the provided data.

        Parameters
        ----------
        X_train : numpy.ndarray
            The training data features.
        y_train : numpy.ndarray
            The target labels corresponding to `X_train`.
        target_point : numpy.ndarray
            A single data point to find the neighbors for.

        Returns
        -------
        list of tuple
            A list containing the `k` nearest neighbors. Each element is 
            a tuple formatted as `(distance, label)`.
        """
        distances = []
        
        # Calculate the distance between the target point and every point in X_train
        for i in range(len(X_train)):
            dist = euclidean_distance(target_point, X_train[i])
            distances.append((dist, y_train[i]))
            
        # Sort the distances list by the distance (first element of the tuple)
        distances.sort(key=lambda x: x[0])
        
        # Return the first k neighbors
        return distances[:self.k]
        
    def predict_point(self, X_train, y_train, target_point):
        """
        Predict the class label for a single target point.

        Parameters
        ----------
        X_train : numpy.ndarray
            The training data features.
        y_train : numpy.ndarray
            The target labels corresponding to `X_train`.
        target_point : numpy.ndarray
            The data point to predict the class for.

        Returns
        -------
        any
            The predicted label based on a majority vote among the `k` neighbors.
        """
        neighbors = self._get_neighbors(X_train, y_train, target_point)
        
        # Extract only the labels from the neighbors
        neighbor_labels = [neighbor[1] for neighbor in neighbors]
        
        # Count occurrences using a dictionary
        label_counts = {}
        for label in neighbor_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
                
        # Find the label with the maximum count
        predicted_label = max(label_counts, key=label_counts.get)
        
        return predicted_label
        
    def predict(self, X_train, y_train, X_test):
        """
        Predict the class labels for an array of target points.

        Parameters
        ----------
        X_train : numpy.ndarray
            The training data features.
        y_train : numpy.ndarray
            The target labels corresponding to `X_train`.
        X_test : numpy.ndarray
            The test data features, where each row represents a sample 
            to be predicted.

        Returns
        -------
        numpy.ndarray
            An array of predicted labels corresponding to each sample in `X_test`.
        """
        predictions = [self.predict_point(X_train, y_train, point) for point in X_test]
        return np.array(predictions)
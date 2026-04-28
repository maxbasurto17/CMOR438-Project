import numpy as np
from rice_ml.unsupervised.k_means import KMeans
from rice_ml.measures.distances import euclidean_distance

def test_assign_labels():
    """
    Tests the _assign_labels function to ensure points are matched 
    to the geometrically nearest cluster center.
    """
    kmeans = KMeans(k=2)
    X = np.array([[1.0, 1.0], [1.5, 1.5], [8.0, 8.0], [9.0, 9.0]])
    centers = np.array([[0.0, 0.0], [10.0, 10.0]])
    
    labels = kmeans._assign_labels(X, centers)
    
    # Points 1 and 2 should be closer to center 0 (0,0)
    # Points 3 and 4 should be closer to center 1 (10,10)
    expected_labels = np.array([0, 0, 1, 1])
    
    assert np.array_equal(labels, expected_labels), f"Expected {expected_labels}, got {labels}"

def test_update_centers():
    """
    Tests the _update_centers function to ensure new centers are 
    calculated strictly as the mean of points inside their assigned cluster.
    """
    kmeans = KMeans(k=2)
    X = np.array([[2.0, 2.0], [4.0, 4.0], [8.0, 8.0], [10.0, 10.0]])
    labels = np.array([0, 0, 1, 1])
    
    new_centers = kmeans._update_centers(X, labels)
    
    # The mean of [2, 2] and [4, 4] is [3, 3]
    # The mean of [8, 8] and [10, 10] is [9, 9]
    expected_centers = np.array([[3.0, 3.0], [9.0, 9.0]])
    
    assert np.allclose(new_centers, expected_centers), f"Expected {expected_centers}, got {new_centers}"

def test_fit_and_predict():
    """
    Tests the overall fit and predict pipeline to ensure the algorithm 
    converges and properly labels clear clusters.
    """
    kmeans = KMeans(k=2, max_iter=10)
    
    # Create two very distinct clusters
    cluster_1 = np.array([[1.0, 1.0], [1.1, 1.2], [0.9, 1.0]])
    cluster_2 = np.array([[10.0, 10.0], [10.2, 10.1], [9.9, 9.8]])
    X = np.vstack((cluster_1, cluster_2))
    
    kmeans.fit(X)
    
    # We don't know which cluster will be assigned 0 and 1, 
    # but we do know points in the same group must share the same label.
    preds = kmeans.predict(X)
    
    # First 3 points should have the same label
    assert preds[0] == preds[1] == preds[2], "Cluster 1 points do not share the same label."
    # Last 3 points should have the same label
    assert preds[3] == preds[4] == preds[5], "Cluster 2 points do not share the same label."
    # The two clusters should not have the same label
    assert preds[0] != preds[3], "Distinct clusters were assigned the same label."
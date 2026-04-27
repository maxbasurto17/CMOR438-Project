import unittest
import numpy as np
from rice_ml.unsupervised import DBSCAN

class TestDBSCAN(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create a dense cluster at (0,0)
        cluster1 = np.random.normal(0, 0.1, (20, 2))
        # Create a single outlier far away
        outlier = np.array([[10, 10]])
        self.X = np.vstack([cluster1, outlier])

    def test_noise_detection(self):
        """Ensure the distant point is labeled as noise (-1)."""
        db = DBSCAN(eps=0.5, min_samples=5)
        db.train(self.X)
        # The last point (outlier) should be -1
        self.assertEqual(db.labels_[-1], -1)

    def test_cluster_assignment(self):
        """Ensure the dense points are grouped in the same cluster."""
        db = DBSCAN(eps=0.5, min_samples=5)
        db.train(self.X)
        # Check that the first 20 points have the same non-negative label
        unique_labels = np.unique(db.labels_[:20])
        self.assertEqual(len(unique_labels), 1)
        self.assertGreaterEqual(unique_labels[0], 0)

if __name__ == "__main__":
    unittest.main()
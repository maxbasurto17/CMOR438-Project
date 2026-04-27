import unittest
import numpy as np
from rice_ml.unsupervised import PCA

class TestPCA(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create 3D data that mostly lies on a 2D plane
        x1 = np.random.normal(0, 10, 100)
        x2 = np.random.normal(0, 10, 100)
        x3 = 0.1 * x1 + 0.1 * x2 + np.random.normal(0, 0.1, 100) # x3 is mostly noise
        self.X = np.column_stack([x1, x2, x3])

    def test_projection_shape(self):
        """Ensure the output dimensionality is correct."""
        pca = PCA(n_components=2)
        X_transformed = pca.fit_transform(self.X)
        self.assertEqual(X_transformed.shape, (100, 2))

    def test_variance_order(self):
        """Verify that the first component explains more variance than the second."""
        pca = PCA(n_components=2)
        pca.train(self.X)
        self.assertGreater(pca.explained_variance[0], pca.explained_variance[1])

if __name__ == "__main__":
    unittest.main()
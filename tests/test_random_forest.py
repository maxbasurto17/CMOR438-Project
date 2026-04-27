import unittest
import numpy as np
from rice_ml.supervised import RandomForest

class TestRandomForest(unittest.TestCase):
    def setUp(self):
        """Generate a simple dataset for testing."""
        np.random.seed(42)
        # Create a dataset where if X[0] + X[1] > 1, class is 1, else 0
        self.X = np.random.rand(100, 2)
        self.y = (np.sum(self.X, axis=1) > 1.0).astype(int)

    def test_forest_improvement(self):
        """Check if the forest can learn a simple classification boundary."""
        rf = RandomForest(n_estimators=5, max_depth=3)
        rf.train(self.X, self.y)
        preds = rf.predict(self.X)
        accuracy = np.mean(preds == self.y)
        
        # A simple forest should easily get > 80% on this data
        self.assertGreater(accuracy, 0.8)

    def test_n_estimators(self):
        """Verify the correct number of trees are created."""
        n = 7
        rf = RandomForest(n_estimators=n)
        rf.train(self.X, self.y)
        self.assertEqual(len(rf.trees), n)

if __name__ == "__main__":
    unittest.main()
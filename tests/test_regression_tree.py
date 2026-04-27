import unittest
import numpy as np
from rice_ml.supervised import RegressionTree

class TestRegressionTree(unittest.TestCase):
    def setUp(self):
        """Prepare datasets for various regression scenarios."""
        np.random.seed(42)
        # 1. Step Function (The bread and butter of trees)
        self.X_step = np.arange(0, 10).reshape(-1, 1)
        self.y_step = np.array([0, 0, 0, 0, 0, 10, 10, 10, 10, 10])

        # 2. Multi-feature data with an irrelevant column
        # y only depends on column 0
        self.X_multi = np.array([
            [1, 500], [2, -100], [8, 1000], [9, 0]
        ])
        self.y_multi = np.array([5.0, 5.0, 20.0, 20.0])

    def test_step_function_fit(self):
        """Verify the tree identifies the split point in a jump correctly."""
        model = RegressionTree(max_depth=2)
        model.train(self.X_step, self.y_step)
        
        # Test values on both sides of the jump (at index 5)
        self.assertEqual(model.predict(np.array([[2]]))[0], 0.0)
        self.assertEqual(model.predict(np.array([[8]]))[0], 10.0)

    def test_variance_reduction(self):
        """Ensure tree ignores the noisy/irrelevant second column."""
        model = RegressionTree(max_depth=1)
        model.train(self.X_multi, self.y_multi)
        
        # If it split correctly on col 0, it should predict ~5 for small col 0 values
        test_val = np.array([[1.5, 999]]) # High noise in col 1
        prediction = model.predict(test_val)
        self.assertAlmostEqual(prediction[0], 5.0)

    def test_min_samples_split(self):
        """Verify the tree stops splitting when min_samples_split is reached."""
        # Force a tiny dataset
        X = np.array([[1], [2]])
        y = np.array([10, 20])
        
        # min_samples_split=5 means it should NOT split, just return the average
        model = RegressionTree(min_samples_split=5)
        model.train(X, y)
        
        preds = model.predict(X)
        self.assertEqual(preds[0], 15.0) # Average of 10 and 20

    def test_high_depth_overfit(self):
        """Ensure a high-depth tree can perfectly fit a small dataset."""
        X = np.linspace(0, 1, 5).reshape(-1, 1)
        y = np.array([1.2, 2.5, 0.8, 4.2, 3.1])
        
        model = RegressionTree(max_depth=10)
        model.train(X, y)
        preds = model.predict(X)
        
        np.testing.assert_array_almost_equal(preds, y, decimal=5)

if __name__ == "__main__":
    unittest.main()
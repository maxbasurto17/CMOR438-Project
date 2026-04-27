import unittest
import numpy as np
from rice_ml.supervised import GradientBoostingRegressor

class TestGradientBoosting(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.linspace(0, 6, 100).reshape(-1, 1)
        self.y = np.sin(self.X).ravel() + np.random.normal(0, 0.1, 100)

    def test_improvement_over_baseline(self):
        """Check if GB predicts better than just using the mean."""
        model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1)
        model.train(self.X, self.y)
        preds = model.predict(self.X)
        
        mse_model = np.mean((self.y - preds)**2)
        mse_baseline = np.mean((self.y - np.mean(self.y))**2)
        
        self.assertLess(mse_model, mse_baseline)

    def test_n_estimators_tracking(self):
        """Verify the model stores the correct number of trees."""
        n = 15
        model = GradientBoostingRegressor(n_estimators=n)
        model.train(self.X, self.y)
        self.assertEqual(len(model.trees), n)

if __name__ == "__main__":
    unittest.main()
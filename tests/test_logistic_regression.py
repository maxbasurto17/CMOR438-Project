import unittest
import numpy as np
from rice_ml.supervised import LogisticRegression

class TestLogisticRegression(unittest.TestCase):
    # Note: This test suite focuses on basic functionality and convergence of the Logistic Regression implementation.
    def setUp(self):
        """Prepare a simple dataset for testing."""
        self.model = LogisticRegression(eta=0.1, epochs=100)
        # 4 samples, 2 features
        self.X = np.array([[1, 1], [2, 1], [1, 0], [0, 0]])
        self.y = np.array([1, 1, 0, 0])

    # Test cases
    def test_initialization(self):
        """Check if hyperparameters are set correctly."""
        self.assertEqual(self.model.eta, 0.1)
        self.assertEqual(self.model.epochs, 100)
        self.assertIsNone(self.model.weights)

    # This test checks if the training process correctly updates weights and bias, and if the loss decreases over time.
    def test_training(self):
        """Verify weights are created after training."""
        self.model.train(self.X, self.y)
        self.assertEqual(self.model.weights.shape, (2,))
        self.assertGreater(len(self.model.losses), 0)
        # Loss should decrease (or at least stay low)
        self.assertLess(self.model.losses[-1], self.model.losses[0])

    # This test checks if the predict method returns binary class labels and has the correct shape.
    def test_predictions(self):
        """Verify output shapes and values."""
        self.model.train(self.X, self.y)
        preds = self.model.predict(self.X)
        self.assertEqual(preds.shape, (4,))
        # Check that predictions are only 0s and 1s
        self.assertTrue(np.all((preds == 0) | (preds == 1)))

if __name__ == '__main__':
    unittest.main()
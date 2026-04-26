import sys
import os

# This forces Python to look inside the 'src' folder directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
import numpy as np
from rice_ml.supervised.perceptron import Perceptron

class TestPerceptron(unittest.TestCase):
    def setUp(self):
        # We use a simple linearly separable dataset (AND gate logic)
        # Mapping: 0,0 -> -1 | 0,1 -> -1 | 1,0 -> -1 | 1,1 -> 1
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([-1, -1, -1, 1])
        
        # Initialize a fresh model before each test
        self.model = Perceptron(eta=0.1, epochs=10)

    def test_initialization(self):
        """Test if the model initializes with correct default and custom parameters."""
        self.assertEqual(self.model.eta, 0.1)
        self.assertEqual(self.model.epochs, 10)
        self.assertEqual(self.model.bias, 0)

    def test_training_updates_attributes(self):
        """Test if weights and errors arrays are generated during training."""
        self.model.train(self.X, self.y)
        
        self.assertTrue(hasattr(self.model, 'weights'))
        self.assertTrue(hasattr(self.model, 'errors'))
        self.assertEqual(len(self.model.weights), 2)  # 2 features in X
        self.assertEqual(len(self.model.errors), 10)  # 10 epochs

    def test_prediction_accuracy(self):
        """Test if the model successfully learns to predict linearly separable data."""
        # Train a model with enough epochs to guarantee convergence
        test_model = Perceptron(eta=0.1, epochs=100)
        test_model.train(self.X, self.y)
        
        # Predict on the same training data
        predictions = [test_model.predict(xi) for xi in self.X]
        
        # Check if the predictions perfectly match the target array
        np.testing.assert_array_equal(predictions, self.y)

if __name__ == '__main__':
    unittest.main()
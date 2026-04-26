import unittest
import numpy as np
from rice_ml.supervised import LinearRegression

class TestLinearRegression(unittest.TestCase):
    
    def setUp(self):
        """
        Helper to generate a synthetic dataset where we know 
        the exact 'true' weights and bias.
        Equation: y = 3*x1 + 2*x2 + 1
        """
        np.random.seed(42)
        self.X = np.random.rand(100, 2)  # 100 samples, 2 features
        
        self.true_weights = np.array([3.0, 2.0])
        self.true_bias = 1.0
        
        self.y = np.dot(self.X, self.true_weights) + self.true_bias

    def test_predict_linear_regression(self):
        """
        Tests if the prediction method computes the dot product correctly.
        """
        X_test = np.array([[1, 2], [3, 4]])
        model = LinearRegression()
        
        # Manually set weights for testing prediction logic
        model.weights = np.array([1.0, 1.0])
        model.bias = 0.5
        
        expected_predictions = np.array([3.5, 7.5])
        predictions = model.predict(X_test)
        
        np.testing.assert_array_almost_equal(predictions, expected_predictions)

    def test_train_output_shapes(self):
        """
        Tests if the training method sets attributes of the correct shapes.
        """
        epochs = 50
        model = LinearRegression(learning_rate=0.01, epochs=epochs)
        model.train(self.X, self.y)
        
        self.assertEqual(model.weights.shape, (2,))
        self.assertIsInstance(model.bias, (float, np.float64))
        self.assertEqual(len(model.cost_history), epochs)

    def test_gradient_descent_convergence(self):
        """
        Tests if the class can recover the hidden weights and bias.
        """
        model = LinearRegression(learning_rate=0.1, epochs=1000)
        model.train(self.X, self.y)
        
        # Check learned parameters
        np.testing.assert_array_almost_equal(model.weights, self.true_weights, decimal=1)
        self.assertTrue(np.isclose(model.bias, self.true_bias, atol=0.1))
        
        # Verify that cost actually reduced
        self.assertGreater(model.cost_history[0], model.cost_history[-1])

if __name__ == '__main__':
    unittest.main()
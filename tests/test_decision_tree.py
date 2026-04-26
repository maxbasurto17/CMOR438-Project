import unittest
import numpy as np
from rice_ml.supervised import DecisionTree

class TestDecisionTree(unittest.TestCase):
    def setUp(self):
        """Prepare datasets for various testing scenarios."""
        # 1. XOR-like dataset (Tests if the tree can handle non-linear splits)
        self.X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y_xor = np.array([0, 1, 1, 0])

        # 2. Redundant Feature dataset (Tests if tree ignores useless features)
        self.X_red = np.array([[1, 100], [1, 200], [0, 5], [0, -10]])
        self.y_red = np.array([1, 1, 0, 0]) # Only column 0 matters

    def test_perfect_separation(self):
        """The tree should achieve 100% accuracy on a simple separable dataset."""
        model = DecisionTree(max_depth=3)
        model.train(self.X_red, self.y_red)
        preds = model.predict(self.X_red)
        np.testing.assert_array_equal(preds, self.y_red, 
                                     err_msg="Tree failed to perfectly separate simple data.")

    def test_max_depth_constraint(self):
        """Verify that the tree actually respects the max_depth limit."""
        # With depth 1, the tree can't solve XOR.
        model = DecisionTree(max_depth=1)
        model.train(self.X_xor, self.y_xor)
        
        # A depth-1 tree on XOR usually just predicts the majority class for everything
        preds = model.predict(self.X_xor)
        # It shouldn't be able to get 4/4 correct with only one split
        correct_count = np.sum(preds == self.y_xor)
        self.assertLessEqual(correct_count, 2, "Depth-1 tree somehow solved XOR; check split logic.")

    def test_single_class_input(self):
        """Test how the tree handles data where everyone is the same class."""
        X = np.random.rand(10, 2)
        y = np.array([1] * 10)
        model = DecisionTree()
        model.train(X, y)
        preds = model.predict(np.array([[0.5, 0.5]]))
        self.assertEqual(preds[0], 1, "Tree failed on single-class input.")

    def test_high_dimensionality(self):
        """Ensure the tree doesn't crash with many features."""
        X = np.random.rand(20, 50) # 50 features
        y = np.random.randint(0, 2, 20)
        model = DecisionTree(max_depth=2)
        try:
            model.train(X, y)
        except Exception as e:
            self.fail(f"Training failed on high-dimensional data: {e}")

if __name__ == "__main__":
    unittest.main()
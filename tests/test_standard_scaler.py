import unittest
import numpy as np
from rice_ml.preprocessing import StandardScaler

class TestStandardScaler(unittest.TestCase):
    def test_basic_scaling(self):
        """Test if data is centered to 0 and scaled to unit variance."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Check mean is ~0 and std is ~1
        np.testing.assert_array_almost_equal(np.mean(X_scaled, axis=0), [0, 0])
        np.testing.assert_array_almost_equal(np.std(X_scaled, axis=0), [1, 1])

    def test_constant_feature(self):
        """Test handling of features with zero variance."""
        X = np.array([[1, 5], [2, 5], [3, 5]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Second column should remain 0 and not crash due to division by zero
        self.assertTrue(np.all(X_scaled[:, 1] == 0))

if __name__ == "__main__":
    unittest.main()
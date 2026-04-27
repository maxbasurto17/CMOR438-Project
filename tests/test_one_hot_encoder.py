import unittest
import numpy as np
from rice_ml.preprocessing import OneHotEncoder

class TestOneHotEncoder(unittest.TestCase):
    def test_encoding_logic(self):
        """Test basic one-hot transformation."""
        y = np.array(['cat', 'dog', 'cat', 'bird'])
        encoder = OneHotEncoder()
        y_encoded = encoder.fit_transform(y)
        
        # 4 samples, 3 unique categories (bird, cat, dog)
        self.assertEqual(y_encoded.shape, (4, 3))
        # Each row must have exactly one '1'
        self.assertTrue(np.all(np.sum(y_encoded, axis=1) == 1))

    def test_unseen_category(self):
        """Verify behavior when transforming a category not seen in fit."""
        encoder = OneHotEncoder()
        encoder.fit(['a', 'b'])
        # 'c' is unseen, row should be all zeros
        y_encoded = encoder.transform(['a', 'c'])
        np.testing.assert_array_equal(y_encoded[1], [0, 0])

if __name__ == "__main__":
    unittest.main()
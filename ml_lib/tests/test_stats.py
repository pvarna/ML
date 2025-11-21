import unittest
import numpy as np
from stats import sigmoid, softmax, gini_index, entropy


class TestSigmoid(unittest.TestCase):

    def test_when_scalar_then_correct_value(self):
        # Arrange
        x = 0
        expected = 0.5

        # Act
        actual = sigmoid(x)

        # Assert
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_positive_value_then_between_half_and_one(self):
        # Arrange
        x = 2.0

        # Act
        actual = sigmoid(x)

        # Assert
        self.assertGreater(actual, 0.5)
        self.assertLess(actual, 1.0)

    def test_when_negative_value_then_between_zero_and_half(self):
        # Arrange
        x = -3.0

        # Act
        actual = sigmoid(x)

        # Assert
        self.assertGreater(actual, 0.0)
        self.assertLess(actual, 0.5)

    def test_when_matrix_input_then_shape_preserved(self):
        # Arrange
        x = np.array([[0.0, 1.0], [2.0, -1.0]])

        # Act
        actual = sigmoid(x)

        # Assert
        self.assertEqual(actual.shape, x.shape)


class TestSoftmax(unittest.TestCase):

    def test_when_single_row_then_sums_to_one(self):
        # Arrange
        x = np.array([[1.0, 2.0, 3.0]])

        # Act
        actual = softmax(x)
        row_sum = np.sum(actual, axis=1)

        # Assert
        self.assertAlmostEqual(row_sum[0], 1.0, places=7)

    def test_when_multiple_rows_then_each_row_sums_to_one(self):
        # Arrange
        x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])

        # Act
        actual = softmax(x)
        row_sums = np.sum(actual, axis=1)

        # Assert
        for s in row_sums:
            self.assertAlmostEqual(s, 1.0, places=7)

    def test_when_matrix_input_then_shape_preserved(self):
        # Arrange
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Act
        actual = softmax(x)

        # Assert
        self.assertEqual(actual.shape, x.shape)

    def test_when_all_equal_values_then_uniform_distribution(self):
        # Arrange
        x = np.array([[5.0, 5.0, 5.0]])
        expected = np.array([[1 / 3, 1 / 3, 1 / 3]])

        # Act
        actual = softmax(x)

        # Assert
        np.testing.assert_almost_equal(actual, expected, decimal=7)


class TestGiniIndex(unittest.TestCase):

    def test_when_pure_distribution_then_zero(self):
        # Arrange
        ps = np.array([1.0, 0.0])
        expected = 0.0

        # Act
        actual = gini_index(ps)

        # Assert
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_uniform_two_classes_then_half(self):
        # Arrange
        ps = np.array([0.5, 0.5])
        expected = 0.5

        # Act
        actual = gini_index(ps)

        # Assert
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_mixed_three_classes(self):
        # Arrange
        ps = np.array([0.2, 0.3, 0.5])
        expected = 1 - (0.2**2 + 0.3**2 + 0.5**2)

        # Act
        actual = gini_index(ps)

        # Assert
        self.assertAlmostEqual(actual, expected, places=7)


class TestEntropy(unittest.TestCase):

    def test_when_pure_distribution_then_zero(self):
        # Arrange
        ps = np.array([1.0, 0.0])
        expected = 0.0

        # Act
        try:
            actual = entropy(ps)
        except Exception as e:
            self.fail(
                f"entropy shouldn't raise an exception for zero probability: {e}"
            )

        # Assert
        self.assertTrue(np.isfinite(actual))
        self.assertAlmostEqual(actual, expected, places=4)

    def test_when_uniform_two_classes_then_one(self):
        # Arrange
        ps = np.array([0.5, 0.5])
        expected = 1.0

        # Act
        actual = entropy(ps)

        # Assert
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_uniform_four_classes_then_two(self):
        # Arrange
        ps = np.array([0.25, 0.25, 0.25, 0.25])
        expected = 2.0

        # Act
        actual = entropy(ps)

        # Assert
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_mixed_three_classes(self):
        # Arrange
        ps = np.array([0.2, 0.3, 0.5])
        expected = (-0.2 * np.log2(0.2) - 0.3 * np.log2(0.3) -
                    0.5 * np.log2(0.5))

        # Act
        actual = entropy(ps)

        # Assert
        self.assertAlmostEqual(actual, expected, places=7)

import unittest
import numpy as np
import pandas as pd

from metrics import accuracy_score
from metrics import euclidean_distance
from metrics import manhattan_distance
from metrics import r2_score
from metrics import root_mean_squared_error


class TestAccuracyScore(unittest.TestCase):

    def test_when_sizes_differ_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([1, 0, 1])
        y_pred = pd.Series([1, 0])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = accuracy_score(y_true, y_pred)

    def test_when_normalize_true_then_returns_fraction(self):
        # Arrange
        y_true = pd.Series([1, 0, 1, 1])
        y_pred = pd.Series([1, 1, 1, 0])
        expected = 0.5  # 2 / 4

        # Act
        actual = accuracy_score(y_true, y_pred, normalize=True)

        # Assert
        self.assertEqual(actual, expected)

    def test_when_normalize_false_then_returns_count(self):
        # Arrange
        y_true = pd.Series([1, 0, 1])
        y_pred = pd.Series([0, 0, 1])
        expected = 2

        # Act
        actual = accuracy_score(y_true, y_pred, normalize=False)

        # Assert
        self.assertEqual(actual, expected)


class TestEuclideanDistance(unittest.TestCase):

    def test_when_points_have_different_dimensions_then_throws_runtime_error(
            self):
        # Arrange
        x = np.array([1.0, 2.0])
        y = np.array([1.0])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = euclidean_distance(x, y)

    def test_when_known_points_then_correct_distance(self):
        # Arrange
        x = np.array([0.0, 0.0])
        y = np.array([3.0, 4.0])
        expected = 5.0

        # Act
        actual = euclidean_distance(x, y)

        # Assert
        self.assertAlmostEqual(actual, expected, places=7)


class TestManhattanDistance(unittest.TestCase):

    def test_when_points_have_different_dimensions_then_throws_runtime_error(
            self):
        # Arrange
        x = np.array([1.0, 2.0])
        y = np.array([1.0])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = manhattan_distance(x, y)

    def test_when_known_points_then_correct_distance(self):
        # Arrange
        x = np.array([1.0, -1.0])
        y = np.array([4.0, 5.0])
        expected = 9.0  # |1-4| + |-1-5| = 3 + 6

        # Act
        actual = manhattan_distance(x, y)

        # Assert
        self.assertEqual(actual, expected)


class TestR2Score(unittest.TestCase):

    def test_when_sizes_differ_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = pd.Series([1.0, 2.0])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = r2_score(y_true, y_pred)

    def test_when_perfect_prediction_then_returns_one(self):
        # Arrange
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        expected = 1.0

        # Act
        actual = r2_score(y_true, y_pred)

        # Assert
        self.assertEqual(actual, expected)

    def test_when_constant_y_true_then_returns_zero(self):
        # Arrange
        y_true = [5.0, 5.0, 5.0, 5.0] 
        y_pred = [4.0, 6.0, 5.0, 5.0]
        expected = 0.0

        # Act
        actual = r2_score(y_true, y_pred)

        # Assert
        self.assertEqual(actual, expected)

    def test_when_known_values_then_correct_score(self):
        # Arrange
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 2.0]
        expected = 0.5

        # Act
        actual = r2_score(y_true, y_pred)

        # Assert
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_bad_model_then_negative_r2(self):
        # Arrange
        y_true = [0.0, 1.0]
        y_pred = [10.0, 10.0]

        # Act
        actual = r2_score(y_true, y_pred)

        # Assert
        self.assertLess(actual, 0.0)


class TestRootMeanSquaredError(unittest.TestCase):

    def test_when_sizes_differ_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = pd.Series([1.0, 2.0])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = root_mean_squared_error(y_true, y_pred)

    def test_when_perfect_prediction_then_zero(self):
        # Arrange
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        expected = 0.0

        # Act
        actual = root_mean_squared_error(y_true, y_pred)

        # Assert
        self.assertEqual(actual, expected)

    def test_when_known_values_then_correct_rmse(self):
        # Arrange
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 4.0, 2.0]
        expected = 1.290994449

        # Act
        actual = root_mean_squared_error(y_true, y_pred)

        # Assert
        self.assertAlmostEqual(actual, expected, places=7)


if __name__ == "__main__":
    unittest.main()

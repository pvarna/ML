import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from neighbors import KNeighborsClassifier


class TestKNeighborsClassifierInit(unittest.TestCase):

    def test_when_n_neighbors_is_non_positive_then_throws_runtime_error(self):
        # Arrange & Act & Assert
        with self.assertRaises(RuntimeError):
            _ = KNeighborsClassifier(n_neighbors=0, metric="euclidean")

    def test_when_metric_unknown_then_throws_runtime_error(self):
        # Arrange & Act & Assert
        with self.assertRaises(RuntimeError):
            _ = KNeighborsClassifier(n_neighbors=3, metric="pesho")


class TestKNeighborsClassifierFitPredict(unittest.TestCase):

    def _simple_dataset(self):
        X = pd.DataFrame({
            "x1": [1, 0, 2, 3, 4, 4],
            "x2": [1, 5, 5, 2, 1, 2],
        })
        y = pd.Series([0, 0, 0, 1, 1, 1], name="y")
        return X, y

    def test_when_fit_with_length_mismatch_then_throws_runtime_error(self):
        # Arrange
        X = pd.DataFrame({"x1": [0, 1, 2]})
        y = pd.Series([0, 1])
        knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = knn.fit(X, y)

    def test_when_predict_before_fit_then_throws_runtime_error(self):
        # Arrange
        X, _ = self._simple_dataset()
        knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = knn.predict(X)

    def test_when_k_is_1_and_euclidean_then_points_are_classified_correctly(
            self):
        # Arrange
        X, y = self._simple_dataset()
        knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean").fit(X, y)
        X_test = pd.DataFrame({"x1": [2, 3], "x2": [1, 1]})
        expected = pd.Series([0, 1], index=X_test.index)

        # Act
        actual = knn.predict(X_test)

        # Assert
        self.assertTrue(actual.equals(expected))

    def test_when_k_is_3_then_majority_vote_is_used(self):
        # Arrange
        X, y = self._simple_dataset()
        knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean").fit(X, y)
        X_test = pd.DataFrame({"x1": [2, 3], "x2": [1, 1]})
        expected = pd.Series([1, 1], index=X_test.index)

        # Act
        actual = knn.predict(X_test)

        # Assert
        self.assertTrue(actual.equals(expected))

    def test_when_scoring_then_returns_accuracy_between_0_and_1(self):
        # Arrange
        X, y = self._simple_dataset()
        knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean").fit(X, y)
        X_test = pd.DataFrame({"x1": [0, 10, 0, 10], "x2": [0, 10, 1, 11]})
        y_test = pd.Series([0, 1, 0, 1])

        # Act
        score = knn.score(X_test, y_test)

        # Assert
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_when_metric_is_euclidean_then_euclidean_distance_is_called(self):
        # Arrange
        X, y = self._simple_dataset()
        X_test = pd.DataFrame({"x1": [0.5], "x2": [0.5]})
        knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean").fit(X, y)

        with patch("neighbors.euclidean_distance", return_value=0.0) as mock_euclidean, \
             patch("neighbors.manhattan_distance", return_value=0.0) as mock_manhattan:

            # Act
            _ = knn.predict(X_test)

            # Assert
            mock_euclidean.assert_called()
            mock_manhattan.assert_not_called()

    def test_when_metric_is_manhattan_then_manhattan_distance_is_called(self):
        # Arrange
        X, y = self._simple_dataset()
        X_test = pd.DataFrame({"x1": [0.5], "x2": [0.5]})
        knn = KNeighborsClassifier(n_neighbors=1, metric="manhattan").fit(X, y)

        with patch("neighbors.euclidean_distance", return_value=0.0) as mock_euclidean, \
             patch("neighbors.manhattan_distance", return_value=0.0) as mock_manhattan:

            # Act
            _ = knn.predict(X_test)

            # Assert
            mock_euclidean.assert_not_called()
            mock_manhattan.assert_called()

if __name__ == "__main__":
    unittest.main()
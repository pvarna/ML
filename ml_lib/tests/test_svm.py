import unittest
import numpy as np
import pandas as pd

from svm import SVC


class TestSVCInit(unittest.TestCase):

    def test_when_kernel_unknown_then_throws_runtime_error(self):
        # Arrange & Act & Assert
        with self.assertRaises(RuntimeError):
            _ = SVC(C=1.0, kernel="unknown")


class TestSVCFitPredict(unittest.TestCase):

    def _simple_linearly_separable_dataset(self):
        X = pd.DataFrame({"x": [-2.0, -1.5, -1.0, 1.0, 1.5, 2.0]})
        y = pd.Series([0, 0, 0, 1, 1, 1])

        return X, y

    def test_when_predict_before_fit_then_throws_runtime_error(self):
        # Arrange
        X, _ = self._simple_linearly_separable_dataset()
        svc = SVC(C=1.0, kernel="linear")

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = svc.predict(X)

    def test_when_more_than_two_classes_then_throws_runtime_error(self):
        # Arrange
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
        y = pd.Series([0, 1, 2])
        svc = SVC(C=1.0, kernel="linear")

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = svc.fit(X, y)

    def test_when_linearly_separable_linear_kernel_then_correct_classification(
            self):
        # Arrange
        X, y = self._simple_linearly_separable_dataset()
        svc = SVC(C=10.0, kernel="linear")

        # Act
        svc.fit(X, y)
        predictions = svc.predict(X)

        # Assert
        pd.testing.assert_series_equal(predictions, y, check_dtype=False)
        self.assertEqual(set(svc.classes_), {0, 1})
        self.assertIsNotNone(svc.support_vectors_)
        self.assertIsNotNone(svc.intercept_)
        self.assertGreater(len(svc.support_vectors_), 0)

    def test_when_rbf_kernel_on_separable_data_then_correct_classification(
            self):
        # Arrange
        X, y = self._simple_linearly_separable_dataset()
        svc = SVC(C=10.0, kernel="rbf", gamma=1.0)

        # Act
        svc.fit(X, y)
        predictions = svc.predict(X)

        # Assert
        pd.testing.assert_series_equal(predictions, y, check_dtype=False)
        self.assertEqual(set(svc.classes_), {0, 1})
        self.assertGreater(len(svc.support_vectors_), 0)
        self.assertIsNotNone(svc.intercept_)

    def test_when_polynomial_kernel_then_correct_classification(self):
        # Arrange
        X, y = self._simple_linearly_separable_dataset()
        svc = SVC(C=10.0, kernel="polynomial", degree=3, coef0=1.0)

        # Act
        svc.fit(X, y)
        predictions = svc.predict(X)

        # Assert
        pd.testing.assert_series_equal(predictions, y, check_dtype=False)
        self.assertEqual(set(svc.classes_), {0, 1})
        self.assertGreater(len(svc.support_vectors_), 0)
        self.assertIsNotNone(svc.intercept_)

    def test_when_sigmoid_kernel_then_correct_classification(self):
        # Arrange
        X, y = self._simple_linearly_separable_dataset()
        svc = SVC(C=10.0, kernel="sigmoid", gamma=0.5, coef0=0.0)

        # Act
        svc.fit(X, y)
        predictions = svc.predict(X)

        # Assert
        pd.testing.assert_series_equal(predictions, y, check_dtype=False)
        self.assertEqual(set(svc.classes_), {0, 1})
        self.assertGreater(len(svc.support_vectors_), 0)
        self.assertIsNotNone(svc.intercept_)


if __name__ == "__main__":
    unittest.main()

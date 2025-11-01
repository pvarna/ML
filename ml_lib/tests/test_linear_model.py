import unittest
import numpy as np
import pandas as pd

from linear_model import LinearRegression


class TestLinearRegressionInitFitPredict(unittest.TestCase):

    def test_when_predict_before_fit_then_throws_runtime_error(self):
        # Arrange
        X = pd.DataFrame({"x": [0.0, 1.0]})

        # Act
        lr = LinearRegression()

        # Assert
        with self.assertRaises(RuntimeError):
            _ = lr.predict(X)

    def test_when_collinear_features_then_throws_value_error(self):
        X = pd.DataFrame({
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [2.0, 4.0, 6.0, 8.0],
        })
        y = pd.Series([3.0, 5.0, 7.0, 9.0])

        # Act
        lr = LinearRegression()

        # Assert
        with self.assertRaises(ValueError):
            _ = lr.fit(X, y)

    def test_when_simple_perfect_linear_relation_then_correct_params(self):
        # Arrange
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0]})
        y = pd.Series([1.0, 3.0, 5.0, 7.0, 9.0])  # y = 2*x + 1

        # Act
        lr = LinearRegression().fit(X, y)

        # Assert
        self.assertAlmostEqual(lr.intercept_, 1.0, places=7)
        self.assertEqual(lr.coef_.shape, (1, ))
        self.assertAlmostEqual(lr.coef_[0], 2.0, places=7)

    def test_when_predict_then_returns_expected_values(self):
        # Arrange
        X_train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0]})
        y_train = pd.Series([1.0, 3.0, 5.0, 7.0, 9.0])  # y = 2*x + 1

        X_test = pd.DataFrame({"x": [5.0, 6.0]})
        expected = pd.Series([11.0, 13.0], index=X_test.index)

        # Act
        lr = LinearRegression().fit(X_train, y_train)
        actual = lr.predict(X_test)

        # Assert
        pd.testing.assert_series_equal(actual, expected, check_dtype=False)

    def test_when_score_on_perfect_fit_then_r2_is_one(self):
        # Arrange:
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        y = pd.Series([1.0, 3.0, 5.0, 7.0])  # y = 2*x + 1
        lr = LinearRegression().fit(X, y)

        # Act
        score = lr.score(X, y)

        # Assert
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()

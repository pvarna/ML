import unittest
import numpy as np
import pandas as pd

from linear_model import LinearRegression, LogisticRegression, Ridge, Lasso


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


class TestLogisticRegressionInitFitPredict(unittest.TestCase):

    def test_when_predict_before_fit_then_throws_runtime_error(self):
        # Arrange
        X = pd.DataFrame({"x": [0.0, 1.0]})
        lr = LogisticRegression()

        # Assert
        with self.assertRaises(RuntimeError):
            _ = lr.predict(X)

    def test_when_fit_on_two_classes_then_learns_distinguishable_weights(self):
        # Arrange
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        y = pd.Series(["A", "A", "B", "B"])

        # Act
        lr = LogisticRegression().fit(X, y, lr=0.1, max_iter=2000)

        # Assert
        self.assertEqual(set(lr.classes_), {"A", "B"})
        self.assertEqual(lr.coef_.shape, (2, 1))

    def test_when_fit_on_linearly_separable_data_then_high_score(self):
        # Arrange
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
        y = pd.Series(["A", "A", "A", "B", "B", "B"])
        lr = LogisticRegression()

        # Act
        lr.fit(X, y, lr=0.1, max_iter=3000)
        score = lr.score(X, y)

        # Assert
        self.assertGreater(score, 0.9)

    def test_when_predict_proba_then_rows_sum_to_one(self):
        # Arrange
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        y = pd.Series(["A", "A", "B", "B"])
        lr = LogisticRegression().fit(X, y, lr=0.1, max_iter=2000)

        # Act
        probas = lr.predict_proba(X)

        # Assert
        np.testing.assert_allclose(probas.sum(axis=1),
                                   np.ones(X.shape[0]),
                                   rtol=1e-6)

    def test_when_predict_then_returns_class_labels(self):
        # Arrange
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        y = pd.Series(["A", "A", "B", "B"])
        lr = LogisticRegression().fit(X, y, lr=0.1, max_iter=2000)

        # Act
        predictions = lr.predict(X)

        # Assert
        self.assertTrue(set(predictions.unique()).issubset({"A", "B"}))
        self.assertEqual(predictions.shape[0], X.shape[0])

    def test_when_score_on_perfect_binary_split_then_accuracy_is_one(self):
        # Arrange
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        y = pd.Series(["A", "A", "B", "B"])
        lr = LogisticRegression().fit(X, y, lr=0.1, max_iter=3000)

        # Act
        score = lr.score(X, y)

        # Assert
        self.assertAlmostEqual(score, 1.0, places=1)

    def test_when_fit_on_three_classes_then_correct_number_of_weights(self):
        # Arrange
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
        y = pd.Series(["A", "A", "B", "B", "C", "C"])
        lr = LogisticRegression()

        # Act
        lr.fit(X, y, lr=0.1, max_iter=3000)

        # Assert
        self.assertEqual(set(lr.classes_), {"A", "B", "C"})
        self.assertEqual(lr.coef_.shape, (3, 1))
        self.assertEqual(lr.intercept_.shape, (3, ))

    def test_when_predict_proba_for_three_classes_then_rows_sum_to_one(self):
        # Arrange
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
        y = pd.Series(["A", "A", "B", "B", "C", "C"])
        lr = LogisticRegression().fit(X, y, lr=0.1, max_iter=3000)

        # Act
        probas = lr.predict_proba(X)

        # Assert
        np.testing.assert_allclose(probas.sum(axis=1),
                                   np.ones(X.shape[0]),
                                   rtol=1e-6)

    def test_when_predict_on_three_classes_then_outputs_valid_labels(self):
        # Arrange
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
        y = pd.Series(["A", "A", "B", "B", "C", "C"])
        lr = LogisticRegression().fit(X, y, lr=0.1, max_iter=3000)

        # Act
        predictions = lr.predict(X)

        # Assert
        self.assertTrue(set(predictions.unique()).issubset({"A", "B", "C"}))
        self.assertEqual(predictions.shape[0], X.shape[0])

class TestRidgeInitFitPredict(unittest.TestCase):

    def test_when_predict_before_fit_then_throws_runtime_error(self):
        # Arrange
        X = pd.DataFrame({"x": [0.0, 1.0]})
        
        # Act
        ridge = Ridge(alpha=1.0)

        # Assert
        with self.assertRaises(RuntimeError):
            _ = ridge.predict(X)

    def test_when_collinear_features_and_alpha_gt_zero_then_fits_without_error(self):
        # Arrange
        X = pd.DataFrame({
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [2.0, 4.0, 6.0, 8.0],
        })
        y = pd.Series([3.0, 5.0, 7.0, 9.0])
        ridge = Ridge(alpha=1.0)

        # Act
        _ = ridge.fit(X, y)
        
        # Assert
        self.assertIsNotNone(ridge.coef_)
        self.assertIsNotNone(ridge.intercept_)

    def test_when_collinear_features_and_alpha_is_zero_then_throws_value_error(self):
        # Arrange
        X = pd.DataFrame({
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [2.0, 4.0, 6.0, 8.0],
        })
        y = pd.Series([3.0, 5.0, 7.0, 9.0])
        ridge = Ridge(alpha=0.0)

        # Act & Assert
        with self.assertRaises(ValueError):
            _ = ridge.fit(X, y)

    def test_when_simple_linear_relation_and_alpha_is_zero_then_correct_params(self):
        # Arrange
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0]})
        y = pd.Series([1.0, 3.0, 5.0, 7.0, 9.0])  # y = 2*x + 1

        # Act
        ridge = Ridge(alpha=0.0).fit(X, y)

        # Assert
        self.assertAlmostEqual(ridge.intercept_, 1.0, places=7)
        self.assertEqual(ridge.coef_.shape, (1, ))
        self.assertAlmostEqual(ridge.coef_[0], 2.0, places=7)

    def test_when_simple_linear_relation_and_alpha_gt_zero_then_params_are_shrunk(self):
        # Arrange
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0]})
        y = pd.Series([1.0, 3.0, 5.0, 7.0, 9.0])  # y = 2*x + 1

        # Act
        lr = LinearRegression().fit(X, y)
        ols_coef = lr.coef_[0]
        
        ridge = Ridge(alpha=2.0).fit(X, y)
        ridge_coef = ridge.coef_[0]
        
        # Assert
        self.assertAlmostEqual(ols_coef, 2.0, places=7)
        self.assertLess(abs(ridge_coef), abs(ols_coef))
        self.assertGreater(abs(ridge_coef), 0)

    def test_when_score_on_perfect_fit_and_alpha_gt_zero_then_r2_is_lt_one(self):
        # Arrange:
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        y = pd.Series([1.0, 3.0, 5.0, 7.0])  # y = 2*x + 1
        
        # Act
        ridge = Ridge(alpha=1.0).fit(X, y)
        score = ridge.score(X, y)

        # Assert
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.9)

class TestLassoInitFitPredict(unittest.TestCase):

    def test_when_predict_before_fit_then_throws_runtime_error(self):
        # Arrange
        X = pd.DataFrame({"x": [0.0, 1.0]})
        
        # Act
        lasso = Lasso(alpha=1.0)

        # Assert
        with self.assertRaises(RuntimeError):
            _ = lasso.predict(X)

    def test_when_collinear_features_and_alpha_gt_zero_then_fits_without_error(self):
        # Arrange
        X = pd.DataFrame({
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [2.0, 4.0, 6.0, 8.0],
        })
        y = pd.Series([3.0, 5.0, 7.0, 9.0])
        lasso = Lasso(alpha=1.0)

        # Act
        _ = lasso.fit(X, y)
        
        # Assert
        self.assertIsNotNone(lasso.coef_)
        self.assertIsNotNone(lasso.intercept_)

    def test_when_simple_linear_relation_and_alpha_is_zero_then_correct_params(self):
        # Arrange
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0]})
        y = pd.Series([1.0, 3.0, 5.0, 7.0, 9.0])  # y = 2*x + 1

        # Act
        lasso = Lasso(alpha=0.0).fit(X, y, lr=0.01, max_iter=5000)

        # Assert
        self.assertAlmostEqual(lasso.intercept_, 1.0, places=2)
        self.assertEqual(lasso.coef_.shape, (1, ))
        self.assertAlmostEqual(lasso.coef_[0], 2.0, places=2)

    def test_when_simple_linear_relation_and_alpha_gt_zero_then_params_are_shrunk(self):
        # Arrange
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0]})
        y = pd.Series([1.0, 3.0, 5.0, 7.0, 9.0])  # y = 2*x + 1

        # Act
        lr = LinearRegression().fit(X, y)
        ols_coef = lr.coef_[0]
        
        lasso = Lasso(alpha=0.1).fit(X, y, lr=0.01, max_iter=5000)
        lasso_coef = lasso.coef_[0]
        
        # Assert
        self.assertAlmostEqual(ols_coef, 2.0, places=7)
        self.assertLess(abs(lasso_coef), abs(ols_coef))
        self.assertGreater(abs(lasso_coef), 0)

    def test_when_score_on_perfect_fit_and_alpha_gt_zero_then_r2_is_lt_one(self):
        # Arrange:
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        y = pd.Series([1.0, 3.0, 5.0, 7.0])  # y = 2*x + 1
        
        # Act
        lasso = Lasso(alpha=0.1).fit(X, y)
        score = lasso.score(X, y)

        # Assert
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.9)

    def test_when_noise_features_and_alpha_gt_zero_then_noise_coefs_are_near_zero(self):
        # Arrange
        rng = np.random.RandomState(42)
        X_real = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        y = pd.Series(3.0 * X_real - 2.0)
        
        X = pd.DataFrame({
            "x_real": X_real,
            "x_noise_1": rng.rand(10),
            "x_noise_2": rng.rand(10)
        })

        # Act
        lasso = Lasso(alpha=1.0).fit(X, y, lr=0.001, max_iter=10000)

        # Assert
        self.assertGreater(abs(lasso.coef_[0]), 1.0)
        self.assertAlmostEqual(lasso.coef_[1], 0.0, places=2)
        self.assertAlmostEqual(lasso.coef_[2], 0.0, places=2)

if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
import pandas as pd

from metrics import accuracy_score
from metrics import recall_score
from metrics import precision_score
from metrics import f1_score
from metrics import euclidean_distance
from metrics import manhattan_distance
from metrics import r2_score
from metrics import r2_adjusted_score
from metrics import root_mean_squared_error
from metrics import log_loss


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


class TestR2AdjustedScore(unittest.TestCase):

    def test_when_sizes_differ_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = pd.Series([1.0, 2.0])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = r2_adjusted_score(y_true, y_pred, features_count=1)

    def test_when_perfect_prediction_then_returns_one(self):
        # Arrange
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        features_count = 1
        expected = 1.0

        # Act
        actual = r2_adjusted_score(y_true, y_pred, features_count)

        # Assert
        self.assertEqual(actual, expected)

    def test_when_constant_y_true_then_returns_zero(self):
        # Arrange
        y_true = [5.0, 5.0, 5.0, 5.0]
        y_pred = [4.0, 6.0, 5.0, 5.0]
        features_count = 0
        expected = 0.0

        # Act
        actual = r2_adjusted_score(y_true, y_pred, features_count)

        # Assert
        self.assertEqual(actual, expected)

    def test_when_known_values_then_correct_adjusted_score(self):
        # Arrange
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 2.0]
        features_count = 1
        expected = 0.0

        # Act
        actual = r2_adjusted_score(y_true, y_pred, features_count)

        # Assert
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_bad_model_then_negative_adjusted_r2(self):
        # Arrange
        y_true = [0.0, 1.0]
        y_pred = [10.0, 10.0]
        features_count = 0 

        # Act
        actual = r2_adjusted_score(y_true, y_pred, features_count)

        # Assert
        self.assertLess(actual, 0.0)

    def test_when_too_many_features_then_division_by_zero(self):
        # Arrange
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        features_count = 2

        # Act & Assert
        with self.assertRaises(ZeroDivisionError):
            _ = r2_adjusted_score(y_true, y_pred, features_count)

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


class TestRecallScore(unittest.TestCase):

    def test_when_sizes_differ_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([1, 0, 1])
        y_pred = pd.Series([1, 0])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = recall_score(y_true, y_pred)

    def test_when_invalid_average_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([1, 0])
        y_pred = pd.Series([1, 0])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = recall_score(y_true, y_pred, average="pesho")

    def test_when_invalid_zero_division_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([1, 0])
        y_pred = pd.Series([1, 0])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = recall_score(y_true, y_pred, zero_division="pesho")

    def test_when_binary_average_and_y_non_binary_then_throws_runtime_error(
            self):
        # Arrange
        y_true = pd.Series([1, 2, 3])
        y_pred = pd.Series([1, 0, 1])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = recall_score(y_true, y_pred, average="binary")

    def test_when_binary_average_and_union_of_labels_has_more_than_two_classes_then_throws_runtime_error(
            self):
        # Arrange
        y_true = pd.Series([0, 1, 1])
        y_pred = pd.Series([0, 2, 2])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = recall_score(y_true, y_pred, average="binary")

    def test_when_invalid_pos_label_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([1, 0, 1])
        y_pred = pd.Series([1, 0, 1])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = recall_score(y_true, y_pred, average="binary", pos_label=2)

    def test_when_denominator_is_0_and_warn_then_return_0(self):
        # Arrange
        y_true = pd.Series([0, 0, 0])
        y_pred = pd.Series([0, 0, 0])  # everything is TN

        # Act
        actual = recall_score(y_true,
                              y_pred,
                              average="binary",
                              pos_label=1,
                              zero_division="warn")

        # Assert
        expected = 0
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_denominator_is_0_and_zero_division_is_number_then_return_zero_division(
            self):
        # Arrange
        y_true = pd.Series([0, 0, 0])
        y_pred = pd.Series([0, 0, 0])  # everything is TN

        # Act
        zero_division = 1
        actual = recall_score(y_true,
                              y_pred,
                              average="binary",
                              pos_label=1,
                              zero_division=zero_division)

        # Assert
        expected = zero_division
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_binary_average_and_non_zero_denominator_then_return_correct_result(
            self):
        # Arrange
        y_true = pd.Series([1, 0, 1])
        y_pred = pd.Series([1, 0, 0])

        # Act
        actual_0 = recall_score(y_true, y_pred, average="binary", pos_label=0)
        actual_1 = recall_score(y_true, y_pred, average="binary", pos_label=1)

        # Assert
        expected_0 = 1.0
        expected_1 = 0.5
        self.assertAlmostEqual(actual_0, expected_0, places=7)
        self.assertAlmostEqual(actual_1, expected_1, places=7)

    def test_when_micro_average_and_non_zero_denominator_then_return_correct_result(
            self):
        # Arrange
        y_true = pd.Series([0, 1, 2, 2, 1])
        y_pred = pd.Series([0, 2, 2, 1, 1])

        # Act
        actual = recall_score(y_true, y_pred, average="micro")

        # Assert
        expected = 3 / 5
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_macro_average_and_empty_labels_then_throws_runtime_error(
            self):
        # Arrange
        y_true = pd.Series([0, 1, 1, 0])
        y_pred = pd.Series([0, 1, 0, 1])
        labels = []

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = recall_score(y_true, y_pred, average="macro", labels=labels)

    def test_when_macro_average_and_labels_contains_unknown_label_then_throws_runtime_error(
            self):
        # Arrange
        y_true = pd.Series([0, 1, 1, 0])
        y_pred = pd.Series([0, 1, 0, 1])
        labels = [0, 2]

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = recall_score(y_true, y_pred, average="macro", labels=labels)

    def test_when_labels_have_duplicates_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([0, 1, 1, 0])
        y_pred = pd.Series([0, 1, 0, 1])
        labels = [0, 0, 1]

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = recall_score(y_true, y_pred, average="macro", labels=labels)

    def test_when_macro_average_and_non_zero_denominator_then_return_correct_result(
            self):
        # Arrange
        y_true = pd.Series([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        y_pred = pd.Series([0, 1, 2, 1, 0, 2, 2, 2, 0, 0, 2, 2])

        # Act
        actual = recall_score(y_true, y_pred, average="macro")

        # Assert
        expected = (1 / 3 + 1 / 4 + 3 / 5) / 3
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_weighted_average_and_non_zero_denominator_then_return_correct_result(
            self):
        # Arrange
        y_true = pd.Series([0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        y_pred = pd.Series([0, 1, 2, 2, 2, 2, 2, 0, 2, 0])

        # Act
        actual_weighted = recall_score(y_true, y_pred, average="weighted")
        actual_macro = recall_score(y_true, y_pred, average="macro")

        # Assert
        expected_weighted = (1.0 * 1 + 0.25 * 4 + 0.6 * 5) / 10
        expected_macro = (1.0 + 0.25 + 0.6) / 3
        self.assertAlmostEqual(actual_weighted, expected_weighted, places=7)
        self.assertAlmostEqual(actual_macro, expected_macro, places=7)

    def test_when_none_average_and_non_zero_denominator_then_return_correct_result(
            self):
        # Arrange
        y_true = pd.Series([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        y_pred = pd.Series([0, 1, 2, 1, 0, 2, 2, 2, 0, 0, 2, 2])

        # Act
        actual = recall_score(y_true, y_pred, average=None)

        # Assert
        expected = [1 / 3, 1 / 4, 3 / 5]
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_average_none_and_labels_custom_order_then_match_that_order(
            self):
        # Arrange
        y_true = pd.Series([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        y_pred = pd.Series([0, 1, 2, 1, 0, 2, 2, 2, 0, 0, 2, 2])
        labels = [2, 0, 1]

        # Act
        actual = recall_score(y_true, y_pred, average=None, labels=labels)

        # Assert
        expected = [3 / 5, 1 / 3, 1 / 4]
        self.assertEqual(actual, expected)

    def test_when_average_samples_then_throws_not_implemented_runtime_error(
            self):
        # Arrange
        y_true = pd.Series([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        y_pred = pd.Series([0, 1, 2, 1, 0, 2, 2, 2, 0, 0, 2, 2])

        # Act & Assert
        with self.assertRaises(RuntimeError) as context:
            _ = recall_score(y_true, y_pred, average="samples")


class TestPrecisionScore(unittest.TestCase):

    def test_when_sizes_differ_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([1, 0, 1])
        y_pred = pd.Series([1, 0])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = precision_score(y_true, y_pred)

    def test_when_invalid_average_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([1, 0])
        y_pred = pd.Series([1, 0])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = precision_score(y_true, y_pred, average="pesho")

    def test_when_invalid_zero_division_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([1, 0])
        y_pred = pd.Series([1, 0])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = precision_score(y_true, y_pred, zero_division="pesho")

    def test_when_binary_average_and_y_non_binary_then_throws_runtime_error(
            self):
        # Arrange
        y_true = pd.Series([1, 2, 3])
        y_pred = pd.Series([1, 0, 1])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = precision_score(y_true, y_pred, average="binary")

    def test_when_binary_average_and_union_of_labels_has_more_than_two_classes_then_throws_runtime_error(
            self):
        # Arrange
        y_true = pd.Series([0, 1, 1])
        y_pred = pd.Series([0, 2, 2])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = precision_score(y_true, y_pred, average="binary")

    def test_when_invalid_pos_label_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([1, 0, 1])
        y_pred = pd.Series([1, 0, 1])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = precision_score(y_true, y_pred, average="binary", pos_label=2)

    def test_when_denominator_is_0_and_warn_then_return_0(self):
        # Arrange
        y_true = pd.Series([0, 0, 0])
        y_pred = pd.Series([0, 0, 0])

        # Act
        actual = precision_score(y_true,
                                 y_pred,
                                 average="binary",
                                 pos_label=1,
                                 zero_division="warn")

        # Assert
        expected = 0
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_denominator_is_0_and_zero_division_is_number_then_return_zero_division(
            self):
        # Arrange
        y_true = pd.Series([0, 0, 0])
        y_pred = pd.Series([0, 0, 0])

        # Act
        zero_division = 1
        actual = precision_score(y_true,
                                 y_pred,
                                 average="binary",
                                 pos_label=1,
                                 zero_division=zero_division)

        # Assert
        expected = zero_division
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_binary_average_and_non_zero_denominator_then_return_correct_result(
            self):
        # Arrange
        y_true = pd.Series([1, 0, 1])
        y_pred = pd.Series([1, 0, 0])

        # Act
        actual_0 = precision_score(y_true,
                                   y_pred,
                                   average="binary",
                                   pos_label=0)
        actual_1 = precision_score(y_true,
                                   y_pred,
                                   average="binary",
                                   pos_label=1)

        # Assert
        expected_0 = 0.5
        expected_1 = 1.0
        self.assertAlmostEqual(actual_0, expected_0, places=7)
        self.assertAlmostEqual(actual_1, expected_1, places=7)

    def test_when_micro_average_and_non_zero_denominator_then_return_correct_result(
            self):
        # Arrange
        y_true = pd.Series([0, 1, 2, 2, 1])
        y_pred = pd.Series([0, 2, 2, 1, 1])

        # Act
        actual = precision_score(y_true, y_pred, average="micro")

        # Assert
        expected = 3 / 5
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_macro_average_and_empty_labels_then_throws_runtime_error(
            self):
        # Arrange
        y_true = pd.Series([0, 1, 1, 0])
        y_pred = pd.Series([0, 1, 0, 1])
        labels = []

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = precision_score(y_true, y_pred, average="macro", labels=labels)

    def test_when_macro_average_and_labels_contains_unknown_label_then_throws_runtime_error(
            self):
        # Arrange
        y_true = pd.Series([0, 1, 1, 0])
        y_pred = pd.Series([0, 1, 0, 1])
        labels = [0, 2]

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = precision_score(y_true, y_pred, average="macro", labels=labels)

    def test_when_labels_have_duplicates_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([0, 1, 1, 0])
        y_pred = pd.Series([0, 1, 0, 1])
        labels = [0, 0, 1]

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = precision_score(y_true, y_pred, average="macro", labels=labels)

    def test_when_macro_average_and_non_zero_denominator_then_return_correct_result(
            self):
        # Arrange
        y_true = pd.Series([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        y_pred = pd.Series([0, 1, 2, 1, 0, 2, 2, 2, 0, 0, 2, 2])

        # Act
        actual = precision_score(y_true, y_pred, average="macro")

        # Assert
        expected = (1 / 4 + 1 / 2 + 1 / 2) / 3
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_weighted_average_and_non_zero_denominator_then_return_correct_result(
            self):
        # Arrange
        y_true = pd.Series([0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        y_pred = pd.Series([0, 1, 2, 2, 2, 2, 2, 0, 2, 0])

        # Act
        actual_weighted = precision_score(y_true, y_pred, average="weighted")
        actual_macro = precision_score(y_true, y_pred, average="macro")

        # Assert
        expected_weighted = ((1 / 3) * 1 + 1.0 * 4 + (1 / 2) * 5) / 10
        expected_macro = (1 / 3 + 1.0 + 1 / 2) / 3
        self.assertAlmostEqual(actual_weighted, expected_weighted, places=7)
        self.assertAlmostEqual(actual_macro, expected_macro, places=7)

    def test_when_none_average_and_non_zero_denominator_then_return_correct_result(
            self):
        # Arrange
        y_true = pd.Series([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        y_pred = pd.Series([0, 1, 2, 1, 0, 2, 2, 2, 0, 0, 2, 2])

        # Act
        actual = precision_score(y_true, y_pred, average=None)

        # Assert
        expected = [1 / 4, 1 / 2, 1 / 2]
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_average_none_and_labels_custom_order_then_match_that_order(
            self):
        # Arrange
        y_true = pd.Series([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        y_pred = pd.Series([0, 1, 2, 1, 0, 2, 2, 2, 0, 0, 2, 2])
        labels = [2, 0, 1]

        # Act
        actual = precision_score(y_true, y_pred, average=None, labels=labels)

        # Assert
        expected = [1 / 2, 1 / 4, 1 / 2]
        self.assertEqual(actual, expected)

    def test_when_average_samples_then_throws_not_implemented_runtime_error(
            self):
        # Arrange
        y_true = pd.Series([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        y_pred = pd.Series([0, 1, 2, 1, 0, 2, 2, 2, 0, 0, 2, 2])

        # Act & Assert
        with self.assertRaises(RuntimeError) as context:
            _ = precision_score(y_true, y_pred, average="samples")


class TestF1Score(unittest.TestCase):

    def test_when_sizes_differ_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([1, 0, 1])
        y_pred = pd.Series([1, 0])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = f1_score(y_true, y_pred)

    def test_when_invalid_average_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([1, 0])
        y_pred = pd.Series([1, 0])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = f1_score(y_true, y_pred, average="pesho")

    def test_when_invalid_zero_division_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([1, 0])
        y_pred = pd.Series([1, 0])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = f1_score(y_true, y_pred, zero_division="pesho")

    def test_when_binary_average_and_y_non_binary_then_throws_runtime_error(
            self):
        # Arrange
        y_true = pd.Series([1, 2, 3])
        y_pred = pd.Series([1, 0, 1])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = f1_score(y_true, y_pred, average="binary")

    def test_when_binary_average_and_union_of_labels_has_more_than_two_classes_then_throws_runtime_error(
            self):
        # Arrange
        y_true = pd.Series([0, 1, 1])
        y_pred = pd.Series([0, 2, 2])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = f1_score(y_true, y_pred, average="binary")

    def test_when_invalid_pos_label_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([1, 0, 1])
        y_pred = pd.Series([1, 0, 1])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = f1_score(y_true, y_pred, average="binary", pos_label=2)

    def test_when_denominator_is_0_and_warn_then_return_0(self):
        # Arrange
        y_true = pd.Series([0, 0, 0])
        y_pred = pd.Series([0, 0, 0])

        # Act
        actual = f1_score(y_true,
                          y_pred,
                          average="binary",
                          pos_label=1,
                          zero_division="warn")

        # Assert
        expected = 0.0
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_denominator_is_0_and_zero_division_is_number_then_return_zero_division(
            self):
        # Arrange
        y_true = pd.Series([0, 0, 0])
        y_pred = pd.Series([0, 0, 0])

        # Act
        zero_division = 1.0
        actual = f1_score(y_true,
                          y_pred,
                          average="binary",
                          pos_label=1,
                          zero_division=zero_division)

        # Assert
        expected = zero_division
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_binary_average_and_non_zero_denominator_then_return_correct_result(
            self):
        # Arrange
        y_true = pd.Series([1, 0, 1])
        y_pred = pd.Series([1, 0, 0])

        # Act
        actual_0 = f1_score(y_true, y_pred, average="binary", pos_label=0)
        actual_1 = f1_score(y_true, y_pred, average="binary", pos_label=1)

        # Assert
        expected_0 = 2 / 3
        expected_1 = 2 / 3
        self.assertAlmostEqual(actual_0, expected_0, places=7)
        self.assertAlmostEqual(actual_1, expected_1, places=7)

    def test_when_micro_average_and_non_zero_denominator_then_return_correct_result(
            self):
        # Arrange
        y_true = pd.Series([0, 1, 2, 2, 1])
        y_pred = pd.Series([0, 2, 2, 1, 1])

        # Act
        actual = f1_score(y_true, y_pred, average="micro")

        # Assert
        expected = 3 / 5
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_macro_average_and_non_zero_denominator_then_return_correct_result(
            self):
        # Arrange
        y_true = pd.Series([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        y_pred = pd.Series([0, 1, 2, 1, 0, 2, 2, 2, 0, 0, 2, 2])

        # Act
        actual = f1_score(y_true, y_pred, average="macro")

        # Assert
        expected = (2 / 7 + 1 / 3 + 6 / 11) / 3
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_weighted_average_and_non_zero_denominator_then_return_correct_result(
            self):
        # Arrange
        y_true = pd.Series([0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        y_pred = pd.Series([0, 1, 2, 2, 2, 2, 2, 0, 2, 0])

        # Act
        actual_weighted = f1_score(y_true, y_pred, average="weighted")
        actual_macro = f1_score(y_true, y_pred, average="macro")

        # Assert
        expected_weighted = (0.5 * 1 + 0.4 * 4 + (6 / 11) * 5) / 10
        expected_macro = (0.5 + 0.4 + 6 / 11) / 3
        self.assertAlmostEqual(actual_weighted, expected_weighted, places=7)
        self.assertAlmostEqual(actual_macro, expected_macro, places=7)

    def test_when_none_average_and_non_zero_denominator_then_return_correct_result(
            self):
        # Arrange
        y_true = pd.Series([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        y_pred = pd.Series([0, 1, 2, 1, 0, 2, 2, 2, 0, 0, 2, 2])

        # Act
        actual = f1_score(y_true, y_pred, average=None)

        # Assert
        expected = [2 / 7, 1 / 3, 6 / 11]
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_average_none_and_labels_custom_order_then_match_that_order(
            self):
        # Arrange
        y_true = pd.Series([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        y_pred = pd.Series([0, 1, 2, 1, 0, 2, 2, 2, 0, 0, 2, 2])
        labels = [2, 0, 1]

        # Act
        actual = f1_score(y_true, y_pred, average=None, labels=labels)

        # Assert
        expected = [6 / 11, 2 / 7, 1 / 3]
        self.assertEqual(actual, expected)

    def test_when_average_samples_then_throws_not_implemented_runtime_error(
            self):
        # Arrange
        y_true = pd.Series([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        y_pred = pd.Series([0, 1, 2, 1, 0, 2, 2, 2, 0, 0, 2, 2])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = f1_score(y_true, y_pred, average="samples")


class TestLogLoss(unittest.TestCase):

    def test_when_sizes_differ_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([0, 1, 1])
        y_pred = pd.Series([0.2, 0.7, 0.8, 0.3])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = log_loss(y_true, y_pred)

    def test_when_y_pred_is_1d_and_y_true_has_multiclass_labels_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([0, 1, 2])
        y_pred = pd.Series([0.2, 0.7, 0.8])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = log_loss(y_true, y_pred)

    def test_when_simple_binary_case_then_calculates_correct_loss(self):
        # Arrange
        y_true = pd.Series([0, 1])
        y_pred = pd.Series([0.25, 0.75])
        
        # Act
        expected = -np.log(0.75)
        actual = log_loss(y_true, y_pred)

        # Assert
        self.assertAlmostEqual(expected, actual)

    def test_when_perfect_prediction_then_returns_near_zero_loss(self):
        # Arrange
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = pd.Series([0.0, 1.0, 0.0, 1.0])
        
        # Act
        EPS = 0.000001
        expected = -(np.log(1.0 - EPS))
        actual = log_loss(y_true, y_pred)

        # Assert
        self.assertAlmostEqual(expected, actual)

    def test_when_perfectly_wrong_prediction_then_returns_high_loss(self):
        # Arrange
        y_true = pd.Series([0, 1])
        y_pred = pd.Series([1.0, 0.0])

        # Act
        EPS = 0.000001
        expected = -np.log(EPS)
        actual = log_loss(y_true, y_pred)

        # Assert
        self.assertAlmostEqual(expected, actual)

    def test_when_normalize_is_false_then_returns_sum_of_losses(self):
        # Arrange
        y_true = pd.Series([0, 1])
        y_pred = pd.Series([0.25, 0.75])
        
        # Act
        expected = -np.log(0.75) * 2
        actual = log_loss(y_true, y_pred, normalize=False)

        # Assert
        self.assertAlmostEqual(expected, actual)

    def test_when_labels_are_strings_then_calculates_correct_loss(self):
        # Arrange
        y_true = pd.Series(['cat', 'dog']) 
        y_pred = pd.Series([0.25, 0.75])
        
        # Act
        expected = -np.log(0.75)
        actual = log_loss(y_true, y_pred)

        # Assert
        self.assertAlmostEqual(expected, actual)

    def test_when_multiclass_then_calculates_correct_loss(self):
        # Arrange
        y_true = pd.Series([2, 0, 1])
        y_pred = pd.Series([
            [0.1, 0.2, 0.7],
            [0.8, 0.1, 0.1],
            [0.3, 0.6, 0.1] 
        ])
        
        # Act
        l1 = -np.log(0.7)
        l2 = -np.log(0.8)
        l3 = -np.log(0.6)
        expected = (l1 + l2 + l3) / 3.0
        actual = log_loss(y_true, y_pred)
        
        # Assert
        self.assertAlmostEqual(expected, actual)

    def test_when_multiclass_one_hot_then_calculates_correct_loss(self):
        # Arrange
        y_true = pd.Series([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
        y_pred = pd.Series([
            [0.1, 0.2, 0.7],  
            [0.8, 0.1, 0.1], 
            [0.3, 0.6, 0.1]  
        ])
        
        # Act
        l1 = -np.log(0.7)
        l2 = -np.log(0.8)
        l3 = -np.log(0.6)
        expected = (l1 + l2 + l3) / 3.0
        actual = log_loss(y_true, y_pred)
        
        # Assert
        self.assertAlmostEqual(expected, actual)

    def test_when_binary_with_2d_pred_then_calculates_correct_loss(self):
        # Arrange
        y_true = pd.Series([0, 1])
        y_pred = pd.Series([
            [0.75, 0.25], 
            [0.25, 0.75] 
        ])
        
        # Act
        expected = -np.log(0.75)
        actual = log_loss(y_true, y_pred)
        
        # Assert
        self.assertAlmostEqual(expected, actual)

    def test_when_multiclass_and_normalize_is_false_then_returns_sum(self):
        # Arrange
        y_true = pd.Series([2, 0, 1])
        y_pred = pd.Series([
            [0.1, 0.2, 0.7],  
            [0.8, 0.1, 0.1],  
            [0.2, 0.6, 0.2]  
        ])
        
        # Act
        expected = -np.log(0.7) + -np.log(0.8) + -np.log(0.6)
        actual = log_loss(y_true, y_pred, normalize=False)
        
        # Assert
        self.assertAlmostEqual(expected, actual)

    def test_when_multiclass_with_labels_param_then_calculates_correct_loss(self):
        # Arrange
        y_true = pd.Series(['cat', 'dog'])
        y_pred = pd.Series([
            [0.25, 0.75], 
            [0.75, 0.25] 
        ])
        labels = pd.Series(['dog', 'cat'])
        
        # Act
        expected = -np.log(0.75)
        actual = log_loss(y_true, y_pred, labels=labels)
        
        # Assert
        self.assertAlmostEqual(expected, actual)

    def test_when_y_pred_rows_not_normalized_then_renormalizes_and_calculates_correct_loss(self):
        # Arrange
        y_true = pd.Series([0, 1])
        y_pred = pd.Series([
            [1.5, 0.5],
            [0.5, 1.5]   
        ])
        
        # Act
        expected = -np.log(0.75)
        actual = log_loss(y_true, y_pred)
        
        # Assert
        self.assertAlmostEqual(expected, actual)

    def test_when_inputs_are_empty_then_returns_zero(self):
        # Arrange
        y_true = pd.Series([], dtype='int64')
        y_pred = pd.Series([], dtype='float64')
        
        # Act
        expected = 0.0
        actual = log_loss(y_true, y_pred)
        
        # Assert
        self.assertEqual(expected, actual)

    def test_when_y_pred_is_1d_and_y_true_is_2d_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([[0, 1], [1, 0]])
        y_pred = pd.Series([0.8, 0.2])
        
        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = log_loss(y_true, y_pred)
            
    def test_when_y_pred_cols_mismatch_labels_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([0, 1, 2]) 
        y_pred = pd.Series([[0.8, 0.2], [0.3, 0.7], [0.1, 0.9]])
        
        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = log_loss(y_true, y_pred)
            
    def test_when_y_pred_cols_mismatch_one_hot_labels_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([[0,0,1], [1,0,0]]) 
        y_pred = pd.Series([[0.8, 0.2], [0.3, 0.7]])
        
        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = log_loss(y_true, y_pred)
            
    def test_when_y_true_label_not_in_labels_map_then_throws_runtime_error(self):
        # Arrange
        y_true = pd.Series([0, 1, 2])
        y_pred = pd.Series([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        labels = pd.Series([0, 1])
        
        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = log_loss(y_true, y_pred, labels=labels)


if __name__ == "__main__":
    unittest.main()

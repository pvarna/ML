import unittest
import numpy as np
import pandas as pd

from model_selection import train_test_split


class TestTrainTestSplit(unittest.TestCase):

    def _simple_dataset(self):
        X = pd.DataFrame({
            "f1": [10, 11, 12, 13, 14, 15],
            "f2": [0, 1, 2, 3, 4, 5],
        })
        y = pd.Series([0, 0, 0, 1, 1, 1], name="y")
        return X, y

    def test_when_both_sizes_are_none_then_throws_runtime_error(self):
        # Arrange
        X = pd.DataFrame({"a": [1, 2]})
        y = pd.Series([0, 1])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = train_test_split(X, y, test_size=None, train_size=None)

    def test_when_sizes_out_of_bounds_then_throws_runtime_error(self):
        # Arrange
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1, 0])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = train_test_split(X, y, test_size=1.2)
        with self.assertRaises(RuntimeError):
            _ = train_test_split(X, y, test_size=-0.1)
        with self.assertRaises(RuntimeError):
            _ = train_test_split(X, y, train_size=0.4, test_size=0.7)

    def test_when_X_y_lengths_differ_then_throws_runtime_error(self):
        # Arrange
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = train_test_split(X, y)

    def test_when_less_than_two_items_then_throws_runtime_error(self):
        # Arrange
        X = pd.DataFrame({"a": [1]})
        y = pd.Series([0])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = train_test_split(X, y)

    def test_when_stratify_not_y_or_X_column_then_throws_runtime_error(self):
        # Arrange
        X = pd.DataFrame({"a": [1, 2, 3, 4]})
        y = pd.Series([0, 1, 0, 1])
        bad_stratify = pd.Series([9, 9, 9, 8])

        # Act & Assert
        with self.assertRaises(RuntimeError):
            _ = train_test_split(X, y, stratify=bad_stratify)

    def test_when_stratify_group_has_fewer_than_two_items_then_throws_runtime_error(
            self):
        # Arrange
        X = pd.DataFrame({
            "f1": [1, 2, 3, 4],
            "f2": [5, 6, 7, 8],
        })
        y = pd.Series([0, 0, 0, 1], name="y")
        stratify = y

        # Act & Assert
        with self.assertRaises(RuntimeError) as cm:
            _ = train_test_split(X, y, test_size=0.5, stratify=stratify)

    def test_when_no_shuffle_then_last_rows_go_to_test(self):
        # Arrange
        X, y = self._simple_dataset()
        test_size = 0.33  # train - 4, test - 2
        expected_test_idx = X.index[-2:].to_list()

        # Act
        X_tr, X_te, y_tr, y_te = train_test_split(X,
                                                  y,
                                                  test_size=test_size,
                                                  shuffle=False)

        # Assert
        self.assertEqual(y_te.index.to_list(), expected_test_idx)
        self.assertEqual(len(X_te), 2)
        self.assertEqual(len(X_tr), 4)

    def test_when_shuffle_and_random_state_fixed_then_split_is_deterministic(
            self):
        # Arrange
        X, y = self._simple_dataset()

        # Act
        X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X,
                                                      y,
                                                      test_size=0.5,
                                                      shuffle=True,
                                                      random_state=42)
        X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X,
                                                      y,
                                                      test_size=0.5,
                                                      shuffle=True,
                                                      random_state=42)

        # Assert
        pd.testing.assert_frame_equal(X_tr1, X_tr2, check_dtype=False)
        pd.testing.assert_frame_equal(X_te1, X_te2, check_dtype=False)
        pd.testing.assert_series_equal(y_tr1, y_tr2, check_dtype=False)
        pd.testing.assert_series_equal(y_te1, y_te2, check_dtype=False)

    def test_when_stratify_then_class_proportions_preserved_per_group(self):
        # Arrange
        X, y = self._simple_dataset()
        stratify = y

        # Act
        X_tr, X_te, y_tr, y_te = train_test_split(X,
                                                  y,
                                                  test_size=0.5,
                                                  shuffle=True,
                                                  random_state=42,
                                                  stratify=stratify)

        # Assert
        self.assertGreaterEqual(y_tr.value_counts().get(0, 0), 1)
        self.assertGreaterEqual(y_tr.value_counts().get(1, 0), 1)
        self.assertGreaterEqual(y_te.value_counts().get(0, 0), 1)
        self.assertGreaterEqual(y_te.value_counts().get(1, 0), 1)

        self.assertEqual(len(X_tr) + len(X_te), len(X))
        self.assertEqual(len(y_tr) + len(y_te), len(y))


import unittest
import math
import pandas as pd

from tree import DecisionTreeClassifier, Node


class TestNode(unittest.TestCase):

    def test_when_leaf_node_then_is_leaf_true(self):
        # Arrange
        node = Node(value=1)

        # Act
        is_leaf = node.is_leaf()

        # Assert
        self.assertTrue(is_leaf)

    def test_when_non_leaf_node_then_is_leaf_false(self):
        # Arrange
        node = Node(feature=0, threshold=2.5)

        # Act
        is_leaf = node.is_leaf()

        # Assert
        self.assertFalse(is_leaf)


class TestDecisionTreeClassifierInit(unittest.TestCase):

    def test_when_invalid_criterion_then_raises_runtime_error(self):
        # Arrange & Act & Assert
        with self.assertRaises(RuntimeError):
            _ = DecisionTreeClassifier(min_samples_leaf=1,
                                       min_samples_split=2,
                                       max_depth=3,
                                       criterion="invalid")

    def test_when_gini_criterion_then_criterion_function_is_gini(self):
        # Arrange
        classifier = DecisionTreeClassifier(1, 2, 3, 'gini')
        X, y = make_simple_two_feature()

        # Act
        actual = classifier._impurity(y)

        # Assert
        expected = 0.5
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_entropy_criterion_then_criterion_function_is_entropy(self):
        # Arrange
        classifier = DecisionTreeClassifier(1, 2, 3, 'entropy')
        X, y = make_simple_two_feature()

        # Act
        actual = classifier._impurity(y)

        # Assert (entropy of p=[0.5,0.5] is 1.0)
        expected = 1.0
        self.assertAlmostEqual(actual, expected, places=7)


class TestDecisionTreeClassifierMajorityClass(unittest.TestCase):

    def test_when_majority_class_requested_then_correct_label_returned(self):
        # Arrange
        X, y = make_simple_two_feature()
        classifier = DecisionTreeClassifier(1, 2, 3, 'gini')

        # Act
        actual = classifier._majority_class(y)

        # Assert
        expected = y.value_counts().idxmax()
        self.assertEqual(actual, expected)


class TestDecisionTreeClassifierBestSplit(unittest.TestCase):

    def test_when_valid_splits_exist_then_information_gain_positive(self):
        # Arrange
        X, y = make_simple_two_feature()
        classifier = DecisionTreeClassifier(1, 2, 3, 'gini')

        # Act
        feature, threshold, info_gain = classifier._best_split(X, y)

        # Assert
        self.assertIsNotNone(feature)
        self.assertIsNotNone(threshold)
        self.assertGreaterEqual(info_gain, 0.0)

    def test_when_min_samples_leaf_too_large_then_no_split(self):
        # Arrange
        X, y = make_single_feature()
        classifier = DecisionTreeClassifier(min_samples_leaf=4,
                                     min_samples_split=2,
                                     max_depth=5,
                                     criterion='gini')
        # Act
        feature, threshold, info_gain = classifier._best_split(X, y)

        # Assert
        self.assertIsNone(feature)
        self.assertIsNone(threshold)
        self.assertEqual(info_gain, 0.0)


class TestDecisionTreeClassifierBuildAndPredict(unittest.TestCase):

    def test_when_max_depth_zero_then_root_is_leaf(self):
        # Arrange
        X, y = make_simple_two_feature()
        classifier = DecisionTreeClassifier(1, 2, 0, 'gini')

        # Act
        classifier.fit(X, y)

        # Assert
        self.assertTrue(classifier.root.is_leaf())

    def test_when_all_same_class_then_root_is_that_class(self):
        # Arrange
        X, y = make_simple_two_feature()
        y_same = pd.Series([1] * len(y))
        classifier = DecisionTreeClassifier(1, 2, 3, 'gini')

        # Act
        classifier.fit(X, y_same)

        # Assert
        self.assertTrue(classifier.root.is_leaf())
        self.assertEqual(classifier.root.value, 1)

    def test_when_fit_then_root_not_none(self):
        # Arrange
        X, y = make_simple_two_feature()
        classifier = DecisionTreeClassifier(1, 2, 3, 'gini')

        # Act
        classifier.fit(X, y)

        # Assert
        self.assertIsNotNone(classifier.root)

    def test_when_predict_then_returns_series_of_labels(self):
        # Arrange
        X, y = make_simple_two_feature()
        classifier = DecisionTreeClassifier(1, 2, 3, 'gini').fit(X, y)

        # Act
        predictions = classifier.predict(X)

        # Assert
        self.assertEqual(predictions.shape[0], X.shape[0])
        self.assertTrue(set(predictions.unique()).issubset(set(y.unique())))

    def test_when_predict_sample_then_returns_label(self):
        # Arrange
        X, y = make_simple_two_feature()
        classifier = DecisionTreeClassifier(1, 2, 3, 'gini').fit(X, y)
        sample = X.iloc[0]

        # Act
        label = classifier._predict_sample(classifier.root, sample)

        # Assert
        self.assertIn(label, y.unique())


class TestDecisionTreeClassifierHyperparameters(unittest.TestCase):

    def test_when_min_samples_split_greater_than_num_samples_then_root_leaf(
            self):
        # Arrange
        X, y = make_simple_two_feature()
        classifier = DecisionTreeClassifier(min_samples_leaf=1,
                                     min_samples_split=len(X) + 1,
                                     max_depth=5,
                                     criterion='gini')

        # Act
        classifier.fit(X, y)

        # Assert
        self.assertTrue(classifier.root.is_leaf())

    def test_when_min_samples_leaf_large_then_no_split(self):
        # Arrange
        X, y = make_single_feature()
        classifier = DecisionTreeClassifier(min_samples_leaf=5,
                                     min_samples_split=2,
                                     max_depth=5,
                                     criterion='entropy')

        # Act
        classifier.fit(X, y)

        # Assert
        self.assertTrue(classifier.root.is_leaf())

    def test_when_max_depth_limits_growth_then_tree_shallower(self):
        # Arrange
        X, y = make_simple_two_feature()
        shallow = DecisionTreeClassifier(1, 2, 0, 'gini')
        deep = DecisionTreeClassifier(1, 2, 5, 'gini')

        # Act
        shallow.fit(X, y)
        deep.fit(X, y)

        # Assert
        self.assertTrue(shallow.root.is_leaf())
        self.assertFalse(deep.root.is_leaf())


def make_simple_two_feature():
    X = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6], "B": [5, 4, 3, 2, 1, 0]})
    y = pd.Series([0, 0, 1, 1, 1, 0])
    return X, y


def make_single_feature():
    X = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6]})
    y = pd.Series([0, 0, 1, 1, 1, 0])
    return X, y

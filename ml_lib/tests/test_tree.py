import unittest
import math
import pandas as pd
import numpy as np

from tree import DecisionTreeClassifier, Node, RandomForestClassifier, AdaBoostClassifier
from stats import gini_index, entropy


class TestNode(unittest.TestCase):

    def test_when_leaf_node_then_is_leaf_true(self):
        # Arrange
        node = Node(value=1)

        # Act
        actual = node.is_leaf()
        expected = True

        # Assert
        self.assertEqual(expected, actual)

    def test_when_non_leaf_node_then_is_leaf_false(self):
        # Arrange
        node = Node(feature=0, threshold=2.5)

        # Act
        actual = node.is_leaf()
        expected = False

        # Assert
        self.assertEqual(expected, actual)


class TestDecisionTreeClassifierInit(unittest.TestCase):

    def test_when_invalid_criterion_then_raises_runtime_error(self):
        # Arrange & Act & Assert
        with self.assertRaises(RuntimeError):
            _ = DecisionTreeClassifier(
                min_samples_leaf=1,
                min_samples_split=2,
                max_depth=3,
                criterion="invalid",
            )

    def test_when_gini_criterion_then_criterion_function_is_gini(self):
        # Arrange
        classifier = DecisionTreeClassifier(1, 2, 3, 'gini')
        X, y = make_simple_two_feature()

        # Act
        actual = classifier._impurity(y)
        expected = 0.5

        # Assert
        self.assertAlmostEqual(actual, expected, places=7)

    def test_when_entropy_criterion_then_criterion_function_is_entropy(self):
        # Arrange
        classifier = DecisionTreeClassifier(1, 2, 3, 'entropy')
        X, y = make_simple_two_feature()

        # Act
        actual = classifier._impurity(y)
        expected = 1.0

        # Assert
        self.assertAlmostEqual(actual, expected, places=7)


class TestDecisionTreeClassifierMajorityClass(unittest.TestCase):

    def test_when_majority_class_requested_then_correct_label_returned(self):
        # Arrange
        X, y = make_simple_two_feature()
        classifier = DecisionTreeClassifier(1, 2, 3, 'gini')

        # Act
        actual = classifier._majority_class(y)
        expected = y.value_counts().idxmax()

        # Assert
        self.assertEqual(expected, actual)


class TestDecisionTreeClassifierBestSplit(unittest.TestCase):

    def test_when_valid_splits_exist_then_information_gain_positive(self):
        # Arrange
        X, y = make_simple_two_feature()
        classifier = DecisionTreeClassifier(1, 2, 3, 'gini')

        # Act
        feature, threshold, info_gain = classifier._best_split(X, y)

        # Assert
        expected_feature_none = False
        actual_feature_none = feature is None
        self.assertEqual(expected_feature_none, actual_feature_none)

        expected_threshold_none = False
        actual_threshold_none = threshold is None
        self.assertEqual(expected_threshold_none, actual_threshold_none)

        expected_min_gain = 0.0
        actual_gain = info_gain
        self.assertGreaterEqual(actual_gain, expected_min_gain)

    def test_when_min_samples_leaf_too_large_then_no_split(self):
        # Arrange
        X, y = make_single_feature()
        classifier = DecisionTreeClassifier(
            min_samples_leaf=4,
            min_samples_split=2,
            max_depth=5,
            criterion='gini',
        )

        # Act
        feature, threshold, info_gain = classifier._best_split(X, y)

        # Assert
        expected_feature = None
        actual_feature = feature
        self.assertEqual(expected_feature, actual_feature)

        expected_threshold = None
        actual_threshold = threshold
        self.assertEqual(expected_threshold, actual_threshold)

        expected_gain = 0.0
        actual_gain = info_gain
        self.assertEqual(expected_gain, actual_gain)


class TestDecisionTreeClassifierBuildAndPredict(unittest.TestCase):

    def test_when_max_depth_zero_then_root_is_leaf(self):
        # Arrange
        X, y = make_simple_two_feature()
        classifier = DecisionTreeClassifier(1, 2, 0, 'gini')

        # Act
        classifier.fit(X, y)
        actual = classifier.root.is_leaf()
        expected = True

        # Assert
        self.assertEqual(expected, actual)

    def test_when_all_same_class_then_root_is_that_class(self):
        # Arrange
        X, y = make_simple_two_feature()
        y_same = pd.Series([1] * len(y))
        classifier = DecisionTreeClassifier(1, 2, 3, 'gini')

        # Act
        classifier.fit(X, y_same)
        actual_is_leaf = classifier.root.is_leaf()
        expected_is_leaf = True
        actual_value = classifier.root.value
        expected_value = 1

        # Assert
        self.assertEqual(expected_is_leaf, actual_is_leaf)
        self.assertEqual(expected_value, actual_value)

    def test_when_fit_then_root_not_none(self):
        # Arrange
        X, y = make_simple_two_feature()
        classifier = DecisionTreeClassifier(1, 2, 3, 'gini')

        # Act
        classifier.fit(X, y)
        actual = classifier.root is not None
        expected = True

        # Assert
        self.assertEqual(expected, actual)

    def test_when_predict_then_returns_series_of_labels(self):
        # Arrange
        X, y = make_simple_two_feature()
        classifier = DecisionTreeClassifier(1, 2, 3, 'gini').fit(X, y)

        # Act
        predictions = classifier.predict(X)
        actual_len = predictions.shape[0]
        expected_len = X.shape[0]
        actual_labels_subset = set(predictions.unique()).issubset(
            set(y.unique()))
        expected_labels_subset = True

        # Assert
        self.assertEqual(expected_len, actual_len)
        self.assertEqual(expected_labels_subset, actual_labels_subset)

    def test_when_predict_sample_then_returns_label(self):
        # Arrange
        X, y = make_simple_two_feature()
        classifier = DecisionTreeClassifier(1, 2, 3, 'gini').fit(X, y)
        sample = X.iloc[0]

        # Act
        label = classifier._predict_sample(classifier.root, sample)
        actual_in_classes = label in y.unique()
        expected_in_classes = True

        # Assert
        self.assertEqual(expected_in_classes, actual_in_classes)


class TestDecisionTreeClassifierHyperparameters(unittest.TestCase):

    def test_when_min_samples_split_greater_than_num_samples_then_root_leaf(
            self):
        # Arrange
        X, y = make_simple_two_feature()
        classifier = DecisionTreeClassifier(
            min_samples_leaf=1,
            min_samples_split=len(X) + 1,
            max_depth=5,
            criterion='gini',
        )

        # Act
        classifier.fit(X, y)
        actual = classifier.root.is_leaf()
        expected = True

        # Assert
        self.assertEqual(expected, actual)

    def test_when_min_samples_leaf_large_then_no_split(self):
        # Arrange
        X, y = make_single_feature()
        classifier = DecisionTreeClassifier(
            min_samples_leaf=5,
            min_samples_split=2,
            max_depth=5,
            criterion='entropy',
        )

        # Act
        classifier.fit(X, y)
        actual = classifier.root.is_leaf()
        expected = True

        # Assert
        self.assertEqual(expected, actual)

    def test_when_max_depth_limits_growth_then_tree_shallower(self):
        # Arrange
        X, y = make_simple_two_feature()
        shallow = DecisionTreeClassifier(1, 2, 0, 'gini')
        deep = DecisionTreeClassifier(1, 2, 5, 'gini')

        # Act
        shallow.fit(X, y)
        deep.fit(X, y)

        actual_shallow_leaf = shallow.root.is_leaf()
        expected_shallow_leaf = True
        actual_deep_leaf = deep.root.is_leaf()
        expected_deep_leaf = False

        # Assert
        self.assertEqual(expected_shallow_leaf, actual_shallow_leaf)
        self.assertEqual(expected_deep_leaf, actual_deep_leaf)


def make_simple_two_feature():
    X = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6], "B": [5, 4, 3, 2, 1, 0]})
    y = pd.Series([0, 0, 1, 1, 1, 0])
    return X, y


def make_single_feature():
    X = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6]})
    y = pd.Series([0, 0, 1, 1, 1, 0])
    return X, y


class TestRandomForestInit(unittest.TestCase):

    def test_when_n_estimators_not_positive_then_raises_runtime_error(self):
        # Arrange & Act & Assert
        with self.assertRaises(RuntimeError):
            RandomForestClassifier(n_estimators=0)

    def test_when_unsupported_criterion_then_raises_runtime_error(self):
        # Arrange & Act & Assert
        with self.assertRaises(RuntimeError):
            RandomForestClassifier(criterion="mse")

    def test_when_unsupported_max_features_type_then_raises_runtime_error(
            self):
        # Arrange & Act & Assert
        with self.assertRaises(RuntimeError):
            RandomForestClassifier(max_features="invalid")

    def test_when_oob_score_true_and_bootstrap_false_then_raises_runtime_error(
            self):
        # Arrange & Act & Assert
        with self.assertRaises(RuntimeError):
            RandomForestClassifier(oob_score=True, bootstrap=False)


class TestRandomForestGetFeaturesCount(unittest.TestCase):

    def test_when_max_features_none_then_uses_all_features(self):
        # Arrange
        rf = RandomForestClassifier(max_features=None)
        n_features = 10

        # Act
        actual = rf._get_features_count(n_features)
        expected = n_features

        # Assert
        self.assertEqual(expected, actual)

    def test_when_max_features_int_greater_than_n_features_then_clipped(self):
        # Arrange
        rf = RandomForestClassifier(max_features=20)
        n_features = 5

        # Act
        actual = rf._get_features_count(n_features)
        expected = n_features

        # Assert
        self.assertEqual(expected, actual)

    def test_when_max_features_int_less_than_one_then_at_least_one(self):
        # Arrange
        rf = RandomForestClassifier(max_features=0)
        n_features = 5

        # Act
        actual = rf._get_features_count(n_features)
        expected = 1

        # Assert
        self.assertEqual(expected, actual)

    def test_when_max_features_float_valid_then_scaled_and_clipped(self):
        # Arrange
        rf = RandomForestClassifier(max_features=0.6)
        n_features = 10

        # Act
        actual = rf._get_features_count(n_features)
        expected = max(1, min(int(0.6 * n_features), n_features))

        # Assert
        self.assertEqual(expected, actual)

    def test_when_max_features_float_out_of_range_then_raises_runtime_error(
            self):
        # Arrange
        rf = RandomForestClassifier(max_features=1.5)
        n_features = 10

        # Act & Assert
        with self.assertRaises(RuntimeError):
            rf._get_features_count(n_features)

    def test_when_max_features_sqrt_then_uses_sqrt_of_features(self):
        # Arrange
        rf = RandomForestClassifier(max_features="sqrt")
        n_features = 16

        # Act
        actual = rf._get_features_count(n_features)
        expected = max(1, int(math.sqrt(n_features)))

        # Assert
        self.assertEqual(expected, actual)

    def test_when_max_features_log2_then_uses_log2_of_features(self):
        # Arrange
        rf = RandomForestClassifier(max_features="log2")
        n_features = 16

        # Act
        actual = rf._get_features_count(n_features)
        expected = max(1, int(math.log2(n_features)))

        # Assert
        self.assertEqual(expected, actual)


class TestRandomForestBootstrapSampleIndices(unittest.TestCase):

    def test_when_bootstrap_true_then_samples_with_replacement(self):
        # Arrange
        n_samples = 100
        rf = RandomForestClassifier(bootstrap=True, random_state=42)

        # Act
        indices = rf._bootstrap_sample_indices(n_samples)

        # Assert
        expected_len = n_samples
        actual_len = len(indices)
        self.assertEqual(expected_len, actual_len)

        expected_in_range = True
        actual_in_range = ((0 <= indices) & (indices < n_samples)).all()
        self.assertEqual(expected_in_range, actual_in_range)

    def test_when_bootstrap_false_then_returns_arange(self):
        # Arrange
        n_samples = 10
        rf = RandomForestClassifier(bootstrap=False)

        # Act
        indices = rf._bootstrap_sample_indices(n_samples)

        # Assert
        expected = np.arange(n_samples)
        actual = indices
        self.assertTrue(np.array_equal(expected, actual))


class TestRandomForestFit(unittest.TestCase):

    def test_when_fit_with_mismatched_lengths_then_raises_runtime_error(self):
        # Arrange
        X, y = make_simple_two_feature()
        y_shorter = y.iloc[:-1]
        rf = RandomForestClassifier()

        # Act & Assert
        with self.assertRaises(RuntimeError):
            rf.fit(X, y_shorter)

    def test_when_fit_then_creates_expected_number_of_trees(self):
        # Arrange
        X, y = make_simple_two_feature()
        rf = RandomForestClassifier(
            n_estimators=5,
            max_features="sqrt",
            bootstrap=True,
            random_state=0,
            criterion="entropy",
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=1,
            oob_score=True,
        )

        # Act
        rf.fit(X, y)

        # Assert
        expected_trees_len = 5
        actual_trees_len = len(rf.trees_)
        self.assertEqual(expected_trees_len, actual_trees_len)

        expected_features_per_tree_len = 5
        actual_features_per_tree_len = len(rf.features_per_tree_)
        self.assertEqual(expected_features_per_tree_len,
                         actual_features_per_tree_len)

        expected_n_features_not_none = True
        actual_n_features_not_none = rf.n_features_ is not None
        self.assertEqual(expected_n_features_not_none,
                         actual_n_features_not_none)

        expected_classes_not_none = True
        actual_classes_not_none = rf.classes_ is not None
        self.assertEqual(expected_classes_not_none, actual_classes_not_none)

        expected_all_correct_len = True
        actual_all_correct_len = all(
            len(feats) == rf._get_features_count(rf.n_features_)
            for feats in rf.features_per_tree_)
        self.assertEqual(expected_all_correct_len, actual_all_correct_len)

    def test_when_fit_with_oob_score_true_then_oob_score_attribute_in_valid_range_or_none(
            self):
        # Arrange
        X, y = make_simple_two_feature()
        rf = RandomForestClassifier(
            n_estimators=10,
            bootstrap=True,
            oob_score=True,
            random_state=0,
        )

        # Act
        rf.fit(X, y)

        # Assert
        if rf.oob_score_ is not None:
            expected_min = 0.0
            expected_max = 1.0
            actual = rf.oob_score_
            self.assertGreaterEqual(actual, expected_min)
            self.assertLessEqual(actual, expected_max)


class TestRandomForestPredict(unittest.TestCase):

    def test_when_predict_before_fit_then_raises_runtime_error(self):
        # Arrange
        X, _ = make_simple_two_feature()
        rf = RandomForestClassifier()

        # Act & Assert
        with self.assertRaises(RuntimeError):
            rf.predict(X)

    def test_when_predict_after_fit_then_returns_series_of_correct_length(
            self):
        # Arrange
        X, y = make_simple_two_feature()
        rf = RandomForestClassifier(
            n_estimators=3,
            max_features=1,
            bootstrap=True,
            random_state=1,
        ).fit(X, y)

        # Act
        y_pred = rf.predict(X)

        # Assert
        expected_len = len(X)
        actual_len = len(y_pred)
        self.assertEqual(expected_len, actual_len)


class TestRandomForestScore(unittest.TestCase):

    def test_when_score_on_training_data_then_between_zero_and_one(self):
        # Arrange
        X, y = make_simple_two_feature()
        rf = RandomForestClassifier(
            n_estimators=5,
            max_features="sqrt",
            bootstrap=True,
            random_state=2,
        ).fit(X, y)

        # Act
        score = rf.score(X, y)

        # Assert
        expected_min = 0.0
        expected_max = 1.0
        actual = score
        self.assertGreaterEqual(actual, expected_min)
        self.assertLessEqual(actual, expected_max)


class TestAdaBoostInit(unittest.TestCase):

    def test_when_n_estimators_not_positive_then_raises_runtime_error(self):
        # Arrange & Act & Assert
        with self.assertRaises(RuntimeError):
            AdaBoostClassifier(n_estimators=0)

    def test_when_learning_rate_not_positive_then_raises_runtime_error(self):
        # Arrange & Act & Assert
        with self.assertRaises(RuntimeError):
            AdaBoostClassifier(learning_rate=0.0)

    def test_when_estimator_none_then_uses_decision_tree_stump_defaults(self):
        # Arrange & Act
        clf = AdaBoostClassifier()

        # Assert
        expected_instance = DecisionTreeClassifier
        actual_instance = type(clf.estimator)
        self.assertEqual(expected_instance, actual_instance)

        expected_max_depth = 1
        actual_max_depth = clf.estimator.max_depth
        self.assertEqual(expected_max_depth, actual_max_depth)

        expected_min_samples_leaf = 1
        actual_min_samples_leaf = clf.estimator.min_samples_leaf
        self.assertEqual(expected_min_samples_leaf, actual_min_samples_leaf)

        expected_min_samples_split = 2
        actual_min_samples_split = clf.estimator.min_samples_split
        self.assertEqual(expected_min_samples_split, actual_min_samples_split)

    def test_when_custom_estimator_and_params_provided_then_attributes_are_set(
            self):
        # Arrange
        base_estimator = DecisionTreeClassifier(
            min_samples_leaf=2,
            min_samples_split=5,
            max_depth=3,
            criterion="entropy",
        )

        # Act
        clf = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=7,
            learning_rate=0.5,
            random_state=42,
        )

        # Assert
        expected_estimator = base_estimator
        actual_estimator = clf.estimator
        self.assertIs(expected_estimator, actual_estimator)

        expected_n_estimators = 7
        actual_n_estimators = clf.n_estimators
        self.assertEqual(expected_n_estimators, actual_n_estimators)

        expected_learning_rate = 0.5
        actual_learning_rate = clf.learning_rate
        self.assertEqual(expected_learning_rate, actual_learning_rate)

        expected_random_state = 42
        actual_random_state = clf.random_state
        self.assertEqual(expected_random_state, actual_random_state)


class TestAdaBoostGetFreshEstimator(unittest.TestCase):

    def test_when_base_estimator_with_gini_then_new_estimator_has_same_structure(
            self):
        # Arrange
        base_estimator = DecisionTreeClassifier(
            min_samples_leaf=2,
            min_samples_split=3,
            max_depth=4,
            criterion="gini",
        )
        ada = AdaBoostClassifier(estimator=base_estimator)
        self.assertIs(ada.estimator.criterion, gini_index)

        # Act
        fresh = ada._get_fresh_estimator()

        # Assert
        expected_type = DecisionTreeClassifier
        actual_type = type(fresh)
        self.assertEqual(expected_type, actual_type)

        expected_min_samples_leaf = 2
        actual_min_samples_leaf = fresh.min_samples_leaf
        self.assertEqual(expected_min_samples_leaf, actual_min_samples_leaf)

        expected_min_samples_split = 3
        actual_min_samples_split = fresh.min_samples_split
        self.assertEqual(expected_min_samples_split, actual_min_samples_split)

        expected_max_depth = 4
        actual_max_depth = fresh.max_depth
        self.assertEqual(expected_max_depth, actual_max_depth)

        expected_criterion = gini_index
        actual_criterion = fresh.criterion
        self.assertIs(expected_criterion, actual_criterion)

    def test_when_base_estimator_with_entropy_then_new_estimator_has_same_structure(
            self):
        # Arrange
        base_estimator = DecisionTreeClassifier(
            min_samples_leaf=1,
            min_samples_split=2,
            max_depth=2,
            criterion="entropy",
        )
        ada = AdaBoostClassifier(estimator=base_estimator)
        self.assertIs(ada.estimator.criterion, entropy)

        # Act
        fresh = ada._get_fresh_estimator()

        # Assert
        expected_type = DecisionTreeClassifier
        actual_type = type(fresh)
        self.assertEqual(expected_type, actual_type)

        expected_min_samples_leaf = 1
        actual_min_samples_leaf = fresh.min_samples_leaf
        self.assertEqual(expected_min_samples_leaf, actual_min_samples_leaf)

        expected_min_samples_split = 2
        actual_min_samples_split = fresh.min_samples_split
        self.assertEqual(expected_min_samples_split, actual_min_samples_split)

        expected_max_depth = 2
        actual_max_depth = fresh.max_depth
        self.assertEqual(expected_max_depth, actual_max_depth)

        expected_criterion = entropy
        actual_criterion = fresh.criterion
        self.assertIs(expected_criterion, actual_criterion)

    def test_when_base_estimator_criterion_unknown_then_raises_runtime_error(
            self):
        # Arrange
        base_estimator = DecisionTreeClassifier(
            min_samples_leaf=1,
            min_samples_split=2,
            max_depth=1,
            criterion="gini",
        )
        ada = AdaBoostClassifier(estimator=base_estimator)
        ada.estimator.criterion = object()

        # Act & Assert
        with self.assertRaises(RuntimeError):
            ada._get_fresh_estimator()


class TestAdaBoostPredictScore(unittest.TestCase):

    def test_when_predict_score_after_fit_then_returns_dataframe_with_expected_shape(
            self):
        # Arrange
        X, y = make_simple_two_feature()
        ada = AdaBoostClassifier(
            n_estimators=5,
            learning_rate=0.8,
            random_state=0,
        ).fit(X, y)

        # Act
        scores = ada._predict_score(X)

        # Assert
        expected_type = pd.DataFrame
        actual_type = type(scores)
        self.assertEqual(expected_type, actual_type)

        expected_rows = len(X)
        expected_cols = len(ada.classes_)
        actual_rows, actual_cols = scores.shape
        self.assertEqual(expected_rows, actual_rows)
        self.assertEqual(expected_cols, actual_cols)

        expected_columns = ada.classes_
        actual_columns = scores.columns.to_numpy()
        self.assertTrue(np.array_equal(expected_columns, actual_columns))

        expected_index = X.index.to_numpy()
        actual_index = scores.index.to_numpy()
        self.assertTrue(np.array_equal(expected_index, actual_index))


class TestAdaBoostFit(unittest.TestCase):

    def test_when_fit_with_mismatched_lengths_then_raises_runtime_error(self):
        # Arrange
        X, y = make_simple_two_feature()
        y_shorter = y.iloc[:-1]
        ada = AdaBoostClassifier()

        # Act & Assert
        with self.assertRaises(RuntimeError):
            ada.fit(X, y_shorter)

    def test_when_fit_then_creates_expected_number_of_estimators_and_classes(
            self):
        # Arrange
        X, y = make_simple_two_feature()
        ada = AdaBoostClassifier(
            n_estimators=7,
            learning_rate=0.5,
            random_state=1,
        )

        # Act
        ada.fit(X, y)

        # Assert
        expected_estimators_len = 7
        actual_estimators_len = len(ada.estimators_)
        self.assertEqual(expected_estimators_len, actual_estimators_len)

        expected_weights_len = 7
        actual_weights_len = len(ada.estimator_weights_)
        self.assertEqual(expected_weights_len, actual_weights_len)

        expected_classes_not_none = True
        actual_classes_not_none = ada.classes_ is not None
        self.assertEqual(expected_classes_not_none, actual_classes_not_none)

        expected_unique_classes = len(np.unique(y))
        actual_unique_classes = len(ada.classes_)
        self.assertEqual(expected_unique_classes, actual_unique_classes)

    def test_when_fit_multiple_times_then_estimators_are_reset_each_time(self):
        # Arrange
        X, y = make_simple_two_feature()
        ada = AdaBoostClassifier(
            n_estimators=3,
            random_state=0,
        )
        ada.fit(X, y)
        first_estimators_ids = [id(est) for est in ada.estimators_]

        # Act
        ada.fit(X, y)
        second_estimators_ids = [id(est) for est in ada.estimators_]

        # Assert
        expected_len = 3
        actual_len = len(ada.estimators_)
        self.assertEqual(expected_len, actual_len)

        expected_reset = False
        actual_reset = first_estimators_ids == second_estimators_ids
        self.assertEqual(expected_reset, actual_reset)

    def test_when_fit_with_custom_base_estimator_and_params_then_respects_hyperparameters(
            self):
        # Arrange
        base_estimator = DecisionTreeClassifier(
            min_samples_leaf=2,
            min_samples_split=4,
            max_depth=2,
            criterion="entropy",
        )
        X, y = make_simple_two_feature()
        ada = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=4,
            learning_rate=0.7,
            random_state=2,
        )

        # Act
        ada.fit(X, y)

        # Assert
        expected_len = 4
        actual_len = len(ada.estimators_)
        self.assertEqual(expected_len, actual_len)

        for est in ada.estimators_:
            expected_type = DecisionTreeClassifier
            actual_type = type(est)
            self.assertEqual(expected_type, actual_type)

            expected_min_samples_leaf = 2
            actual_min_samples_leaf = est.min_samples_leaf
            self.assertEqual(expected_min_samples_leaf,
                             actual_min_samples_leaf)

            expected_min_samples_split = 4
            actual_min_samples_split = est.min_samples_split
            self.assertEqual(expected_min_samples_split,
                             actual_min_samples_split)

            expected_max_depth = 2
            actual_max_depth = est.max_depth
            self.assertEqual(expected_max_depth, actual_max_depth)


class TestAdaBoostPredict(unittest.TestCase):

    def test_when_predict_before_fit_then_raises_runtime_error(self):
        # Arrange
        X, _ = make_simple_two_feature()
        ada = AdaBoostClassifier()

        # Act & Assert
        with self.assertRaises(RuntimeError):
            ada.predict(X)

    def test_when_predict_after_fit_then_returns_series_with_correct_length(
            self):
        # Arrange
        X, y = make_simple_two_feature()
        ada = AdaBoostClassifier(
            n_estimators=5,
            learning_rate=0.8,
            random_state=3,
        ).fit(X, y)

        # Act
        y_pred = ada.predict(X)

        # Assert
        expected_type = pd.Series
        actual_type = type(y_pred)
        self.assertEqual(expected_type, actual_type)

        expected_len = len(X)
        actual_len = len(y_pred)
        self.assertEqual(expected_len, actual_len)

    def test_when_predict_with_same_random_state_then_predictions_are_deterministic(
            self):
        # Arrange
        X, y = make_simple_two_feature()

        ada1 = AdaBoostClassifier(
            n_estimators=10,
            learning_rate=0.9,
            random_state=42,
        ).fit(X, y)

        ada2 = AdaBoostClassifier(
            n_estimators=10,
            learning_rate=0.9,
            random_state=42,
        ).fit(X, y)

        # Act
        y_pred_1 = ada1.predict(X)
        y_pred_2 = ada2.predict(X)

        # Assert
        expected_equal = True
        actual_equal = np.array_equal(y_pred_1.to_numpy(), y_pred_2.to_numpy())
        self.assertEqual(expected_equal, actual_equal)


class TestAdaBoostScore(unittest.TestCase):

    def test_when_score_on_training_data_then_between_zero_and_one(self):
        # Arrange
        X, y = make_simple_two_feature()
        ada = AdaBoostClassifier(
            n_estimators=5,
            learning_rate=1.0,
            random_state=0,
        ).fit(X, y)

        # Act
        score = ada.score(X, y)

        # Assert
        expected_min = 0.0
        expected_max = 1.0
        actual = score
        self.assertGreaterEqual(actual, expected_min)
        self.assertLessEqual(actual, expected_max)

    def test_when_score_uses_predict_results(self):
        # Arrange
        X, y = make_simple_two_feature()
        ada = AdaBoostClassifier(
            n_estimators=3,
            learning_rate=0.5,
            random_state=1,
        ).fit(X, y)

        # Act
        score_direct = ada.score(X, y)
        y_pred = ada.predict(X)
        manual_score = (y_pred == y).mean()

        # Assert
        expected_score = manual_score
        actual_score = score_direct
        self.assertAlmostEqual(expected_score, actual_score)


if __name__ == "__main__":
    unittest.main()

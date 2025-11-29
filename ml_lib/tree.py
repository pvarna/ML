import pandas as pd
import numpy as np
from typing import Optional, Union, List

from .stats import gini_index, entropy


class Node:

    def __init__(self,
                 feature: Optional[int] = None,
                 threshold: Optional[float] = None,
                 left: Optional['Node'] = None,
                 right: Optional['Node'] = None,
                 value: Optional[int] = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeClassifier:

    def __init__(self,
                 min_samples_leaf: int,
                 min_samples_split: int,
                 max_depth: Optional[int] = None,
                 criterion: str = "gini"):
        if criterion not in ['gini', 'entropy']:
            raise RuntimeError("Unsupported criterion type")

        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = gini_index if criterion == 'gini' else entropy

        self.root: Optional[Node] = None

    def _majority_class(self, y: pd.Series) -> int:
        return y.value_counts().idxmax()

    def _impurity(self, y: pd.Series) -> float:
        ps = y.value_counts(normalize=True).to_numpy()
        return self.criterion(ps)

    def _best_split(self, X: pd.DataFrame, y: pd.Series):
        num_samples, num_features = X.shape
        parent_impurity = self._impurity(y)

        best_feature = None
        best_threshold = None
        best_information_gain = 0.0

        for feature in range(num_features):
            values = X.iloc[:, feature].unique()

            for threshold in values:
                left_mask = X.iloc[:, feature] <= threshold
                right_mask = X.iloc[:, feature] > threshold

                if left_mask.sum() < self.min_samples_leaf:
                    continue
                if right_mask.sum() < self.min_samples_leaf:
                    continue

                left_impurity = self._impurity(y[left_mask])
                right_impurity = self._impurity(y[right_mask])

                n = num_samples
                impurity_after_split = (left_mask.sum() / n) * left_impurity + \
                                       (right_mask.sum() / n) * right_impurity

                information_gain = parent_impurity - impurity_after_split

                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_information_gain

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int) -> Node:
        num_samples, num_features = X.shape

        if num_samples < self.min_samples_split:
            return Node(value=self._majority_class(y))

        if self.max_depth is not None and self.max_depth <= depth:
            return Node(value=self._majority_class(y))

        if len(y.unique()) == 1:
            return Node(value=y.iloc[0])

        feature, threshold, information_gain = self._best_split(X, y)

        if information_gain <= 0.0:
            return Node(value=self._majority_class(y))

        left_mask = X.iloc[:, feature] <= threshold
        right_mask = X.iloc[:, feature] > threshold

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=feature,
                    threshold=threshold,
                    left=left_child,
                    right=right_child)

    def _predict_sample(self, node: Node, sample: pd.Series) -> int:
        while not node.is_leaf():
            if sample.iloc[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.value

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.root = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X.apply(lambda sample: self._predict_sample(self.root, sample),
                       axis=1)


class RandomForestClassifier:

    def __init__(self,
                 n_estimators: int = 100,
                 criterion: str = "gini",
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Union[str, int, float, None] = "sqrt",
                 bootstrap: bool = True,
                 random_state: Optional[int] = None,
                 oob_score: bool = False):
        if n_estimators <= 0:
            raise RuntimeError("n_estimators must be positive")

        if criterion not in ['gini', 'entropy']:
            raise RuntimeError("Unsupported criterion type")

        if max_features not in ['sqrt', 'log2', None
                                ] and not isinstance(max_features,
                                                     (int, float)):
            raise RuntimeError("Unsupported max_features type")

        if oob_score and not bootstrap:
            raise RuntimeError(
                "OOB score can only be computed if bootstrap is True")

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.oob_score = oob_score

        self.trees_: List[DecisionTreeClassifier] = []
        self.features_per_tree_: List[np.ndarray] = []

        self.n_features_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self.oob_score_: Optional[float] = None

        self._rng = np.random.RandomState(random_state)

    def _get_features_count(self, n_features: int) -> int:
        if self.max_features is None:
            return n_features

        if isinstance(self.max_features, int):
            return max(1, min(self.max_features, n_features))

        if isinstance(self.max_features, float):
            if not (0.0 < self.max_features <= 1.0):
                raise RuntimeError(
                    "max_features as float must be in (0.0, 1.0]")
            return max(1, min(int(self.max_features * n_features), n_features))

        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))

        if self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))

        raise RuntimeError("Unsupported max_features type")

    def _bootstrap_sample_indices(self, n_samples: int) -> np.ndarray:
        if self.bootstrap:
            return self._rng.randint(0, n_samples, size=n_samples)
        else:
            return np.arange(n_samples)

    def _compute_oob_score(self, y: pd.Series,
                           oob_sample_indices: List[np.ndarray],
                           oob_predictions: List[np.ndarray]):
        n_samples = len(y)

        votes_per_sample: List[List[int]] = [[] for _ in range(n_samples)]

        for indices, preds in zip(oob_sample_indices, oob_predictions):
            for idx, pred in zip(indices, preds):
                votes_per_sample[idx].append(pred)

        y_oob_true = []
        y_oob_pred = []

        for i in range(n_samples):
            votes = votes_per_sample[i]
            if len(votes) == 0:
                continue

            votes_series = pd.Series(votes)
            majority_vote = votes_series.value_counts().idxmax()

            y_oob_true.append(y.iloc[i])
            y_oob_pred.append(majority_vote)

        if len(y_oob_true) == 0:
            self.oob_score_ = None
        else:
            y_oob_true_series = pd.Series(y_oob_true)
            y_oob_pred_series = pd.Series(y_oob_pred)
            self.oob_score_ = (y_oob_true_series == y_oob_pred_series).mean()

    def _predict_single_tree(self, tree: DecisionTreeClassifier,
                             feature_indices: np.ndarray,
                             X: pd.DataFrame) -> pd.Series:
        X_subset = X.iloc[:, feature_indices]
        return tree.predict(X_subset)

    def _majority_vote(self, sample_preds: pd.Series) -> int:
        return sample_preds.value_counts().idxmax()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if len(X) != len(y):
            raise RuntimeError("X and y must have the same number of samples")

        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.classes_ = np.unique(y)

        self.trees_ = []
        self.features_per_tree_ = []

        oob_predictions: List[np.ndarray] = []
        oob_sample_indices: List[np.ndarray] = []

        features_per_tree = self._get_features_count(n_features)

        for _ in range(self.n_estimators):
            sample_indices = self._bootstrap_sample_indices(n_samples)
            sample_indices.sort()

            X_sample = X.iloc[sample_indices]
            y_sample = y.iloc[sample_indices]

            feature_indices = self._rng.choice(n_features,
                                               size=features_per_tree,
                                               replace=False)
            feature_indices.sort()

            X_sample_tree = X_sample.iloc[:, feature_indices]

            tree = DecisionTreeClassifier(
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                criterion=self.criterion)
            tree.fit(X_sample_tree, y_sample)

            self.trees_.append(tree)
            self.features_per_tree_.append(feature_indices)

            if self.oob_score and self.bootstrap:
                all_indices = np.arange(n_samples)
                oob_indices = np.setdiff1d(all_indices, sample_indices)

                if oob_indices.size > 0:
                    X_oob = X.iloc[oob_indices, feature_indices]
                    oob_pred = tree.predict(X_oob).to_numpy()
                    oob_sample_indices.append(oob_indices)
                    oob_predictions.append(oob_pred)

        if self.oob_score and self.bootstrap:
            self._compute_oob_score(y, oob_sample_indices, oob_predictions)

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self.trees_:
            raise RuntimeError("The model has not been fitted yet")

        all_tree_predictions = []

        for tree, feature_indices in zip(self.trees_, self.features_per_tree_):
            preds = self._predict_single_tree(tree, feature_indices, X)
            all_tree_predictions.append(preds)

        preds_df = pd.concat(all_tree_predictions, axis=1)

        return preds_df.apply(self._majority_vote, axis=1)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        y_pred = self.predict(X)
        return (y_pred == y).mean()


class AdaBoostClassifier:

    def __init__(self,
                  estimator=None,
                  n_estimators: int = 50,
                  learning_rate: float = 1.0,
                  random_state: Optional[int] = None):
        if n_estimators <= 0:
            raise RuntimeError("n_estimators must be positive")
        
        if learning_rate <= 0.0:
            raise RuntimeError("learning_rate must be positive")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.estimator = estimator if estimator is not None else DecisionTreeClassifier(
            min_samples_leaf=1,
            min_samples_split=2,
            max_depth=1,
            criterion="gini")

        self.estimators_: List = []
        self.estimator_weights_: List[float] = []

        self.classes_: Optional[np.ndarray] = None

        self._rng = np.random.RandomState(random_state)

    def _get_fresh_estimator(self):
        criterion_str = ""
        if self.estimator.criterion is gini_index:
            criterion_str = "gini"
        elif self.estimator.criterion is entropy:
            criterion_str = "entropy"
        else:
            raise RuntimeError("Unknown criterion in base estimator")
        
        return DecisionTreeClassifier(
            min_samples_leaf=self.estimator.min_samples_leaf,
            min_samples_split=self.estimator.min_samples_split,
            max_depth=self.estimator.max_depth,
            criterion=criterion_str)

    def _predict_score(self, X: pd.DataFrame) -> pd.DataFrame:
        scores = pd.DataFrame(0.0, index=X.index, columns=self.classes_)

        for estimator, alpha in zip(self.estimators_, self.estimator_weights_):
            preds = estimator.predict(X)
            for cls in self.classes_:
                scores.loc[preds == cls, cls] += alpha

        return scores

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if len(X) != len(y):
            raise RuntimeError("X and y must have the same number of samples")

        n_samples = len(y)
        self.classes_ = np.unique(y)

        sample_weights = np.ones(n_samples) / n_samples

        self.estimators_ = []
        self.estimator_weights_ = []

        for _ in range(self.n_estimators):
            indices = self._rng.choice(n_samples,
                                       size=n_samples,
                                       replace=True,
                                       p=sample_weights)

            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]

            estimator = self._get_fresh_estimator()
            estimator.fit(X_sample, y_sample)

            y_pred = estimator.predict(X)

            incorrect = (y_pred != y).to_numpy()
            estimator_error = np.sum(sample_weights * incorrect)

            estimator_error = np.clip(estimator_error, 1e-10, 1 - 1e-10)

            alpha = self.learning_rate * np.log(
                (1 - estimator_error) / estimator_error)

            sample_weights *= np.exp(alpha * incorrect)
            sample_weights /= np.sum(sample_weights)

            self.estimators_.append(estimator)
            self.estimator_weights_.append(alpha)

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self.estimators_:
            raise RuntimeError("The model has not been fitted yet")

        scores = self._predict_score(X)
        return scores.idxmax(axis=1)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        y_pred = self.predict(X)
        return (y_pred == y).mean()

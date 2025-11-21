import pandas as pd
from typing import Optional

from stats import gini_index, entropy


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

    def __init__(self, min_samples_leaf: int, min_samples_split: int,
                 max_depth: int, criterion: str):
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

        if self.max_depth <= depth:
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

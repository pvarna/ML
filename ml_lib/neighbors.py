import numpy as np
import pandas as pd

from ml_lib.metrics import euclidean_distance, manhattan_distance, accuracy_score


class KNeighborsClassifier:

    def __init__(self, n_neighbors: int, metric: str):
        if n_neighbors < 1:
            raise RuntimeError("N Neigbors is a non-positive number")

        if metric not in {"euclidean", "manhattan"}:
            raise RuntimeError("Metric is neither euclidean, nor manhattan")

        self.n_neighbors = n_neighbors
        self.metric = metric
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if len(X) != len(y):
            raise RuntimeError(
                "X and y don't have the same number of examples")

        self.X_train = X.copy()
        self.y_train = y.copy()
        self._fitted = True

        return self

    def _calculate_distances(self, x: pd.Series) -> np.ndarray:
        if self.metric == "euclidean":
            distances = self.X_train.apply(
                lambda row: euclidean_distance(row.to_numpy(), x.to_numpy()),
                axis=1)
        else:
            distances = self.X_train.apply(
                lambda row: euclidean_distance(row.to_numpy(), x.to_numpy()),
                axis=1)
        return distances.to_numpy()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("The model is not fitted yet")

        predictions = []

        for _, x in X.iterrows():
            distances = self._calculate_distances(x)
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_labels = self.y_train.iloc[nearest_indices]

            values, counts = np.unique(nearest_labels, return_counts=True)
            majority_label = values[np.argmax(counts)]
            predictions.append(majority_label)

        return pd.Series(predictions, index=X.index)

    def score(self, X: pd.DataFrame, y_true: pd.Series) -> float:
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)

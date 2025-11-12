import numpy as np
import pandas as pd
from typing import Optional, Any, Dict, Tuple

DataPair = Tuple[pd.DataFrame, pd.Series]
StratifyGroups = Dict[Any, DataPair]
SplitGroups = Dict[Any, Tuple[DataPair, DataPair]]


def _validate_train_test_sizes(train_size: Optional[float],
                               test_size: Optional[float]):
    if test_size is None and train_size is None:
        raise RuntimeError("Train size and test size should not be both None")
    if train_size is None:
        train_size = 1 - test_size
    if test_size is None:
        test_size = 1 - train_size
    if not (0 < train_size < 1):
        raise RuntimeError("Train size should be in the interval (0, 1)")
    if not (0 < test_size < 1):
        raise RuntimeError("Test size should be in the interval (0, 1)")
    if abs(train_size + test_size - 1.0) > 0.00001:
        raise RuntimeError("The sum of train size and test size should be 1")

    return float(train_size), float(test_size)


def _validate_X_y_sizes(X: pd.DataFrame, y: pd.Series):
    if len(X) < 2:
        raise RuntimeError("Can't split fewer than 2 items")
    if len(X) != len(y):
        raise RuntimeError("X and y have different length")


def _validate_stratify_parameter(stratify: Optional[pd.Series],
                                 X: pd.DataFrame, y: pd.Series):
    if stratify is not None:
        if not (stratify.equals(y)
                or any(stratify.equals(X[col]) for col in X.columns)):
            raise RuntimeError("Stratify is neither y, nor any column from X")


def _get_stratify_groups(X: pd.DataFrame, y: pd.Series,
                         stratify: pd.Series) -> StratifyGroups:
    groups = {}
    for stratify_class in stratify.unique():
        column_filter = stratify == stratify_class
        groups[stratify_class] = (X[column_filter].copy(),
                                  y[column_filter].copy())

    return groups


def _split_stratify_groups(stratify_groups: StratifyGroups, test_size: float,
                           shuffle: bool,
                           random_state: Optional[int]) -> SplitGroups:
    split_groups = {}
    rng = None
    if shuffle:
        rng = np.random.default_rng(seed=random_state)

    for stratify_class, (X, y) in stratify_groups.items():
        assert (len(X) == len(y))
        count = len(X)

        if count < 2:
            raise RuntimeError("Stratify split group has fewer than 2 items")

        test_count = int(np.ceil(count * test_size))
        test_count = min(max(1, test_count), count - 1)

        idx = X.index.to_numpy()

        if rng is not None:
            perm = rng.permutation(idx)
            test_idx = perm[:test_count]
            train_idx = perm[test_count:]
        else:
            test_idx = idx[-test_count:]
            train_idx = idx[:-test_count]

        X_train, y_train = X.loc[train_idx].copy(), y.loc[train_idx].copy()
        X_test, y_test = X.loc[test_idx].copy(), y.loc[test_idx].copy()

        split_groups[stratify_class] = ((X_train, y_train), (X_test, y_test))

    return split_groups


def _merge_split_groups(
        split_groups: SplitGroups) -> Tuple[DataPair, DataPair]:
    X_train_parts, y_train_parts, X_test_parts, y_test_parts = [], [], [], []

    for _, ((X_train, y_train), (X_test, y_test)) in split_groups.items():
        X_train_parts.append(X_train)
        y_train_parts.append(y_train)
        X_test_parts.append(X_test)
        y_test_parts.append(y_test)

    X_train = pd.concat(X_train_parts, axis=0)
    y_train = pd.concat(y_train_parts, axis=0)
    X_test = pd.concat(X_test_parts, axis=0)
    y_test = pd.concat(y_test_parts, axis=0)

    return (X_train, y_train), (X_test, y_test)


def _post_merge_shuffle(X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, y_test: pd.Series, shuffle: bool,
                        random_state: Optional[int]):
    if shuffle:
        rng = np.random.default_rng(seed=random_state)
        train_idx = rng.permutation(X_train.index)
        test_idx = rng.permutation(X_test.index)
        X_train, y_train = X_train.loc[train_idx], y_train.loc[train_idx]
        X_test, y_test = X_test.loc[test_idx], y_test.loc[test_idx]

    return X_train, X_test, y_train, y_test


def _build_groups(X: pd.DataFrame, y: pd.Series,
                  stratify: Optional[pd.Series]) -> StratifyGroups:
    if stratify is None:
        return {"__ALL__": (X.copy(), y.copy())}
    return _get_stratify_groups(X, y, stratify)


def train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: Optional[float] = 0.25,
    train_size: Optional[float] = None,
    shuffle: bool = True,
    random_state: Optional[int] = None,
    stratify: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    _validate_X_y_sizes(X, y)
    train_size, test_size = _validate_train_test_sizes(train_size, test_size)
    _validate_stratify_parameter(stratify, X, y)

    stratify_groups = _build_groups(X, y, stratify)
    split_groups = _split_stratify_groups(stratify_groups, test_size, shuffle,
                                          random_state)
    (X_train, y_train), (X_test, y_test) = _merge_split_groups(split_groups)

    X_train, X_test, y_train, y_test = _post_merge_shuffle(
        X_train, X_test, y_train, y_test, shuffle, random_state)

    return X_train, X_test, y_train, y_test

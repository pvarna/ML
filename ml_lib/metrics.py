import numpy as np
import pandas as pd
from typing import List
from math import sqrt
import warnings


def accuracy_score(y_true: pd.Series,
                   y_pred: pd.Series,
                   normalize: bool = True) -> int | float:
    if len(y_true) != len(y_pred):
        raise RuntimeError(
            "Sizes of correct and predicted labels are different")

    correct = (y_true == y_pred).sum()
    return correct / len(y_true) if normalize else correct


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape != y.shape:
        raise RuntimeError("Points have different dimensions")

    return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))


def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape != y.shape:
        raise RuntimeError("Points have different dimensions")

    return np.sum(np.abs(x - y))


def r2_score(y_true: List[float], y_pred: List[float]) -> float:
    if len(y_true) != len(y_pred):
        raise RuntimeError(
            "Sizes of correct and predicted labels are different")

    mean_y = sum(y_true) / len(y_true)
    numerator = sum((y_t - y_p)**2 for y_t, y_p in zip(y_true, y_pred))
    denominator = sum((y - mean_y)**2 for y in y_true)

    if abs(denominator) < 0.00001:
        return 0.0

    return 1 - numerator / denominator


def root_mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    if len(y_true) != len(y_pred):
        raise RuntimeError(
            "Sizes of correct and predicted labels are different")

    n = len(y_true)
    mse = sum((y_t - y_p)**2 for y_t, y_p in zip(y_true, y_pred)) / n

    return sqrt(mse)


def _validate_labels_arg(labels: List[int | float | bool | str],
                         y_true: pd.Series, y_pred: pd.Series):
    if labels == None:
        return sorted(set(pd.unique(y_true)) | set(pd.unique(y_pred)))
    if len(labels) == 0:
        raise RuntimeError("labels must be non-empty")
    if len(set(labels)) != len(labels):
        raise RuntimeError("labels must not contain duplicates")

    known = set(pd.unique(y_true)) | set(pd.unique(y_pred))
    unknown = [lab for lab in labels if lab not in known]
    if unknown:
        raise RuntimeError(f"Unknown label(s) in labels: {unknown}")

    return labels


def _safe_div(zero_division: str | float) -> float:
    if zero_division == "warn":
        warnings.warn(f"Denominator is 0; returning 0.0")
        return 0.0
    return float(zero_division)


def _binary_recall(y_true: pd.Series,
                   y_pred: pd.Series,
                   pos_label: int | float | bool | str,
                   zero_division: str | float = "warn") -> float:
    tp = ((y_true == pos_label) & (y_pred == pos_label)).sum()
    fn = ((y_true == pos_label) & (y_pred != pos_label)).sum()
    denom = tp + fn

    result = float(tp) / float(denom) if denom != 0 else _safe_div(
        zero_division)
    return result


def recall_score(y_true: pd.Series,
                 y_pred: pd.Series,
                 labels: List[int | float | bool | str] = None,
                 pos_label: int | float | bool | str = 1,
                 average: str = "binary",
                 zero_division: str | float = "warn") -> float | List[float]:
    if len(y_true) != len(y_pred):
        raise RuntimeError(
            "Sizes of correct and predicted labels are different")

    allowed = {"binary", "micro", "macro", "weighted", "samples", None}
    if average not in allowed:
        raise RuntimeError("Invalid average")
    if not (zero_division == "warn" or isinstance(zero_division,
                                                  (int, float))):
        raise RuntimeError("Zero division is neither 'warn' nor a number")

    if average == "binary":
        uniq = set(pd.unique(y_true)) | set(pd.unique(y_pred))
        if len(uniq) > 2:
            raise RuntimeError("Average is binary and targets are not binary")
        if y_true.nunique() == 2 and pos_label not in y_true.values:
            raise RuntimeError("Given pos_label not present in y_true")

        return _binary_recall(y_true, y_pred, pos_label, zero_division)

    if average == "micro":
        denom = len(y_true)
        tp_global = (y_true == y_pred).sum()
        result = float(tp_global) / float(denom) if denom != 0 else _safe_div(
            zero_division)
        return result

    if average == "macro":
        labels = _validate_labels_arg(labels, y_true, y_pred)

        recalls = [
            _binary_recall(y_true, y_pred, label, zero_division)
            for label in labels
        ]
        return float(np.mean(recalls))

    if average == "weighted":
        labels = _validate_labels_arg(labels, y_true, y_pred)

        supports = [(y_true == label).sum() for label in labels]
        recalls = [
            _binary_recall(y_true, y_pred, label, zero_division)
            for label in labels
        ]

        denom = int(np.sum(supports))

        result = float(np.average(
            recalls,
            weights=supports)) if denom != 0 else _safe_div(zero_division)
        return result

    if average is None:
        labels = _validate_labels_arg(labels, y_true, y_pred)

        recalls = [
            _binary_recall(y_true, y_pred, label, zero_division)
            for label in labels
        ]

        return recalls

    raise RuntimeError("average='samples' is not yet implemented")


def _binary_precision(y_true: pd.Series,
                      y_pred: pd.Series,
                      pos_label: int | float | bool | str,
                      zero_division: str | float = "warn") -> float:
    tp = ((y_true == pos_label) & (y_pred == pos_label)).sum()
    fp = ((y_true != pos_label) & (y_pred == pos_label)).sum()
    denom = tp + fp

    result = float(tp) / float(denom) if denom != 0 else _safe_div(
        zero_division)
    return result


def precision_score(
        y_true: pd.Series,
        y_pred: pd.Series,
        labels: List[int | float | bool | str] = None,
        pos_label: int | float | bool | str = 1,
        average: str = "binary",
        zero_division: str | float = "warn") -> float | List[float]:
    if len(y_true) != len(y_pred):
        raise RuntimeError(
            "Sizes of correct and predicted labels are different")

    allowed = {"binary", "micro", "macro", "weighted", "samples", None}
    if average not in allowed:
        raise RuntimeError("Invalid average")
    if not (zero_division == "warn" or isinstance(zero_division,
                                                  (int, float))):
        raise RuntimeError("Zero division is neither 'warn' nor a number")

    if average == "binary":
        uniq = set(pd.unique(y_true)) | set(pd.unique(y_pred))
        if len(uniq) > 2:
            raise RuntimeError("Average is binary and targets are not binary")
        if y_true.nunique() == 2 and pos_label not in y_true.values:
            raise RuntimeError("Given pos_label not present in y_true")

        return _binary_precision(y_true, y_pred, pos_label, zero_division)

    if average == "micro":
        tp_global = (y_true == y_pred).sum()
        denom = len(y_pred)
        result = float(tp_global) / float(denom) if denom != 0 else _safe_div(
            zero_division)
        return result

    if average == "macro":
        labels = _validate_labels_arg(labels, y_true, y_pred)

        precisions = [
            _binary_precision(y_true, y_pred, label, zero_division)
            for label in labels
        ]
        return float(np.mean(precisions))

    if average == "weighted":
        labels = _validate_labels_arg(labels, y_true, y_pred)

        supports = [(y_true == label).sum() for label in labels]
        precisions = [
            _binary_precision(y_true, y_pred, label, zero_division)
            for label in labels
        ]

        denom = int(np.sum(supports))

        result = float(np.average(
            precisions,
            weights=supports)) if denom != 0 else _safe_div(zero_division)
        return result

    if average is None:
        labels = _validate_labels_arg(labels, y_true, y_pred)

        precisions = [
            _binary_precision(y_true, y_pred, label, zero_division)
            for label in labels
        ]

        return precisions

    raise RuntimeError("average='samples' is not yet implemented")


def _binary_f1(y_true: pd.Series,
               y_pred: pd.Series,
               pos_label: int | float | bool | str,
               zero_division: str | float = "warn") -> float:
    tp = ((y_true == pos_label) & (y_pred == pos_label)).sum()
    fp = ((y_true != pos_label) & (y_pred == pos_label)).sum()
    fn = ((y_true == pos_label) & (y_pred != pos_label)).sum()
    denom = 2 * tp + fp + fn
    return (2.0 * tp /
            float(denom)) if denom != 0 else _safe_div(zero_division)


def f1_score(y_true: pd.Series,
             y_pred: pd.Series,
             labels: List[int | float | bool | str] = None,
             pos_label: int | float | bool | str = 1,
             average: str = "binary",
             zero_division: str | float = "warn") -> float | List[float]:
    if len(y_true) != len(y_pred):
        raise RuntimeError(
            "Sizes of correct and predicted labels are different")

    allowed = {"binary", "micro", "macro", "weighted", "samples", None}
    if average not in allowed:
        raise RuntimeError("Invalid average")
    if not (zero_division == "warn" or isinstance(zero_division,
                                                  (int, float))):
        raise RuntimeError("Zero division is neither 'warn' nor a number")

    if average == "binary":
        uniq = set(pd.unique(y_true)) | set(pd.unique(y_pred))
        if len(uniq) > 2:
            raise RuntimeError("Average is binary and targets are not binary")
        if y_true.nunique() == 2 and pos_label not in y_true.values:
            raise RuntimeError("Given pos_label not present in y_true")
        return _binary_f1(y_true, y_pred, pos_label, zero_division)

    if average == "micro":
        tp_global = (y_true == y_pred).sum()
        denom = len(y_true)
        return float(tp_global) / float(denom) if denom != 0 else _safe_div(
            zero_division)

    if average == "macro":
        labels = _validate_labels_arg(labels, y_true, y_pred)
        f1s = [
            _binary_f1(y_true, y_pred, label, zero_division)
            for label in labels
        ]
        return float(np.mean(f1s))

    if average == "weighted":
        labels = _validate_labels_arg(labels, y_true, y_pred)
        supports = [(y_true == label).sum() for label in labels]
        f1s = [
            _binary_f1(y_true, y_pred, label, zero_division)
            for label in labels
        ]
        denom = int(np.sum(supports))
        return float(np.average(
            f1s, weights=supports)) if denom != 0 else _safe_div(zero_division)

    if average is None:
        labels = _validate_labels_arg(labels, y_true, y_pred)
        return [
            _binary_f1(y_true, y_pred, label, zero_division)
            for label in labels
        ]

    raise RuntimeError("average='samples' is not yet implemented")


def log_loss(y_true: pd.Series, y_pred: pd.Series) -> float:
    if len(y_true) != len(y_pred):
        raise RuntimeError(
            "Sizes of correct and predicted labels are different")

    EPS = 0.00001

    y_true = y_true.to_numpy()
    y_pred = y_pred.to_numpy()

    if np.isscalar(y_pred[0]) or np.ndim(y_pred[0]) == 0:
        p1 = np.clip(y_pred.astype(float), EPS, 1 - EPS)
        p_true = np.where(y_true == 1, p1, 1.0 - p1)
        return float(-np.mean(np.log(p_true)))

    probs = np.array([np.array(p, dtype=float) for p in y_pred], dtype=float)
    probs = np.clip(probs, EPS, 1 - EPS)
    p_true = probs[np.arange(len(y_true)), y_true]
    return float(-np.mean(np.log(p_true)))

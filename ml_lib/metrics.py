import numpy as np
import pandas as pd


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

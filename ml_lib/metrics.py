import numpy as np
import pandas as pd
from typing import List
from math import sqrt


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
    numerator = sum((y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_pred))
    denominator = sum((y - mean_y) ** 2 for y in y_true)

    if abs(denominator) < 0.00001:
        return 0.0
    
    return 1 - numerator / denominator

def root_mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    if len(y_true) != len(y_pred):
        raise RuntimeError(
            "Sizes of correct and predicted labels are different")
    
    n = len(y_true)
    mse = sum((y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_pred)) / n

    return sqrt(mse)

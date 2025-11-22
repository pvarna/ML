import numpy as np
import pandas as pd
from typing import List
from math import sqrt


def linear(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    if X.ndim != 2 or Y.ndim != 2:
        raise RuntimeError("X and Y must be 2D arrays")
    if X.shape[1] != Y.shape[1]:
        raise RuntimeError("X and Y must have same number of features")

    return X @ Y.T


def polynomial(X: np.ndarray,
               Y: np.ndarray,
               degree: int = 3,
               gamma: float = None,
               coef0: float = 1.0) -> np.ndarray:
    if X.ndim != 2 or Y.ndim != 2:
        raise RuntimeError("X and Y must be 2D arrays")
    if X.shape[1] != Y.shape[1]:
        raise RuntimeError("X and Y must have same number of features")

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    return (gamma * (X @ Y.T) + coef0)**degree


def rbf(X: np.ndarray, Y: np.ndarray, gamma: float = None) -> np.ndarray:
    if X.ndim != 2 or Y.ndim != 2:
        raise RuntimeError("X and Y must be 2D arrays")
    if X.shape[1] != Y.shape[1]:
        raise RuntimeError("X and Y must have same number of features")

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    result = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            diff = X[i] - Y[j]
            result[i, j] = np.exp(-gamma * np.dot(diff, diff))
    return result


def sigmoid(X: np.ndarray,
            Y: np.ndarray,
            gamma: float = None,
            coef0: float = 1.0) -> np.ndarray:
    if X.ndim != 2 or Y.ndim != 2:
        raise RuntimeError("X and Y must be 2D arrays")
    if X.shape[1] != Y.shape[1]:
        raise RuntimeError("X and Y must have same number of features")

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    return np.tanh(gamma * (X @ Y.T) + coef0)

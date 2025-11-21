import numpy as np
from numpy.typing import ArrayLike

def sigmoid(x: ArrayLike) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def softmax(x: ArrayLike) -> np.ndarray:
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

EPS = 1e-15

def gini_index(ps: np.ndarray) -> float:
    return 1 - np.sum(ps ** 2)

def entropy(ps: np.ndarray) -> float:
    return np.sum(-ps * np.log2(ps + EPS))
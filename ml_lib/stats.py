import numpy as np
from numpy.typing import ArrayLike

def sigmoid(x: ArrayLike) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def softmax(x: ArrayLike) -> np.ndarray:
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

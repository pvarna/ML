import numpy as np
import pandas as pd

from .kernels import linear, polynomial, rbf, sigmoid
import quadprog


class SVC:

    def __init__(self,
                 C: float,
                 kernel: str,
                 gamma: float = None,
                 degree: int = 3,
                 coef0: float = 1.0):
        if kernel not in ['linear', 'polynomial', 'rbf', 'sigmoid']:
            raise RuntimeError("Unsupported kernel type")

        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

        self.support_vectors_ = None

        self.intercept_ = None
        self.alphas_ = None
        self.classes_ = None

        self._y_mapped = None
        self._support_alphas = None
        self._support_y = None

    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        if self.kernel == "linear":
            return linear(X, Y)
        elif self.kernel == "polynomial":
            return polynomial(X, Y, degree=self.degree, coef0=self.coef0)
        elif self.kernel == "rbf":
            return rbf(X, Y, gamma=self.gamma)
        elif self.kernel == "sigmoid":
            return sigmoid(X, Y, gamma=self.gamma, coef0=self.coef0)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_np = X.to_numpy(dtype=float)
        y_np = y.to_numpy(dtype=float)

        n_samples, n_features = X_np.shape

        classes = np.unique(y_np)
        if len(classes) != 2:
            raise RuntimeError(
                "This SVC implementation supports only binary classification")
        self.classes_ = classes

        y_mapped = np.where(y_np == classes[0], -1, 1)
        self._y_mapped = y_mapped

        K = self._compute_kernel(X_np, X_np)

        EPS = 0.00001
        G = (y_mapped[:, np.newaxis] * y_mapped[np.newaxis, :]) * K
        G += EPS * np.eye(G.shape[0])
        G = np.asarray(G, dtype=float)

        a = np.ones(n_samples, dtype=float)

        C_eq = y_mapped.reshape(1, -1)
        b_eq = np.array([0.0], dtype=float)

        I = np.eye(n_samples, dtype=float)
        C_ge0 = I
        b_ge0 = np.zeros(n_samples, dtype=float)

        C_leC = -I
        b_leC = -self.C * np.ones(n_samples, dtype=float)

        C_full = np.vstack([C_eq, C_ge0, C_leC])
        b_full = np.hstack([b_eq, b_ge0, b_leC])

        C_qp = C_full.T

        meq = 1
        factorized = False

        alphas, _, _, _, _, _ = quadprog.solve_qp(G, a, C_qp, b_full, meq,
                                                  factorized)

        self.alphas_ = alphas

        eps = 1e-5
        support_vector_indices = np.where(alphas > eps)[0]

        self.support_ = support_vector_indices
        self.support_vectors_ = X_np[support_vector_indices]
        self._support_alphas = alphas[support_vector_indices]
        self._support_y = y_mapped[support_vector_indices]

        edge_mask = (alphas > eps) & (alphas < self.C - eps)
        edge_indices = np.where(edge_mask)[0]

        if len(edge_indices) == 0:
            raise RuntimeError("No support vectors found")

        y_alpha = alphas * y_mapped

        b_values = []
        for idx in edge_indices:
            decision_without_b = np.sum(y_alpha * K[:, idx])
            b_i = y_mapped[idx] - decision_without_b
            b_values.append(b_i)

        self.intercept_ = np.mean(b_values)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.support_vectors_ is None or self.intercept_ is None:
            raise RuntimeError("The model has not been fitted yet")

        X_np = X.to_numpy(dtype=float)
        K_sv_new = self._compute_kernel(self.support_vectors_, X_np)

        alpha_y_support = self._support_alphas * self._support_y
        decision_values = alpha_y_support @ K_sv_new + self.intercept_

        y_mapped_pred = np.where(decision_values >= 0, 1, -1)

        y_pred = np.where(y_mapped_pred == -1, self.classes_[0],
                          self.classes_[1])
        return pd.Series(y_pred, index=X.index)

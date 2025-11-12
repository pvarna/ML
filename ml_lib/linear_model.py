import numpy as np
import pandas as pd

from metrics import r2_score
from stats import sigmoid, softmax


class LinearRegression:

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_np = np.hstack([np.ones((X.shape[0], 1)), X.to_numpy()])
        y_np = y.to_numpy().reshape(-1, 1)

        X_T_and_X_product = X_np.T @ X_np
        if np.linalg.det(X_T_and_X_product) == 0:
            raise ValueError(
                "Matrix is not invertible. Please remove collinear features.")

        X_T_and_X_product_inverted = np.linalg.inv(X_T_and_X_product)
        beta = X_T_and_X_product_inverted @ X_np.T @ y_np

        self.intercept_ = beta[0, 0]
        self.coef_ = beta[1:, 0]

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("The model is not fitted yet")

        X_np = X.to_numpy()
        y_pred = self.intercept_ + X_np @ self.coef_

        return pd.Series(y_pred, index=X.index)

    def score(self, X: pd.DataFrame, y_true: pd.Series) -> float:
        y_pred = self.predict(X)
        return r2_score(y_true, y_pred)


class LogisticRegression:

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            C=1.0,
            lr=0.01,
            max_iter=1000,
            random_state=None):
        X_np = np.hstack([np.ones((X.shape[0], 1)), X.to_numpy()])
        y_np = y.to_numpy()

        n_samples, n_features = X_np.shape
        n_features = n_features - 1

        self.classes_ = np.unique(y_np)
        n_classes = len(self.classes_)

        rng = np.random.RandomState(random_state)

        all_weights = rng.normal(0, 0.01, (n_features + 1, n_classes))

        lambda_ = 1.0 / C
        m = n_samples

        for i, c in enumerate(self.classes_):
            y_binary = (y_np == c).astype(int).reshape(-1, 1)
            w = all_weights[:, i].reshape(-1, 1)

            for _ in range(max_iter):
                z = X_np @ w
                h = sigmoid(z)

                grad_loss = (1 / m) * (X_np.T @ (h - y_binary))

                grad_reg = (lambda_ / m) * w
                grad_reg[0, 0] = 0

                gradient = grad_loss + grad_reg

                w = w - lr * gradient

            all_weights[:, i] = w.flatten()

        self.intercept_ = all_weights[0, :]
        self.coef_ = all_weights[1:, :].T

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("The model is not fitted yet")

        X_np = X.to_numpy()

        z = X_np @ self.coef_.T + self.intercept_
        probas = softmax(z)

        return probas

    def predict(self, X: pd.DataFrame) -> pd.Series:
        probas = self.predict_proba(X)
        indices = np.argmax(probas, axis=1)
        predictions = self.classes_[indices]

        return pd.Series(predictions, index=X.index)

    def score(self, X: pd.DataFrame, y_true: pd.Series) -> float:
        y_pred = self.predict(X)
        return np.mean(y_pred.to_numpy() == y_true.to_numpy())


class Ridge:

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_np = np.hstack([np.ones((X.shape[0], 1)), X.to_numpy()])
        y_np = y.to_numpy().reshape(-1, 1)

        n_features_plus_intercept = X_np.shape[1]

        I_m = np.eye(n_features_plus_intercept)

        I_m[0, 0] = 0

        regularized_X_T_X = (X_np.T @ X_np) + (self.alpha * I_m)
        regularized_X_T_X_inv = np.linalg.inv(regularized_X_T_X)

        beta = regularized_X_T_X_inv @ X_np.T @ y_np

        self.intercept_ = beta[0, 0]
        self.coef_ = beta[1:, 0]

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("The model is not fitted yet")

        X_np = X.to_numpy()
        y_pred = self.intercept_ + X_np @ self.coef_

        return pd.Series(y_pred, index=X.index)

    def score(self, X: pd.DataFrame, y_true: pd.Series) -> float:
        y_pred = self.predict(X)
        return r2_score(y_true, y_pred)


class Lasso:

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, lr=0.01, max_iter=1000):
        X_np = np.hstack([np.ones((X.shape[0], 1)), X.to_numpy()])
        y_np = y.to_numpy().reshape(-1, 1)

        n_samples, n_features_plus_intercept = X_np.shape
        m = n_samples

        beta = np.zeros((n_features_plus_intercept, 1))

        for _ in range(max_iter):
            y_pred = X_np @ beta
            residuals = y_pred - y_np

            grad_loss = (2 / m) * (X_np.T @ residuals)

            grad_reg = self.alpha * np.sign(beta)
            grad_reg[0, 0] = 0

            gradient = grad_loss + grad_reg

            beta = beta - lr * gradient

        self.intercept_ = beta[0, 0]
        self.coef_ = beta[1:, 0]

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("The model is not fitted yet")

        X_np = X.to_numpy()
        y_pred = self.intercept_ + X_np @ self.coef_

        return pd.Series(y_pred, index=X.index)

    def score(self, X: pd.DataFrame, y_true: pd.Series) -> float:
        y_pred = self.predict(X)
        return r2_score(y_true, y_pred)

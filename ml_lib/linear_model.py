import numpy as np
import pandas as pd

from ml_lib.metrics import r2_score, root_mean_squared_error

class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None


    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_np = np.hstack([np.ones((X.shape[0], 1)), X.to_numpy()])
        y_np = y.to_numpy().reshape(-1, 1)
        
        X_T_and_X_product = X_np.T @ X_np
        if np.linalg.det(X_T_and_X_product) == 0:
            raise ValueError("Matrix is not invertible. Please remove collinear features.")
        
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
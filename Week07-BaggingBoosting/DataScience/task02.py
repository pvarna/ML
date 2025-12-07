import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from ml_lib.metrics import r2_adjusted_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

DATASET_PATH = os.path.join("DATA", "bike_sharing.csv")
OUTPUT_XLSX = "data_audit.xlsx"
ASSETS_DIR = "assets"
FEATURES = ["registered", "casual"]
TARGET = "count"
RANDOM_STATE = 21
TEST_SIZE = 0.2


def build_param_grid():
    return {
        "depth": [4, 6, 8],
        "learning_rate": [0.03, 0.1],
        "l2_leaf_reg": [1, 3, 5],
        "iterations": [300, 500]
    }


def fit_predict_and_score(model, model_name, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    r2_adj = r2_adjusted_score(y_test, y_pred, X_train.shape[1])

    print(f"{model_name} RMSE: {rmse:.4f}")
    print(f"{model_name} R^2 adjusted: {r2_adj:.4f}")

    plot_residuals(y_test, y_pred)


def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color="red")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals (y_true - y_pred)")
    plt.title(f"Residuals vs Predicted")
    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv(DATASET_PATH, delimiter=',')

    print(df.head())

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    regressor = LinearRegression()

    fit_predict_and_score(regressor, "Linear Regression", X_train, X_test,
                          y_train, y_test)
    
    # model_report - https://docs.google.com/spreadsheets/d/1cse5W3T0YuGj0-ukVKiFyh8E9f9plzHQPMZC3gBzR5o/edit?usp=sharing


if __name__ == '__main__':
    main()

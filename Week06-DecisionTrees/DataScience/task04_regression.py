import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
import matplotlib.pyplot as plt

DATASET_PATH = os.path.join("..", "..", "DATA", "auto.csv")
OUTPUT_XLSX = "data_audit.xlsx"
ASSETS_DIR = "assets"
TARGET = "mpg"
FEATURE_TO_ENCODE = "origin"
FEATURES = ["weight", "hp"]
RANDOM_STATE = 21
TEST_SIZE = 0.2


def encode_categorical(df, column):
    dummies = pd.get_dummies(df[column], drop_first=True, dtype=int)
    df = pd.concat([df, dummies], axis=1).drop(columns=[column])

    return df

def main():
    df = pd.read_csv(DATASET_PATH, delimiter=',')
    df = encode_categorical(df, FEATURE_TO_ENCODE)

    print(df.head())

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                             y,
                                             test_size=TEST_SIZE,
                                             random_state=RANDOM_STATE)
    
    estimators = [
        ("lr", LinearRegression()),
        ("ridge", Ridge()),
        ("lasso", Lasso(0.01)),
        ("dt", DecisionTreeRegressor()),
    ]

    regressor = VotingRegressor(estimators=estimators)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")

    residuals = y_test - y_pred
    plt.figure(figsize=(7, 4))
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.scatter(y_pred, residuals)
    plt.xlabel('Predicted Loudness')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.title('Residuals vs Predicted Values')
    plt.tight_layout()
    plt.show()

    # model_report - https://docs.google.com/spreadsheets/d/1jvu44vhKY_0GgcuLKlKdO3X5X7op71HPl5ep_HG60VY/edit?usp=sharing


if __name__ == '__main__':
    main()

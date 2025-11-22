import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

DATASET_PATH = os.path.join("..", "..", "DATA", "music_clean.csv")
TARGET = "loudness"
RANDOM_STATE = 21
TEST_SIZE = 0.2
FEATURES = [
    "energy", "acousticness", "instrumentalness", "genre", "valence",
    "danceability", "popularity"
]


def main():
    df = pd.read_csv(DATASET_PATH, delimiter=',', index_col=0)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    pipeline = Pipeline([('model', Ridge(random_state=RANDOM_STATE, alpha=0.0001))])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

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

    # model_report - https://docs.google.com/spreadsheets/d/1eZ76KCPt2xwSmXSrSKP9J8kmUFCqYxK5T5ltdGg0SpY/edit?usp=sharing


if __name__ == '__main__':
    main()

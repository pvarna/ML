import os
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

DATASET_PATH = os.path.join("DATA", "music_dirty.txt")
TARGET = "popularity"
FEATURES_TO_IGNORE = ["danceability"]
NON_NUMERIC_FEATURE = "genre"
RANDOM_STATE = 21
TEST_SIZE = 0.2
MAX_ITER = 5000
BEST_ALPHA = 15


def create_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)

    return pd.DataFrame(data)


def encode_categorical(df, column):
    dummies = pd.get_dummies(df[column], drop_first=True, dtype=int)
    df = pd.concat([df, dummies], axis=1).drop(columns=[column])

    return df


def build_scorers():
    return {}


def main():
    df = create_dataset(DATASET_PATH)
    df = encode_categorical(df, NON_NUMERIC_FEATURE)

    X = df.drop(columns=[TARGET] + FEATURES_TO_IGNORE)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    ridge_pipeline = Pipeline([("scaler", StandardScaler()),
                               ("ridge",
                                Ridge(alpha=15,
                                      random_state=RANDOM_STATE,
                                      max_iter=MAX_ITER))])
    ridge_pipeline.fit(X_train, y_train)
    y_pred_test = ridge_pipeline.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = root_mean_squared_error(y_test, y_pred_test)
    print(f"Ridge alpha=15 Test R^2: {test_r2:.4f}")
    print(f"Ridge alpha=15 Test RMSE: {test_rmse:.4f}")

    residuals = y_test - y_pred_test
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred_test, residuals, alpha=0.6)
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel('Predicted popularity')
    plt.ylabel('Residual (y_true - y_pred)')
    plt.title('Residuals around 0 (Ridge alpha=15)')
    plt.tight_layout()
    plt.show()

    # model_report - https://docs.google.com/spreadsheets/d/1y8cNUGIYnAT4wwONkP4BTzGItrFdzkOJCtrx_KSENYw/edit?gid=0#gid=0


if __name__ == '__main__':
    main()

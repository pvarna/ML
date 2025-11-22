import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

DATASET_PATH = os.path.join("..", "..", "DATA", "music_clean.csv")
TARGET = "loudness"
RANDOM_STATE = 21
TEST_SIZE = 0.2


def main():
    df = pd.read_csv(DATASET_PATH, delimiter=',', index_col=0)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    _, _, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    baseline_mean = y_train.mean()
    y_pred = pd.Series(baseline_mean, index=y_test.index)

    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Baseline Mean Prediction: {baseline_mean:.4f}")
    print(f"Baseline RMSE: {rmse:.4f}")
    print(f"Baseline R^2: {r2:.4f}")

if __name__ == '__main__':
    main()

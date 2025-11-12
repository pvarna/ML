import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, root_mean_squared_error)

DATASET_PATH = os.path.join("..", "..", "DATA", "diabetes_clean.csv")
TARGET = "glucose"
FEATURES_TO_IGNORE = []
RANDOM_STATE = 21
TEST_SIZE = 0.2


def main():
    df = pd.read_csv(DATASET_PATH)
    X = df.drop(columns=FEATURES_TO_IGNORE + [TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    mean_y = y_train.mean()
    y_pred_baseline = np.full_like(y_test, fill_value=mean_y)

    baseline_r2 = r2_score(y_test, y_pred_baseline)
    baseline_rmse = root_mean_squared_error(y_test, y_pred_baseline)

    print(f"Mean target value: {mean_y}")
    print(f"Baseline R^2: {baseline_r2:.4f}")
    print(f"Baseline RMSE: {baseline_rmse:.4f}")


if __name__ == "__main__":
    main()

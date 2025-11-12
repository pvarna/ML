import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ml_lib.metrics import r2_score, root_mean_squared_error
from ml_lib.model_selection import train_test_split
from ml_lib.linear_model import Ridge

DATASET_PATH = os.path.join("DATA", "diabetes_clean.csv")
TARGET = "glucose"
FEATURES_TO_IGNORE = ["pregnancies", "triceps"]
RANDOM_STATE = 21
TEST_SIZE = 0.2
CV_FOLDS = 5
MAX_ITER = 5000


def main():
    df = pd.read_csv(DATASET_PATH)
    X = df.drop(columns=FEATURES_TO_IGNORE + [TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    ridge = Ridge(10).fit(X_train, y_train)
    y_pred = ridge.predict(X_test)

    print(f"R^2: {r2_score(y_test.tolist(), y_pred.tolist())}")
    print(f"RMSE: {root_mean_squared_error(y_test.tolist(), y_pred.tolist())}")


if __name__ == "__main__":
    main()

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ml_lib.metrics import (
    f1_score,
    recall_score,
    precision_score,
)

from ml_lib.model_selection import train_test_split
from ml_lib.linear_model import LogisticRegression

DATASET_PATH = os.path.join("DATA", "diabetes_clean.csv")
TARGET = "diabetes"
FEATURES_TO_IGNORE = []
RANDOM_STATE = 21
TEST_SIZE = 0.2
MAX_ITER = 5000


def main():
    df = pd.read_csv(DATASET_PATH)
    X = df.drop(columns=FEATURES_TO_IGNORE + [TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    lr = LogisticRegression().fit(X_train,
                                  y_train,
                                  C=10,
                                  lr=0.01,
                                  max_iter=MAX_ITER,
                                  random_state=RANDOM_STATE)
    y_pred = lr.predict(X_test)

    print(f"F1 score: {f1_score(y_test, y_pred)}")
    print(f"Recall score: {recall_score(y_test, y_pred)}")
    print(f"Precision score: {precision_score(y_test, y_pred)}")


if __name__ == "__main__":
    main()

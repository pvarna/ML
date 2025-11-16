import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATASET_PATH = os.path.join("..", "..", "DATA", "music_clean.csv")
TARGET = "is_popular"
FEATURE_TO_TRANSFORM = "popularity"
RANDOM_STATE = 21
TEST_SIZE = 0.2


def main():
    df = pd.read_csv(DATASET_PATH, delimiter=',', index_col=0)

    popularity_mean = df["popularity"].mean()
    df["is_popular"] = (df["popularity"] > popularity_mean).astype(int)
    df = df.drop(columns="popularity")
    print(df.head())

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    _, _, y_train, y_test = train_test_split(X,
                                             y,
                                             test_size=TEST_SIZE,
                                             random_state=RANDOM_STATE)

    baseline_mode = y_train.mode()[0]
    y_pred = pd.Series(baseline_mode, index=y_test.index)

    report = classification_report(y_test, y_pred, zero_division=0, digits=4)
    
    print(f"Baseline Mode Prediction: {baseline_mode:.4f}")
    print("Baseline Classification Report:")
    print(report)


if __name__ == '__main__':
    main()

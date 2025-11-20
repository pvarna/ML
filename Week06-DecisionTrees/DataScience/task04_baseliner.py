import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

DATASET_PATH = os.path.join("..", "..", "DATA", "auto.csv")
OUTPUT_XLSX = "data_audit.xlsx"
ASSETS_DIR = "assets"
TARGET = "mpg"
FEATURE_TO_ENCODE = "origin"
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

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    _, _, y_train, y_test = train_test_split(X,
                                             y,
                                             test_size=TEST_SIZE,
                                             random_state=RANDOM_STATE)

    baseline_mean = y_train.mean()
    y_pred = pd.Series(baseline_mean, index=y_test.index)

    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Baseline Mean Prediction: {baseline_mean:.4f}")
    print(f"Baseline RMSE: {rmse:.4f}")
    print(f"Baseline R^2: {r2:.4f}")


if __name__ == '__main__':
    main()

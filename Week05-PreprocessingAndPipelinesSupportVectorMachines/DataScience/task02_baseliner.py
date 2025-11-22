import os
import json
import pandas as pd
import numpy as np

DATASET_PATH = os.path.join("DATA", "music_dirty.txt")
TARGET = "popularity"
NON_NUMERIC_FEATURE = "genre"
RANDOM_STATE = 21
TEST_SIZE = 0.2

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

def create_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)

    return pd.DataFrame(data)



def encode_categorical(df, column):
    dummies = pd.get_dummies(df[column], drop_first=True, dtype=int)
    df = pd.concat([df, dummies], axis=1).drop(columns=[column])

    return df

def main():
    df = create_dataset(DATASET_PATH)
    df = encode_categorical(df, NON_NUMERIC_FEATURE)

    X = df.drop(columns=[TARGET])
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

if __name__ == '__main__':
    main()

import os
import json
import pandas as pd
import numpy as np

DATASET_PATH = os.path.join("..", "..", "DATA", "music_dirty_missing_vals.txt")
TARGET = "Rock"
NON_NUMERIC_FEATURE = "genre"
RANDOM_STATE = 21
TEST_SIZE = 0.2

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report

def create_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)

    return pd.DataFrame(data)

def main():
    df = create_dataset(DATASET_PATH)

    df['Rock'] = (df['genre'] == 'Rock').astype(int)
    df = df.drop(columns=['genre'])

    print(df.head())

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    mode_y = y_train.mode().iloc[0]
    y_pred_baseline = np.full_like(y_test, fill_value=mode_y)

    baseline_f1 = f1_score(y_test, y_pred_baseline, zero_division=0)
    baseline_precision = precision_score(y_test, y_pred_baseline, zero_division=0)
    baseline_recall = recall_score(y_test, y_pred_baseline, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_baseline)

    print(f"Mode target value: {mode_y}")
    print(f"Baseline F1: {baseline_f1:.4f}")
    print(f"Baseline Precision: {baseline_precision:.4f}")
    print(f"Baseline Recall: {baseline_recall:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_test, y_pred_baseline, zero_division=0))

if __name__ == '__main__':
    main()

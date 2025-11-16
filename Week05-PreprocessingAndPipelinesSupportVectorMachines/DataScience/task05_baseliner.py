import os
import json
import pandas as pd
import numpy as np

DATASET_PATH = os.path.join("..", "..", "DATA", "music_dirty_missing_vals.txt")
TARGET = "genre"
TEST_SIZE = 0.2
RANDOM_STATE = 21

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

def create_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)

    return pd.DataFrame(data)

def main():
    df = create_dataset(DATASET_PATH)

    df[TARGET] = df[TARGET].fillna('Unknown')
    codes, uniques = pd.factorize(df[TARGET])
    df[TARGET] = codes
    label_mapping = {code: genre for code, genre in enumerate(uniques)}

    print("Label mapping (numeric -> genre):")
    print(label_mapping)
    print(df.head())

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    mode_y = y_train.mode().iloc[0]
    y_pred_baseline = np.full(shape=y_test.shape, fill_value=mode_y)

    baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
    baseline_f1_macro = f1_score(y_test, y_pred_baseline, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred_baseline)

    print(f"Mode target numeric label: {mode_y} -> {label_mapping.get(mode_y)}")
    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
    print(f"Baseline Macro F1: {baseline_f1_macro:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report (macro metrics will be low for non-mode classes):")
    labels = sorted(label_mapping.keys())
    target_names = [label_mapping[i] for i in labels]
    print(classification_report(y_test, y_pred_baseline, labels=labels, target_names=target_names, zero_division=0, digits=4))

if __name__ == '__main__':
    main()

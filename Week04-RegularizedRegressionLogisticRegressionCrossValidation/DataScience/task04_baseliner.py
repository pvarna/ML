import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

DATASET_PATH = os.path.join("..", "..", "DATA", "diabetes_clean.csv")
TARGET = "diabetes"
RANDOM_STATE = 21
TEST_SIZE = 0.2


def show_metrics(y_true, y_pred):
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:\n",
          classification_report(y_true, y_pred, zero_division=0))


def main():
    df = pd.read_csv(DATASET_PATH, delimiter=',')
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    most_common_class = y_train.mode()[0]
    y_pred_baseline = np.full_like(y_test, fill_value=most_common_class)

    show_metrics(y_test, y_pred_baseline)


if __name__ == '__main__':
    main()

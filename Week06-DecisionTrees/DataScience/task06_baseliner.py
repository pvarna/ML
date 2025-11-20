import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer

DATASET_PATH = os.path.join("..", "..", "DATA",
                            "indian_liver_patient_dataset.csv")
OUTPUT_XLSX = "data_audit.xlsx"
ASSETS_DIR = "assets"
TARGET = "has_liver_disease"
FEATURE_TO_ENCODE = "Gender"
RANDOM_STATE = 21
TEST_SIZE = 0.2


def encode_categorical(df, column):
    dummies = pd.get_dummies(df[column], drop_first=True, dtype=int)
    df = pd.concat([df, dummies], axis=1).drop(columns=[column])

    return df


def create_target_column(df):
    df[TARGET] = df["Selector"].apply(lambda x: 0 if x == 2 else 1)
    df = encode_categorical(df, FEATURE_TO_ENCODE)
    df.drop(columns=["Selector"], inplace=True)

    return df


def main():
    df = pd.read_csv(DATASET_PATH, delimiter=',')
    df = create_target_column(df)

    print(df.head())

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    _, _, y_train, y_test = train_test_split(X,
                                             y,
                                             test_size=TEST_SIZE,
                                             random_state=RANDOM_STATE,
                                             stratify=y)

    baseline_mode = y_train.mode()[0]
    y_pred = pd.Series(baseline_mode, index=y_test.index)

    report = classification_report(y_test, y_pred, zero_division=0, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Baseline Mode Prediction: {baseline_mode:.4f}")
    print("Baseline Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)


if __name__ == '__main__':
    main()

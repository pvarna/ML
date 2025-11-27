import os
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from ml_lib.tree import DecisionTreeClassifier
from ml_lib.model_selection import train_test_split

DATASET_PATH = os.path.join("DATA", "indian_liver_patient_dataset.csv")
TARGET = "has_liver_disease"
FEATURE_TO_ENCODE = "Gender"
FEATURES = ["ALB"]
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
    df = df.dropna(subset=FEATURES + [TARGET])

    print(df.head())

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    model = DecisionTreeClassifier(min_samples_leaf=1, min_samples_split=2)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, zero_division=0, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)


if __name__ == '__main__':
    main()

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt

DATASET_PATH = os.path.join("..", "..", "DATA",
                            "indian_liver_patient_dataset.csv")
OUTPUT_XLSX = "data_audit.xlsx"
ASSETS_DIR = "assets"
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
    
    estimators = [
        ("knn", KNeighborsClassifier()),
        ("logreg", LogisticRegression(random_state=RANDOM_STATE)),
        ("dt", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ]

    model = VotingClassifier(estimators=estimators, voting='hard')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, zero_division=0, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    # model_report - https://docs.google.com/spreadsheets/d/15rf69swjbQGIJvVL6ZzTJvD-zOJB8BfWNe4J48R-04g/edit?usp=sharing


if __name__ == '__main__':
    main()

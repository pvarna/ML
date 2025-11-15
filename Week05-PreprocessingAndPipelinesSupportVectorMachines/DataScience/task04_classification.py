import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATASET_PATH = os.path.join("..", "..", "DATA", "music_dirty_missing_vals.txt")
TARGET = "Rock"
NON_NUMERIC_FEATURE = "genre"
RANDOM_STATE = 21
TEST_SIZE = 0.2
FEATURES_TO_IGNORE = []

def create_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def evaluate_and_print(y_true, y_pred, y_scores=None):
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    print(f"F1: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    if y_scores is not None:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        print(f"ROC AUC: {roc_auc:.4f}")
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()


def main():
    df = create_dataset(DATASET_PATH)
    df = df[df[NON_NUMERIC_FEATURE].notna()]

    df[TARGET] = (df[NON_NUMERIC_FEATURE] == 'Rock').astype(int)
    df = df.drop(columns=[NON_NUMERIC_FEATURE])

    print(df.head())

    X = df.drop(columns=[TARGET]+FEATURES_TO_IGNORE)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    logreg_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(penalty="l1", C=1, solver="liblinear", max_iter=5000)),
    ])
    logreg_pipe.fit(X_train, y_train)
    y_pred = logreg_pipe.predict(X_test)
    y_scores = logreg_pipe.predict_proba(X_test)[:, 1]
    evaluate_and_print(y_test, y_pred, y_scores=y_scores)

    # model_report - https://docs.google.com/spreadsheets/d/1p6NGDMq0-hD5uWBSpaBEnJsD38-THxC0QyYj7gI7diY/edit?usp=sharing


if __name__ == "__main__":
    main()

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

DATASET_PATH = os.path.join("..", "..", "DATA", "music_dirty_missing_vals.txt")
TARGET = "genre"
RANDOM_STATE = 21
TEST_SIZE = 0.2
FEATURES_TO_IGNORE = []
# FEATURES_TO_IGNORE = ["acousticness", "danceability", "duration_ms", "liveness", "tempo", "valence"]


def create_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def evaluate_and_print(name, y_true, y_pred, label_mapping):
    print(f"===== {name} =====")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:")
    labels = sorted(label_mapping.keys())
    target_names = [label_mapping[i] for i in labels]
    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0, digits=4))
    print()


def main():
    df = create_dataset(DATASET_PATH)
    df[TARGET] = df[TARGET].fillna("Unknown")
    codes, uniques = pd.factorize(df[TARGET])
    df[TARGET] = codes
    label_mapping = {code: genre for code, genre in enumerate(uniques)}
    print("Label mapping (numeric -> genre):")
    print(label_mapping)

    X = df.drop(columns=[TARGET] + FEATURES_TO_IGNORE)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(penalty="l1", C=10, solver="saga", class_weight=None, max_iter=50000)),
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    evaluate_and_print("LogisticRegression", y_test, y_pred, label_mapping)

    classes = sorted(label_mapping.keys())
    y_score = pipe.predict_proba(X_test)

    y_test_bin = label_binarize(y_test, classes=classes)
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {label_mapping[cls]} (AUC = {roc_auc:.3f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Logistic Regression")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.show()

    # model_report - https://docs.google.com/spreadsheets/d/1ctclq6TKEZ4luWBCVvbQ955kqyGKaaZfBfgTAPa2X2s/edit?usp=sharing


if __name__ == "__main__":
    main()

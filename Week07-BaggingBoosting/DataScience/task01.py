import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

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


def plot_roc_curve(classifier, X_test, y_test):
    y_score = classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

def plot_precision_recall_curve(classifier, X_test, y_test):
    y_score = classifier.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, y_score)
    avg_precision = average_precision_score(y_test, y_score)

    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision–Recall Curve')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()



def main():
    df = pd.read_csv(DATASET_PATH, delimiter=',')
    df = create_target_column(df)
    df = df.dropna().reset_index(drop=True)

    print(df.head())

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    classifier = XGBClassifier(random_state=RANDOM_STATE)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    plot_roc_curve(classifier, X_test, y_test)
    plot_precision_recall_curve(classifier, X_test, y_test)

    # model_report - https://docs.google.com/spreadsheets/d/1YGIyP-WXlwKaQTimuxFHrLoNRT_Ho2xOJ0-7r7ajbC8/edit?usp=sharing


if __name__ == '__main__':
    main()

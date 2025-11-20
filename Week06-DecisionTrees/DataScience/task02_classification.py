import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt

TARGET = "diagnosis"
FEATURES = ["mean radius", "mean texture", "worst radius"]
RANDOM_STATE = 21
TEST_SIZE = 0.2


def main():
    breast = load_breast_cancer()
    df = pd.DataFrame(breast.data, columns=breast.feature_names)
    df[TARGET] = breast.target

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    estimators = [
        ("knn", KNeighborsClassifier()),
        ("logreg", LogisticRegression()),
        ("dt", DecisionTreeClassifier()),
    ]

    soft_classifier = VotingClassifier(estimators=estimators, voting='soft')

    soft_classifier.fit(X_train, y_train)
    y_pred_soft = soft_classifier.predict(X_test)

    report_soft = classification_report(y_test, y_pred_soft, zero_division=0, digits=4)
    cm_soft = confusion_matrix(y_test, y_pred_soft)

    print("Classification Report:")
    print(report_soft)
    print("Confusion Matrix:")
    print(cm_soft)

    y_score = soft_classifier.predict_proba(X_test)[:, 1]
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

    # model_report - https://docs.google.com/spreadsheets/d/17fGwqcFUEQjWSO7KPNfcfKeDkMbFH6Rx7ltU5hPyxi0/edit?usp=sharing


if __name__ == '__main__':
    main()

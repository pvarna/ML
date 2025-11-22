import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

DATASET_PATH = os.path.join("..", "..", "DATA", "music_clean.csv")
TARGET = "is_popular"
FEATURE_TO_TRANSFORM = "popularity"
FEATURES = ["genre"]
RANDOM_STATE = 21
TEST_SIZE = 0.2


def main():
    df = pd.read_csv(DATASET_PATH, delimiter=',', index_col=0)

    popularity_mean = df[FEATURE_TO_TRANSFORM].mean()
    df[TARGET] = (df[FEATURE_TO_TRANSFORM] > popularity_mean).astype(int)
    df = df.drop(columns=FEATURE_TO_TRANSFORM)
    print(df.head())

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, zero_division=0, digits=4)
    print(report)
    m = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(m)

    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - KNN')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    # model_report - https://docs.google.com/spreadsheets/d/1QOsHeyFR3T-CTAKGgIUBFbLWdjjCq7O1RpXzldkEX3Y/edit?usp=sharing 

if __name__ == '__main__':
    main()

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from ml_lib.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

DATASET_PATH = os.path.join("DATA", "music_clean.csv")
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

    model = SVC(C=1.0, kernel="rbf")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, zero_division=0, digits=4)
    print(report)
    m = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(m)


if __name__ == '__main__':
    main()

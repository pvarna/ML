import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ml_lib.metrics import accuracy_score
from ml_lib.model_selection import train_test_split
from ml_lib.neighbors import KNeighborsClassifier

TELECOM_PATH = "DATA/telecom_churn_clean.csv"


def main():
    df_telecom = pd.read_csv(TELECOM_PATH, delimiter=',', index_col=0)
    X = df_telecom.drop(columns=["churn"])
    y = df_telecom["churn"]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=21,
                                                        stratify=y)

    knn = KNeighborsClassifier(n_neighbors=11,
                               metric="manhattan")
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, y_pred_knn)

    print(f"KNN accuracy: {knn_acc:.4f}")

if __name__ == '__main__':
    main()

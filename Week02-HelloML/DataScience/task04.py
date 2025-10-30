import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

TELECOM_PATH = "../../DATA/telecom_churn_clean.csv"


def main():
    df_telecom = pd.read_csv(TELECOM_PATH, delimiter=',', index_col=0)
    X = df_telecom.drop(columns=["churn"])
    y = df_telecom["churn"]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=21,
                                                        stratify=y)

    most_common_class = y_train.mode()[0]
    y_pred_baseline = np.full_like(y_test, fill_value=most_common_class)

    baseline_acc = accuracy_score(y_test, y_pred_baseline)

    knn = KNeighborsClassifier(n_neighbors=11,
                               weights="distance",
                               p=1,
                               metric="minkowski")
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, y_pred_knn)

    print(f"Most common class (baseline prediction): {most_common_class}")
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    print(f"KNN accuracy: {knn_acc:.4f}")

    # Model report: https://docs.google.com/spreadsheets/d/1NrVld1ji6osZ8qsHH8mIhG4znXbgW49j2ONODqQlZeU/edit?usp=sharing


if __name__ == '__main__':
    main()

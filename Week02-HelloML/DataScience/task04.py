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
    # X = df_telecom[["account_length", "customer_service_calls"]]
    X = df_telecom.drop(columns=["churn"])
    y = df_telecom["churn"]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=21,
                                                        stratify=y)

    # ---- Baseline model ----
    most_common_class = y_train.mode()[0]
    y_pred_baseline = np.full_like(y_test, fill_value=most_common_class)

    baseline_acc = accuracy_score(y_test, y_pred_baseline)

    # ---- KNN model (default parameters) ----
    knn = KNeighborsClassifier(n_neighbors=11,
                               weights="distance",
                               p=1,
                               metric="minkowski")
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, y_pred_knn)

    # ---- Print results ----
    print(f"Most common class (baseline prediction): {most_common_class}")
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    print(f"KNN accuracy: {knn_acc:.4f}")

    # Model report: https://docs.google.com/spreadsheets/d/1NrVld1ji6osZ8qsHH8mIhG4znXbgW49j2ONODqQlZeU/edit?usp=sharing

    # ps = list(range(1, 21, 1))
    # train_accuracies = {}
    # test_accuracies = {}

    # best_p = None
    # best_test_acc = -1.0

    # for p in ps:
    #     knn = KNeighborsClassifier(n_neighbors=11, p=p, weights="distance")
    #     knn.fit(X_train, y_train)

    #     y_train_pred = knn.predict(X_train)
    #     y_test_pred = knn.predict(X_test)

    #     train_acc = accuracy_score(y_train, y_train_pred)
    #     test_acc = accuracy_score(y_test, y_test_pred)

    #     train_accuracies[p] = train_acc
    #     test_accuracies[p] = test_acc

    #     if test_acc >= best_test_acc:
    #         best_p = p
    #         best_test_acc = test_acc

    # print(f"Baseline accuracy (most common class={most_common_class}): {baseline_acc:.4f}")
    # print(f"Best n by test accuracy: {best_p}  |  Test acc: {best_test_acc:.4f}  |  Train acc: {train_accuracies[best_p]:.4f}")

    # plt.figure(figsize=(8, 6))
    # plt.title('KNN: Varying P (Power parameter for the Minkowski metric)')
    # plt.plot(ps, [train_accuracies[p] for p in ps], label='Training Accuracy')
    # plt.plot(ps, [test_accuracies[p] for p in ps], label='Testing Accuracy')
    # plt.legend()
    # plt.xlabel('P (Power parameter for the Minkowski metric)')
    # plt.ylabel('Accuracy')
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    main()

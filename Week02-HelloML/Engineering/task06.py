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

    ks = list(range(1, 51, 1))
    train_accuracies = {}
    test_accuracies = {}

    best_k = None
    best_test_acc = -1.0

    for k in ks:
        print("Testing with k = {k}")
        knn = KNeighborsClassifier(n_neighbors=k, metric="manhattan")
        knn.fit(X_train, y_train)

        y_train_pred = knn.predict(X_train)
        y_test_pred = knn.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        train_accuracies[k] = train_acc
        test_accuracies[k] = test_acc

        if test_acc >= best_test_acc:
            best_k = k
            best_test_acc = test_acc

    print(f"Best k by test accuracy: {best_k}  |  Test acc: {best_test_acc:.4f}  |  Train acc: {train_accuracies[best_k]:.4f}")

    plt.figure(figsize=(8, 6))
    plt.title('KNN: Varying K (K-Nearest neigbors)')
    plt.plot(ks, [train_accuracies[p] for p in ks], label='Training Accuracy')
    plt.plot(ks, [test_accuracies[p] for p in ks], label='Testing Accuracy')
    plt.legend()
    plt.xlabel('K (K-Nearest neigbors)')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

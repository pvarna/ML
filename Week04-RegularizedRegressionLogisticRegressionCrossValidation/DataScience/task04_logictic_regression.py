import itertools
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

DATASET_PATH = os.path.join("..", "..", "DATA", "diabetes_clean.csv")
TARGET = "diabetes"
FEATURES_TO_IGNORE = []
RANDOM_STATE = 21
TEST_SIZE = 0.2

def iter_param_grid(param_grid):
    """Yield dicts of all param combinations from a list of grids."""
    for grid in param_grid:
        keys = list(grid.keys())
        for values in itertools.product(*(grid[k] for k in keys)):
            yield dict(zip(keys, values))

def run_logreg_grid(X_train, y_train, X_test, y_test, param_grid):
    counter = 1
    data = []
    for params in iter_param_grid(param_grid):
        # build model with safe defaults
        model = LogisticRegression(
            penalty=params["penalty"],
            C=params["C"],
            solver=params["solver"],
            class_weight=params["class_weight"],
            max_iter=5000,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("="*80)
        print(f"Counter: {counter}, Params: {params}")
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

        data.append({
          "C": params["C"],
          "penalty": params["penalty"],
          "solver": params["solver"],
          "class_weight": params["class_weight"],
          "F1_weighted_avg": report["weighted avg"]["f1-score"],
          "F1_class_0": report["0"]["f1-score"],
          "F1_class_1": report["1"]["f1-score"],
          "Recall_weighted_avg": report["weighted avg"]["recall"],
          "Recall_class_0": report["0"]["recall"],
          "Recall_class_1": report["1"]["recall"],
          "Confusion_matrix": matrix
        })

        if (params["C"] == 0.1 and params['penalty'] == "l1" and params["solver"] == "liblinear" and params["class_weight"] == "balanced"):
            print(report)
            RocCurveDisplay.from_predictions(y_test, y_pred)
            plt.tight_layout()
            plt.show()

        counter += 1

    with open('metrics.csv', 'w', newline='') as csvfile:
        fieldnames = ['C', 'penalty', 'solver', 'class_weight', "F1_weighted_avg", "F1_class_0", "F1_class_1", "Recall_weighted_avg", "Recall_class_0", "Recall_class_1", "Confusion_matrix"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def main():
    df = pd.read_csv(DATASET_PATH, delimiter=',')
    X = df.drop(columns=FEATURES_TO_IGNORE+[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    param_grid = [
        {
            'penalty': ['l1'],
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'saga'],
            'class_weight': [None, 'balanced']
        },
        {
            'penalty': ['l2'],
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear'],
            'class_weight': [None, 'balanced']
        }
    ]

    run_logreg_grid(X_train, y_train, X_test, y_test, param_grid)

    # model_report - https://docs.google.com/spreadsheets/d/1SXu7rERsiOZA4tW6u-gnbQg3OUiI6pESTW4Q-3ATIzs/edit?usp=sharing


if __name__ == '__main__':
    main()

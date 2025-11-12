import os
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    make_scorer,
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression

DATASET_PATH = os.path.join("..", "..", "DATA", "diabetes_clean.csv")
TARGET = "diabetes"
FEATURES_TO_IGNORE = []
RANDOM_STATE = 21
TEST_SIZE = 0.2
CV_FOLDS = 5
MAX_ITER = 5000
N_ITER = 50


def build_param_grid():
    return [
        {
            "penalty": ["l1"],
            "C": [0.1, 1, 10],
            "solver": ["liblinear", "saga"],
            "class_weight": [None, "balanced"],
        },
        {
            "penalty": ["l2"],
            "C": [0.1, 1, 10],
            "solver": ["lbfgs", "liblinear"],
            "class_weight": [None, "balanced"],
        },
    ]


def build_scorers():
    return {
        "f1_weighted":
        make_scorer(f1_score, average="weighted", zero_division=0),
        "f1_pos":
        make_scorer(f1_score, pos_label=1, zero_division=0),
        "f1_neg":
        make_scorer(f1_score, pos_label=0, zero_division=0),
        "recall_weighted":
        make_scorer(recall_score, average="weighted", zero_division=0),
        "recall_pos":
        make_scorer(recall_score, pos_label=1, zero_division=0),
        "recall_neg":
        make_scorer(recall_score, pos_label=0, zero_division=0),
    }


def fit_grid_search(X_train, y_train):
    base = LogisticRegression(random_state=RANDOM_STATE, max_iter=MAX_ITER)

    cv = StratifiedKFold(n_splits=CV_FOLDS,
                         shuffle=True,
                         random_state=RANDOM_STATE)

    gs = GridSearchCV(
        estimator=base,
        param_grid=build_param_grid(),
        scoring=build_scorers(),
        refit="f1_weighted",
        cv=cv,
        return_train_score=False,
    )
    gs.fit(X_train, y_train)
    return gs


def save_cv_metrics_csv(gs,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        out_path="metrics.csv"):
    results = pd.DataFrame(gs.cv_results_)

    keep_cols = [
        "param_C",
        "param_penalty",
        "param_solver",
        "param_class_weight",
        "mean_test_f1_weighted",
        "mean_test_f1_neg",
        "mean_test_f1_pos",
        "mean_test_recall_weighted",
        "mean_test_recall_neg",
        "mean_test_recall_pos",
    ]
    keep_cols = [c for c in keep_cols if c in results.columns]
    tidy = results[keep_cols].copy()

    matrices = []
    for params in gs.cv_results_["params"]:
        model = LogisticRegression(random_state=RANDOM_STATE,
                                   **params,
                                   max_iter=MAX_ITER)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        matrices.append(cm)

    tidy["Confusion_matrix"] = matrices
    tidy.to_csv(out_path, index=False)


def evaluate_on_test(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    print("Best model on test set:")
    print(classification_report(y_test, y_pred, zero_division=0))
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("Confusion matrix:")
    print(matrix)

    RocCurveDisplay.from_predictions(y_test, y_score)
    plt.title("ROC Curve – Best Logistic Regression Model")
    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv(DATASET_PATH)
    X = df.drop(columns=FEATURES_TO_IGNORE + [TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    gs = fit_grid_search(X_train, y_train)
    print(f"Best params: {gs.best_params_}")

    save_cv_metrics_csv(gs, X_train, y_train, X_test, y_test)
    evaluate_on_test(gs.best_estimator_, X_test, y_test)

    # model_report - https://docs.google.com/spreadsheets/d/1SXu7rERsiOZA4tW6u-gnbQg3OUiI6pESTW4Q-3ATIzs/edit?usp=sharing


if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def encode_categorical(df, column):
    dummies = pd.get_dummies(df[column], drop_first=True, dtype=int)
    df = pd.concat([df, dummies], axis=1).drop(columns=[column])

    return df

def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)

    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_importances)), sorted_importances)
    plt.yticks(range(len(sorted_importances)), sorted_names)
    plt.xlabel("Feature importance")
    plt.title("Decision Tree Feature Importances")
    plt.tight_layout()
    plt.show()


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


def plot_model_cv_bars(gs):
    res = pd.DataFrame(gs.cv_results_)

    res["param_label"] = res["params"].apply(
        lambda d: ", ".join(f"{v}" for _, v in d.items()))

    df = res[["param_label", "mean_train_score", "mean_test_score"]].copy()
    df = df.sort_values("mean_test_score",
                        ascending=True).reset_index(drop=True)

    y = np.arange(len(df))
    h = 0.35

    plt.figure()

    plt.barh(y - h / 2, df["mean_train_score"], height=h, label="Train")
    plt.barh(y + h / 2, df["mean_test_score"], height=h, label="Test")

    plt.yticks(y, df["param_label"])

    plt.xlabel("F1 (weighted)")
    plt.title("Bagging Classifier – CV F1 (weighted) by parameters")
    plt.legend()
    plt.tight_layout()
    plt.show()


def grid_search(classifier):
    grid_search = GridSearchCV(classifier,
                               build_param_grid(),
                               cv=5,
                               scoring="f1_weighted",
                               return_train_score=True)

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


def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color="red")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals (y_true - y_pred)")
    plt.title(f"Residuals vs Predicted")
    plt.tight_layout()
    plt.show()
    
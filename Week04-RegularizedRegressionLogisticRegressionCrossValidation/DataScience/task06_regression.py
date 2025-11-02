import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (r2_score, root_mean_squared_error, make_scorer)

DATASET_PATH = os.path.join("..", "..", "DATA", "diabetes_clean.csv")
TARGET = "glucose"
FEATURES_TO_IGNORE = []
RANDOM_STATE = 21
TEST_SIZE = 0.2
CV_FOLDS = 5
MAX_ITER = 5000

ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

PRIMARY_SCORER = "r2"


def build_scorers():
    return {
        "r2": make_scorer(r2_score),
        "neg_rmse": make_scorer(root_mean_squared_error,
                                greater_is_better=False),
    }


def plot_model_cv_bars(gs, score="r2", param_name=None, title=""):
    res = pd.DataFrame(gs.cv_results_)

    if param_name and f"param_{param_name}" in res.columns:
        res["param_label"] = res[f"param_{param_name}"].astype(str)
    else:
        res["param_label"] = res["params"].astype(str)

    cols = ["param_label", f"mean_train_{score}", f"mean_test_{score}"]
    df = res[cols].copy()

    df = df.sort_values(f"mean_test_{score}", ascending=True)
    y = np.arange(len(df))
    h = 0.35

    plt.figure()
    plt.barh(y - h / 2, df[f"mean_train_{score}"], height=h, label="Train")
    plt.barh(y + h / 2, df[f"mean_test_{score}"], height=h, label="Test")
    plt.yticks(y, df["param_label"])
    plt.xlabel(score.upper())
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_on_test(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"R^2:   {r2:0.4f}")
    print(f"RMSE: {rmse:0.4f}")

    plt.figure()
    plt.scatter(y_test, y_pred)
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color="red")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{name} – Predicted vs Actual")
    plt.tight_layout()
    plt.show()

    residuals = y_test - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color="red")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals (y_true - y_pred)")
    plt.title(f"{name} – Residuals vs Predicted")
    plt.tight_layout()
    plt.show()


def save_cv_results_csv(gs, param_cols, filename):
    results = pd.DataFrame(gs.cv_results_)
    keep_cols = (param_cols + [
        "mean_test_r2",
        "mean_test_neg_rmse",
    ])
    keep_cols = [c for c in keep_cols if c in results.columns]

    results[keep_cols].to_csv(filename, index=False)


def main():
    df = pd.read_csv(DATASET_PATH)
    X = df.drop(columns=FEATURES_TO_IGNORE + [TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    linreg = LinearRegression()
    linreg_grid = GridSearchCV(
        estimator=linreg,
        param_grid=[{}],
        scoring=build_scorers(),
        refit=PRIMARY_SCORER,
        cv=cv,
        return_train_score=True,
    )
    linreg_grid.fit(X_train, y_train)
    print("LinearRegression best params:", linreg_grid.best_params_)
    plot_model_cv_bars(linreg_grid,
                       score="r2",
                       title="Linear Regression – CV R^2 by parameters")
    save_cv_results_csv(
        linreg_grid,
        param_cols=[],
        filename="metrics_linear.csv",
    )
    evaluate_on_test("LinearRegression (best)", linreg_grid.best_estimator_,
                     X_test, y_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ridge = Ridge(random_state=RANDOM_STATE)
    ridge_grid = GridSearchCV(
        estimator=ridge,
        param_grid={"alpha": ALPHA_GRID},
        scoring=build_scorers(),
        refit=PRIMARY_SCORER,
        cv=cv,
        return_train_score=True,
    )
    ridge_grid.fit(X_train_scaled, y_train)
    plot_model_cv_bars(ridge_grid,
                       score="r2",
                       param_name="alpha",
                       title="Ridge – CV R^2 by alpha")
    print("Ridge best params:", ridge_grid.best_params_)
    save_cv_results_csv(
        ridge_grid,
        param_cols=["param_alpha"],
        filename="metrics_ridge.csv",
    )
    evaluate_on_test("Ridge (best)", ridge_grid.best_estimator_, X_test_scaled,
                     y_test)

    lasso = Lasso(random_state=RANDOM_STATE, max_iter=MAX_ITER)
    lasso_grid = GridSearchCV(
        estimator=lasso,
        param_grid={"alpha": ALPHA_GRID},
        scoring=build_scorers(),
        refit=PRIMARY_SCORER,
        cv=cv,
        return_train_score=True,
    )
    lasso_grid.fit(X_train_scaled, y_train)
    plot_model_cv_bars(lasso_grid,
                       score="r2",
                       param_name="alpha",
                       title="Lasso – CV R^2 by alpha")
    print("Lasso best params:", lasso_grid.best_params_)
    save_cv_results_csv(
        lasso_grid,
        param_cols=["param_alpha"],
        filename="metrics_lasso.csv",
    )
    evaluate_on_test("Lasso (best)", lasso_grid.best_estimator_, X_test_scaled,
                     y_test)

    models = {
        "LinearRegression": (linreg_grid.best_estimator_, X_test),
        "Ridge": (ridge_grid.best_estimator_, X_test_scaled),
        "Lasso": (lasso_grid.best_estimator_, X_test_scaled),
    }
    rows = []
    for name, (m, Xt) in models.items():
        y_pred = m.predict(Xt)
        r2 = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        rows.append([name, r2, rmse])
    summary = pd.DataFrame(rows, columns=["model", "r2", "rmse"])
    print(summary)

    # model_report - https://docs.google.com/spreadsheets/d/1FZF-kkxXZyQPouweBr2qDnkLcnizZY_tFMiaZZK-Iqk/edit?usp=sharing


if __name__ == "__main__":
    main()

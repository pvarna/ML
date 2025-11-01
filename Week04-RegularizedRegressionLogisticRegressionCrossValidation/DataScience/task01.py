import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso

DATASET_PATH = os.path.join("..", "..", "DATA",
                            "advertising_and_sales_clean.csv")
TARGET = "sales"
RANDOM_STATE = 21
FEATURES_TO_IGNORE = ["influencer"]


def r2_adjusted_score(samples_count, features_count, r2_score):
    return 1 - (1 - r2_score) * (samples_count - 1) / (samples_count -
                                                       features_count - 1)


def plot_horizontal_cv_bars(
        df,
        y="params_label",
        test="mean_test_score",
        train="mean_train_score",
        title="Lasso CV: Train vs Test (sorted by mean test score)"):
    df = df.sort_values(test, ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df))
    h = 0.35

    plt.figure(figsize=(9, 5.5))
    plt.barh(y_pos - h / 2, df[test], h, label="Mean CV Test R²")
    plt.barh(y_pos + h / 2, df[train], h, label="Mean CV Train R²")
    plt.yticks(y_pos, df[y])
    plt.xlabel("Score")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.xlim(0, max(df[[test, train]].to_numpy().max() + 0.02, 1.0))
    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv(DATASET_PATH, delimiter=',')
    X = df.drop(columns=FEATURES_TO_IGNORE + [TARGET])
    y = df[TARGET]

    kF = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    regression = Lasso(random_state=RANDOM_STATE)
    param_grid = {
        "alpha": [0.001, 0.01, 0.1, 1, 10],
        "positive": [True, False]
    }
    gs = GridSearchCV(
        regression,
        param_grid=param_grid,
        scoring="r2",
        cv=kF,
        return_train_score=True,
        n_jobs=-1,
    )
    gs.fit(X, y)

    cv = pd.DataFrame(gs.cv_results_)
    cv["params_label"] = cv["params"].apply(
        lambda p: " | ".join(f"{k}={v}" for k, v in p.items()))

    sorted_cv = cv.sort_values("mean_test_score", ascending=True)

    for _, row in sorted_cv.iterrows():
        print(f"{row['params_label']}: "
              f"mean_train_score={row['mean_train_score']:.4f}, "
              f"mean_test_score={row['mean_test_score']:.4f}")

    best_model = gs.best_estimator_
    coefficients = pd.Series(best_model.coef_, index=X.columns)
    print("Coefficients:")
    print(coefficients)

    plot_horizontal_cv_bars(
        cv[["params_label", "mean_test_score", "mean_train_score"]])
    
    # model_report - https://docs.google.com/spreadsheets/d/1nPC2wWj7tw9tw7x0ZcMxZn9yXoiPdyiF2fVgILwRmSw/edit?usp=sharing


if __name__ == '__main__':
    main()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.linear_model import LinearRegression

DATASET_PATH = os.path.join("..", "..", "DATA",
                            "advertising_and_sales_clean.csv")
TARGET = "sales"
NON_NUMERICAL_FEATURES = ["influencer"]
FEATURES_TO_IGNORE = ["influencer", "radio", "social_media"]


def r2_adjusted_score(samples_count, features_count, r2_score):
    return 1 - (1 - r2_score) * (samples_count - 1) / (samples_count -
                                                       features_count - 1)


def main():
    df = pd.read_csv(DATASET_PATH, delimiter=',')
    X = df.drop(columns=FEATURES_TO_IGNORE + [TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=21)

    samples_count, features_count = X_test.shape

    mean_y = y_train.mean()
    y_pred_baseline = np.full_like(y_test, fill_value=mean_y)

    baseline_r2 = r2_score(y_test, y_pred_baseline)
    baseline_r2_adj = r2_adjusted_score(samples_count, features_count,
                                        baseline_r2)
    baseline_rmse = root_mean_squared_error(y_test, y_pred_baseline)

    print(f"Mean target value: {mean_y}")
    print(f"Baseline R^2: {baseline_r2:.4f}")
    print(f"Baseline R^2 adjusted: {baseline_r2_adj:.4f}")
    print(f"Baseline RMSE: {baseline_rmse:.4f}")

    linear_regression = LinearRegression(positive=True)
    linear_regression.fit(X_train, y_train)

    y_pred_linear_regression = linear_regression.predict(X_test)
    linear_regression_r2 = r2_score(y_test, y_pred_linear_regression)
    linear_regression_r2_adj = r2_adjusted_score(samples_count, features_count,
                                                 linear_regression_r2)
    linear_regression_r2_rmse = root_mean_squared_error(
        y_test, y_pred_linear_regression)

    print(f"Linear regression R^2: {linear_regression_r2:.4f}")
    print(f"Linear regression R^2 adjusted: {linear_regression_r2_adj:.4f}")
    print(f"Linear regression RMSE: {linear_regression_r2_rmse:.4f}")

    # model_report - https://docs.google.com/spreadsheets/d/1FtbAG1QNvNbQBJJKCIoQrwFHvGgexOraszZJURoRaHY/edit?usp=sharing


if __name__ == '__main__':
    main()

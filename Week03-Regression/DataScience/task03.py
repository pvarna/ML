import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

DATASET_PATH = os.path.join("..", "..", "DATA",
                            "advertising_and_sales_clean.csv")
TARGET = "sales"
NON_NUMERICAL_FEATURES = ["influencer"]
FEATURES_TO_IGNORE = ["influencer", "radio", "social_media"]

def r2_adjusted_score(samples_count, features_count, r2_score):
    return 1 - (1 - r2_score) * (samples_count - 1) / (samples_count - features_count - 1)

def main():
    df = pd.read_csv(DATASET_PATH, delimiter=',')
    # df = pd.get_dummies(df, columns=NON_NUMERICAL_FEATURES, drop_first=True) # https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
    X = df.drop(columns=FEATURES_TO_IGNORE+[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=21)

    samples_count, features_count = X_test.shape

    mean_y = y_train.mean()
    y_pred_baseline = np.full_like(y_test, fill_value=mean_y)

    baseline_r2 = r2_score(y_test, y_pred_baseline)
    baseline_r2_adj = r2_adjusted_score(samples_count, features_count, baseline_r2)
    baseline_mse = mean_squared_error(y_test, y_pred_baseline)

    print(f"Mean target value: {mean_y}")
    print(f"Baseline R^2: {baseline_r2:.4f}")
    print(f"Baseline R^2 adjusted: {baseline_r2_adj:.4f}")
    print(f"Baseline MSE: {baseline_mse:.4f}")
    

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train)

    y_pred_linear_regression = linear_regression.predict(X_test)
    linear_regression_r2 = r2_score(y_test, y_pred_linear_regression)
    linear_regression_r2_adj = r2_adjusted_score(samples_count, features_count, linear_regression_r2)
    linear_regression_r2_mse = mean_squared_error(y_test, y_pred_linear_regression)

    print(f"Linear regression R^2: {linear_regression_r2:.4f}")
    print(f"Linear regression R^2 adjusted: {linear_regression_r2_adj:.4f}")
    print(f"Linear regression MSE: {linear_regression_r2_mse:.4f}")


    ax = sns.scatterplot(x=X["tv"], y=y)
    sns.lineplot(x=X_test["tv"], y=y_pred_linear_regression, color="red", label="Prediction")

    ax.set_title(f"Linear Regression")
    ax.set_xlabel(f"TV Feature")
    ax.set_ylabel(f"Sales Target")
    plt.legend()
    plt.tight_layout()
    plt.show()

    residuals = y_test - y_pred_linear_regression
    sns.scatterplot(x=y_pred_linear_regression, y=residuals)
    plt.axhline(0, color="red")
    plt.title("Residuals vs Predicted Values")
    plt.xlabel("Predicted Sales")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.tight_layout()
    plt.show()

    # https://docs.google.com/spreadsheets/d/1FtbAG1QNvNbQBJJKCIoQrwFHvGgexOraszZJURoRaHY/edit?usp=sharing


if __name__ == '__main__':
    main()

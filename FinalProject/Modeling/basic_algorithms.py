import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    cohen_kappa_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

DATASET_PATH = os.path.join(
    "..", "DATA", "distributed_system_architecture_stress_dataset.csv")
TARGET = "system_state"
CATEGORICAL_FEATURES = [
    "architecture_type", "deployment_type", "communication_type"
]
NUMERIC_FEATURES = [
    "num_services", "num_databases", "requests_per_second", "avg_payload_kb",
    "read_write_ratio", "peak_traffic_multiplier", "cpu_utilization_percent",
    "memory_utilization_percent", "network_latency_ms", "packet_loss_percent",
    "avg_latency_ms", "p95_latency_ms", "error_rate_percent",
    "db_connection_pool_exhausted", "retry_storm_detected",
    "circuit_breaker_open"
]
RANDOM_STATE = 42
TEST_SIZE = 0.2
DIGITS = 6


def one_hot_encode_feature(df, feature):
    dummies = pd.get_dummies(df[feature],
                             drop_first=True,
                             prefix=feature,
                             dtype=int)
    df = pd.concat([df.drop(columns=[feature]), dummies], axis=1)

    return df


def main():
    df = pd.read_csv(DATASET_PATH, delimiter=',')

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()

    for feature in CATEGORICAL_FEATURES:
        X = one_hot_encode_feature(X, feature)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "Support Vector Machine": SVC(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    }

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)

        macro_f1 = f1_score(y_test, predictions, average='macro')
        cohen_kappa = cohen_kappa_score(y_test, predictions)

        print(f"{name} Performance:")
        print(f"Macro F1 Score: {macro_f1:.{DIGITS}f}")
        print(f"Cohen's Kappa: {cohen_kappa:.{DIGITS}f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, digits=DIGITS))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        print("-" * 50)

if __name__ == '__main__':
    main()

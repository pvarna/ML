from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

RANDOM_STATE = 42


def main():
    # Load the dataset
    df = pd.read_csv(
        "../DATA/distributed_system_architecture_stress_dataset.csv")

    le = LabelEncoder()

    df['architecture_type'] = le.fit_transform(df['architecture_type'])
    df['system_state'] = le.fit_transform(df['system_state'])
    df['root_cause'] = le.fit_transform(
        df['root_cause'])  # Convert categorical columns to numeric

    df['deployment_type'] = le.fit_transform(df['deployment_type'])
    df['communication_type'] = le.fit_transform(df['communication_type'])

    X = df.drop(['system_state', 'root_cause'], axis=1)
    y = df['system_state']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(df['system_state'].value_counts())

    # -------- Logistic Regression --------
    lr = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)

    # -------- Random Forest --------
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    # -------- Comparison Table --------
    comparison = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [
            accuracy_score(y_test, pred_lr),
            accuracy_score(y_test, pred_rf),
        ],
        "Precision": [
            precision_score(y_test, pred_lr, average='weighted'),
            precision_score(y_test, pred_rf, average='weighted'),
        ],
        "Recall": [
            recall_score(y_test, pred_lr, average='weighted'),
            recall_score(y_test, pred_rf, average='weighted'),
        ],
        "F1 Score": [
            f1_score(y_test, pred_lr, average='weighted'),
            f1_score(y_test, pred_rf, average='weighted'),
        ],
        "F1 Score Macro": [
            f1_score(y_test, pred_lr, average='macro'),
            f1_score(y_test, pred_rf, average='macro'),
        ],
        "Cohen's Kappa": [
            cohen_kappa_score(y_test, pred_lr),
            cohen_kappa_score(y_test, pred_rf),
        ],
    })

    print("Confusion Matrix for Logistic Regression:\n",
          confusion_matrix(y_test, pred_lr))
    print("Confusion Matrix for Random Forest:\n",
          confusion_matrix(y_test, pred_rf))
    print("\nModel Comparison:\n", comparison)


if __name__ == "__main__":
    main()

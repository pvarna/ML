import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

TELECOM_PATH = "../../DATA/telecom_churn_clean.csv"


def main():
    df_telecom = pd.read_csv(TELECOM_PATH, delimiter=',', index_col=0)
    X = df_telecom[["account_length", "customer_service_calls"]]
    y = df_telecom["churn"]

    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(X, y)

    X_new = np.array([[30.0, 17.5],
                  [107.0, 24.1],
                  [213.0, 10.9]])
    
    predictions = knn.predict(X_new)
    print(f'{predictions=}')



if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FILE = "../../DATA/cars.csv"


def main():
    df_cars = pd.read_csv(FILE, delimiter=',')
    print(df_cars)

    print("\nAfter setting first column as index:")
    df_cars_better = pd.read_csv(
        FILE, delimiter=',', index_col=0
    )  # https://stackoverflow.com/questions/36606931/how-to-set-in-pandas-the-first-column-and-row-as-index
    print(df_cars_better)


if __name__ == '__main__':
    main()

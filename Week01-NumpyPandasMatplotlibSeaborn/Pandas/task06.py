import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FILE_CARS_ADVANCED = "../../DATA/cars_advanced.csv"


def main():
    df_cars_advanced = pd.read_csv(FILE_CARS_ADVANCED,
                                   delimiter=',',
                                   index_col=0)

    for label, row in df_cars_advanced.iterrows():
        print(f"Label is \"{label}\":")
        print("Row contents:")
        print(row)
        print()


if __name__ == '__main__':
    main()

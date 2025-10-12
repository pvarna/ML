import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FILE_CARS_ADVANCED = "../../DATA/cars_advanced.csv"


def main():
    df_cars_advanced = pd.read_csv(FILE_CARS_ADVANCED,
                                   delimiter=',',
                                   index_col=0)

    print(df_cars_advanced)
    print()

    df_cars_advanced["COUNTRY"] = df_cars_advanced["country"].str.upper()
    print(df_cars_advanced)


if __name__ == '__main__':
    main()

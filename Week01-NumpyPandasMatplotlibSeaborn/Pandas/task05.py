import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FILE_CARS_ADVANCED = "../../DATA/cars_advanced.csv"


def main():
    df_cars_advanced = pd.read_csv(FILE_CARS_ADVANCED,
                                   delimiter=',',
                                   index_col=0)

    df_cars_dr = df_cars_advanced[df_cars_advanced["drives_right"]]
    print(df_cars_dr)

    print()

    df_cars_more_than_500 = df_cars_advanced.loc[
        df_cars_advanced["cars_per_cap"] > 500, "country"]
    print(df_cars_more_than_500)

    print()

    df_cars_between_10_and_80_v1 = df_cars_advanced[
        df_cars_advanced["cars_per_cap"].between(10, 80)]["country"]
    print(df_cars_between_10_and_80_v1)

    print()

    df_cars_between_10_and_80_v2 = df_cars_advanced.loc[
        (df_cars_advanced["cars_per_cap"] >= 10) &
        (df_cars_advanced["cars_per_cap"] <= 80), "country"]
    print(df_cars_between_10_and_80_v2)


if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FILE_CARS = "../../DATA/cars.csv"
FILE_CARS_ADVANCED = "../../DATA/cars_advanced.csv"


def main():
    df_cars = pd.read_csv(FILE_CARS, delimiter=',', index_col=0)
    df_cars_advanced = pd.read_csv(FILE_CARS_ADVANCED,
                                   delimiter=',',
                                   index_col=0)

    country_as_series = df_cars["country"]
    print(country_as_series)

    print()

    country_as_df = df_cars[["country"]]
    print(country_as_df)

    print()

    country_dr_df = df_cars[["country", "drives_right"]]
    print(country_dr_df)

    print()

    first_three_obs = df_cars.iloc[:3]
    print(first_three_obs)

    print()

    other_obs = df_cars.iloc[4:7]
    print(other_obs)

    print()

    japan_as_series = df_cars_advanced.loc["JPN"]
    print(japan_as_series)

    print()

    australian_egypt_as_df = df_cars_advanced.loc[["AUS", "EG"]]
    print(australian_egypt_as_df)

    print()

    morocco_dr = df_cars_advanced.loc[["MOR"], ["drives_right"]]
    print(morocco_dr)

    print()

    russia_morocco_country_dr = df_cars_advanced.loc[
        ["RU", "MOR"], ["country", "drives_right"]]
    print(russia_morocco_country_dr)


if __name__ == '__main__':
    main()

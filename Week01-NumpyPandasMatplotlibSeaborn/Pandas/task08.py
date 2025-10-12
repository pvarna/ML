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

    uppercase_countries = []
    for _, row in df_cars_advanced.iterrows():
        uppercase_countries.append(row["country"].upper())

    df_cars_advanced["COUNTRY"] = uppercase_countries
    print(df_cars_advanced)



if __name__ == '__main__':
    main()

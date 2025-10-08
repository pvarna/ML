import pandas as pd

def main():
    names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
    dr =  [True, False, False, False, True, True, True]
    cpc = [809, 731, 588, 18, 200, 70, 45]

    data = {
        'country': names,
        'drives_right': dr,
        'cars_per_cap': cpc
    }

    df_vehicles = pd.DataFrame(data)
    print(df_vehicles)



if __name__ == '__main__':
    main()

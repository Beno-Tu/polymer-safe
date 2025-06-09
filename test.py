import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def main():
    PathRoot = r"C:\\Users\\nycu_dev1\\Desktop\\YEN-HSU Code\\"

    preprocess_data1 = pd.read_pickle(PathRoot + "preprocess_data/" + f'preprocess_data_{2018}_{10}_to_{2019}_{9}.pickle')
    preprocess_data2 = pd.read_pickle(PathRoot + "preprocess_data/" + f'preprocess_data_{2019}_{10}_to_{2020}_{9}.pickle')
    preprocess_data3 = pd.read_pickle(PathRoot + "preprocess_data/" + f'preprocess_data_{2020}_{10}_to_{2021}_{9}.pickle')
    preprocess_data4 = pd.read_pickle(PathRoot + "preprocess_data/" + f'preprocess_data_{2021}_{10}_to_{2022}_{9}.pickle')
    preprocess_data4 = preprocess_data4[preprocess_data4['Date'] < 20220101]
    preprocess_data = pd.concat([preprocess_data1, preprocess_data2, preprocess_data3, preprocess_data4])
    preprocess_data.reset_index(drop=True, inplace=True)
    preprocess_data = preprocess_data.dropna()
    print(preprocess_data)
    print(preprocess_data.columns.to_list)

    PathRoot = r"C:\\Users\\nycu_dev1\\Desktop\\YEN-HSU Code\\formationtable\\"

    # Date interval of data
    start_date = datetime(2018, 10, 19)
    end_date = datetime(2022, 1, 1) 

    # a list to store all df
    all_data = []

    # check if file exist
    current_date = start_date
    while current_date <= end_date:
        file_path = PathRoot + current_date.strftime('%Y%m%d') + 'for150del16_AB.csv'

        if os.path.exists(file_path):
            pre_df = pd.read_csv(file_path)
            # Add date column from file name
            pre_df.insert(0, 'Date', current_date.strftime('%Y%m%d'))
            all_data.append(pre_df)  # add df into list
        current_date += timedelta(days=1)

    # Merge all df in list
    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        df = df[["Date", "S1", "S2","Johansen_intercept", "Johansen_slope", "Johansen_std"]]
        filtered_df=df[df['Johansen_slope']==0]
        filtered_df = filtered_df[["Date", "S1", "S2","Johansen_intercept", "Johansen_std"]] 
        print("all files merged")
    else:
        df = pd.DataFrame()
        print("no file found")

    print(filtered_df)

    #Initialize m_df
    m_df_columns = ['Date', 'S1', 'S2', 'Norm_Spread', 'S1_Return', 'S2_Return', 'Revert',
                    'Norm_Rtop', 'Norm_Top', 'Norm_Close', 'Norm_Tax', 'Johanson_intercept', 'Johanson_std']
    m_df = pd.DataFrame(columns=m_df_columns)

    # Compare preprocess_data and filtered_df
    if not preprocess_data.empty and not filtered_df.empty:
        for _, pre_row in preprocess_data.iterrows():
            for _, filt_row in filtered_df.iterrows():
                if (pre_row['Date'] == filt_row['Date'] and 
                    pre_row['S1'] == filt_row['S1'] and 
                    pre_row['S2'] == filt_row['S2']):
                    new_row = {
                        'Date': pre_row['Date'],
                        'S1': pre_row['S1'],
                        'S2': pre_row['S2'],
                        'Norm_Spread': pre_row['Norm_Spread'],
                        'S1_Return': pre_row['S1_Return'],
                        'S2_Return': pre_row['S2_Return'],
                        'Revert': pre_row['Revert'],
                        'Norm_Rtop': pre_row['Norm_Rtop'],
                        'Norm_Top': pre_row['Norm_Top'],
                        'Norm_Close': pre_row['Norm_Close'],
                        'Norm_Tax': pre_row['Norm_Tax'],
                        'Johanson_intercept': filt_row['Johansen_intercept'],
                        'Johanson_std': filt_row['Johansen_std']
                    }
                    m_df = m_df.append(new_row, ignore_index=True)

    print("Final m_df:")
    print(m_df)

    

if __name__ == "__main__":
    main()
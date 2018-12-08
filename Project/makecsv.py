import os
import pandas as pd
import datetime
from functools import reduce
import numpy as np

DATE = 'date'
END_DATE = datetime.date(2018, 11, 5)
START_DATE = datetime.date(1987, 5, 20)
ALL_X = 'all_x.csv'
ALL_RAW = 'all.csv'
ALL_FILLED = 'all_filled.csv'
ALL_FINAL = 'all_final.csv'


def create_data_frame_by_type(path, key, how):
    dfs = []
    for file in os.listdir(path):
        current = os.path.join(path, file)
        if os.path.isfile(current):
            df = pd.read_csv(current)
            dfs.append(df)
    df_final = reduce(lambda left, right: pd.merge(left, right, on=key, how=how), dfs)
    df_final.to_csv(ALL_X, encoding='utf-8', index=False)
    return df_final

def extract_source(path, key, column_num, new_path):
    for file in os.listdir(path):
        current = os.path.join(path, file)
        x_name = file[:-4]
        if os.path.isfile(current):
            print('Extracting ', current)
            df = pd.read_csv(current, dtype={column_num: 'float64'})
            df[key] = pd.to_datetime(df[key]).dt.date
            df.dropna(subset=[df.columns[column_num]], inplace=True)
            df.drop(df[df[key] < START_DATE].index, inplace=True)
            df.rename(index=str, columns={key: DATE, df.columns[column_num]: x_name}, inplace=True)
            new_file = os.path.join(new_path, file)
            df.to_csv(new_file, columns=[DATE, x_name], encoding='utf-8', index=False)

def generate_nan(column_data, row_id):
    inc = 1
    while pd.isna(column_data[row_id - inc]):
        inc -= 1
    before = column_data[row_id - inc]
    inc = 1
    while pd.isna(column_data[row_id + inc]):
        inc += 1
    after = column_data[row_id + inc]
    column_data[row_id] = abs(after + before) / 2


def generate_positive_negative(yesterday, today):
    if today > yesterday:
        return True
    else:
        return False


def generate_percentage(yesterday, today):
    if today > yesterday:
        return (today - yesterday) / yesterday * 100
    else:
        return -(yesterday - today) / yesterday * 100


def extract_raw(new_path_x, new_path_y):
    extract_source('eia', 'Day', 1, new_path_x)
    extract_source('Fred', 'DATE', 1, new_path_x)
    extract_source('macrotrend', 'date', 1, new_path_x)
    extract_source('yahoofinance', 'Date', 4, new_path_x)
    extract_source('y', 'Day', 1, new_path_y)
    extract_source('weekly','Date', 1, new_path_x)


def create_raw_all(df_x, df_y):
    all_data = pd.merge(df_x, df_y, how='right', on=DATE)
    all_data[DATE] = pd.to_datetime(all_data[DATE]).dt.date
    all_data.sort_values(by=[DATE], inplace=True)
    all_data = all_data.reset_index(drop=True)
    all_data.to_csv(ALL_RAW, encoding='utf-8', index=False)

def create_metrics(all_data):
    all_data['Brent_spot_Price_pct_change_t_vs_t_minus_5'] = all_data['Brent_Spot_Price'] / all_data['Brent_Spot_Price'].shift(5) - 1
    all_data['Brent_spot_Price_pct_change_t_vs_t_minus_25'] = all_data['Brent_Spot_Price'] / all_data['Brent_Spot_Price'].shift(25) - 1
    all_data['Brent_spot_Price_pct_change_t_vs_t_minus_130'] = all_data['Brent_Spot_Price'] / all_data['Brent_Spot_Price'].shift(130) - 1

    all_data['Brent_Spot_Price_pct_change_t_minus_1_vs_t_minus_2'] = all_data['Brent_Spot_Price_pct_change'].shift(1)
    all_data['Brent_Spot_Price_pct_change_t_minus_2_vs_t_minus_3'] = all_data['Brent_Spot_Price_pct_change'].shift(2)
    all_data['Brent_Spot_Price_pct_change_t_minus_3_vs_t_minus_4'] = all_data['Brent_Spot_Price_pct_change'].shift(3)
    all_data['Brent_Spot_Price_pct_change_t_minus_4_vs_t_minus_5'] = all_data['Brent_Spot_Price_pct_change'].shift(4)
    all_data['Brent_Spot_Price_pct_change_t_minus_5_vs_5_minus_6'] = all_data['Brent_Spot_Price_pct_change'].shift(5)

    all_data['higher_1_day_later'] = 0
    all_data['higher_5_days_later'] = 0
    all_data['higher_25_days_later'] = 0
    all_data['higher_130_days_later'] = 0
    all_data['higher_average_1_to_5_days_later'] = 0
    all_data['higher_average_1_to_25_days_later'] = 0
    all_data['higher_average_1_to_130_days_later'] = 0

    all_data['sustained_1_day_later'] = 0
    all_data['sustained_5_days_later'] = 0
    all_data['sustained_25_days_later'] = 0
    all_data['sustained_130_days_later'] = 0
    all_data['sustained_average_1_to_5_days_later'] = 0
    all_data['sustained_average_1_to_25_days_later'] = 0
    all_data['sustained_average_1_to_130_days_later'] = 0

    for i in range(all_data.shape[0] - 130):
        if all_data.loc[i + 1, "Brent_Spot_Price"] > all_data.loc[i, "Brent_Spot_Price"]:
            all_data.loc[i, 'higher_1_day_later'] = 1
        if all_data.loc[i + 5, "Brent_Spot_Price"] > all_data.loc[i, "Brent_Spot_Price"]:
            all_data.loc[i, 'higher_5_days_later'] = 1
        if all_data.loc[i + 25, "Brent_Spot_Price"] > all_data.loc[i, "Brent_Spot_Price"]:
            all_data.loc[i, 'higher_25_days_later'] = 1
        if all_data.loc[i + 130, "Brent_Spot_Price"] > all_data.loc[i, "Brent_Spot_Price"]:
            all_data.loc[i, 'higher_130_days_later'] = 1
        if np.mean(all_data.loc[(i+1):(i+5), "Brent_Spot_Price"]) > all_data.loc[i, "Brent_Spot_Price"]:
            all_data.loc[i, 'higher_average_1_to_5_days_later'] = 1
        if np.mean(all_data.loc[(i+1):(i+25), "Brent_Spot_Price"]) > all_data.loc[i, "Brent_Spot_Price"]:
            all_data.loc[i, 'higher_average_1_to_25_days_later'] = 1
        if np.mean(all_data.loc[(i+1):(i+130), "Brent_Spot_Price"]) > all_data.loc[i, "Brent_Spot_Price"]:
            all_data.loc[i, 'higher_average_1_to_130_days_later'] = 1

        if all_data.loc[i, "Brent_Spot_Price_pct_change"] > 0:
            all_data.loc[i, 'sustained_1_day_later'] = all_data.loc[i, 'higher_1_day_later']
            all_data.loc[i, 'sustained_5_days_later'] = all_data.loc[i, 'higher_5_days_later']
            all_data.loc[i, 'sustained_25_days_later'] = all_data.loc[i, 'higher_25_days_later']
            all_data.loc[i, 'sustained_130_days_later'] = all_data.loc[i, 'higher_130_days_later']
            all_data.loc[i, 'sustained_average_1_to_5_days_later'] = all_data.loc[i, 'higher_average_1_to_5_days_later']
            all_data.loc[i, 'sustained_average_1_to_25_days_later'] = all_data.loc[i, 'higher_average_1_to_25_days_later']
            all_data.loc[i, 'sustained_average_1_to_130_days_later'] = all_data.loc[i, 'higher_average_1_to_130_days_later']
        else:
            all_data.loc[i, 'sustained_1_day_later'] = 1 - all_data.loc[i, 'higher_1_day_later']
            all_data.loc[i, 'sustained_5_days_later'] = 1 - all_data.loc[i, 'higher_5_days_later']
            all_data.loc[i, 'sustained_25_days_later'] = 1 - all_data.loc[i, 'higher_25_days_later']
            all_data.loc[i, 'sustained_130_days_later'] = 1 - all_data.loc[i, 'higher_130_days_later']
            all_data.loc[i, 'sustained_average_1_to_5_days_later'] = 1 - all_data.loc[i, 'higher_average_1_to_5_days_later']
            all_data.loc[i, 'sustained_average_1_to_25_days_later'] = 1 - all_data.loc[i, 'higher_average_1_to_25_days_later']
            all_data.loc[i, 'sustained_average_1_to_130_days_later'] = 1 - all_data.loc[i, 'higher_average_1_to_130_days_later']
    return all_data


if __name__ == "__main__":
    # create process raw data
    new_path_x = 'process_raw'
    if not os.path.exists(new_path_x):
        os.makedirs(new_path_x)

    new_path_y = 'process_y'
    if not os.path.exists(new_path_y):
        os.makedirs(new_path_y)

    extract_raw(new_path_x, new_path_y)

    df_x = pd.DataFrame({DATE: []})
    df_x = create_data_frame_by_type(new_path_x, DATE, 'outer')
    df_y = pd.read_csv(new_path_y + "/Brent_Spot_Price.csv")
    create_raw_all(df_x, df_y)

    all_data = pd.read_csv(ALL_RAW,dtype={1: 'float64'})

    # fill holes
    all_data = all_data.interpolate()
    all_data.to_csv(ALL_FILLED, encoding='utf-8', index=False)

    #percent change
    all_data_no_date = all_data.set_index('date',drop='True')
    all_pct_change = all_data_no_date.pct_change()
    all_data_with_pct_change = pd.merge(all_data, all_pct_change, on=DATE, suffixes=('','_pct_change'))

    all_final = create_metrics(all_data_with_pct_change)
    all_final.to_csv(ALL_FINAL, encoding='utf-8', index=False)

    reduced_rows = range(130,(all_final.shape[0] - 130))
    reduced_columns = [all_final.columns.get_loc(col) for col in all_final.columns if 'pct_change' in col or 'sustained' in col or 'Brent' in col]
    all_reduced = all_final.iloc[reduced_rows,reduced_columns]
    all_reduced.to_csv('all_reduced.csv', encoding='utf-8', index=False)

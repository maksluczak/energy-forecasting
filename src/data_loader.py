import os
import pandas as pd
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

def load_raw_data_and_export_to_csv() -> None:
    # load raw .txt file, manage NaNs and empty records
    df = pd.read_csv(
        RAW_DATA_PATH,
        sep=';',
        na_values=['?', 'NA', '', ' '],
        low_memory=False,
    )

    # connect date and time into one datetime object
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df = df.drop(columns=['Date', 'Time']).set_index('datetime')

    df = df.apply(pd.to_numeric, errors='coerce')
    df.ffill(inplace=True)

    # aggregation to hourly data
    df_hourly = df.resample('h').mean()

    # save to .csv file
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_hourly.to_csv(PROCESSED_DATA_PATH)

def load_clean_data() -> pd.DataFrame:
    if not os.path.exists(PROCESSED_DATA_PATH):
        load_raw_data_and_export_to_csv()

    # return .csv data as dataframe
    return pd.read_csv(PROCESSED_DATA_PATH, index_col='datetime', parse_dates=True)
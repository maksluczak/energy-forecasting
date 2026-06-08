import os
import pandas as pd
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

def load_raw_data_and_export_to_csv():
    df = pd.read_csv(
        RAW_DATA_PATH,
        sep = ";",
        low_memory = False,
        na_values = ["?"]
    )

    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')

    df.set_index('datetime', inplace=True)
    df.drop(columns=['Date', 'Time'], inplace=True)
    df.ffill(inplace=True)

    df_hourly = df['Global_active_power'].resample('h').mean().to_frame()

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_hourly.to_csv(PROCESSED_DATA_PATH)

def load_clean_data():
    if not os.path.exists(PROCESSED_DATA_PATH):
        load_raw_data_and_export_to_csv()

    return pd.read_csv(PROCESSED_DATA_PATH, index_col = 'datetime', parse_dates = True)
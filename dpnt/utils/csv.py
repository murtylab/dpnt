import os

import pandas as pd


def strip_if_str(x):
    return x.strip() if isinstance(x, str) else x


def read_csv(csv_filepath):
    try:
        df = pd.read_csv(csv_filepath, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map(strip_if_str)
        return df
    except Exception as e:
        print(f"Error reading CSV file at {csv_filepath}: {e}")
        return None

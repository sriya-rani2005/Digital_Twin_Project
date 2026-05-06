import pandas as pd
import numpy as np

# Column names for the dataset
columns = ['unit_id', 'cycle', 'op1', 'op2', 'op3'] + \
          [f's{i}' for i in range(1, 22)]

def load_data(filepath):
    df = pd.read_csv(filepath, sep='\s+', header=None, names=columns)
    df.dropna(axis=1, inplace=True)
    return df

def add_rul(df):
    max_cycle = df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycle.columns = ['unit_id', 'max_cycle']
    df = df.merge(max_cycle, on='unit_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop(columns=['max_cycle'], inplace=True)
    # Cap RUL at 125 (standard for CMAPSS)
    df['RUL'] = df['RUL'].clip(upper=125)
    return df

def normalize(df):
    sensor_cols = [f's{i}' for i in range(1, 22) if f's{i}' in df.columns]
    for col in sensor_cols:
        col_range = df[col].max() - df[col].min()
        if col_range > 0:
            df[col] = (df[col] - df[col].min()) / col_range
        else:
            df[col] = 0  # drop constant sensors
    return df

if __name__ == "__main__":
    df = load_data('data/CMaps/train_FD001.txt')
    df = add_rul(df)
    df = normalize(df)
    print("Data loaded successfully!")
    print(df.head())
    print(f"Shape: {df.shape}")
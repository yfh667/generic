from pathlib import Path
import pandas as pd

def read_csv_generic(csv_path, **kwargs):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, **kwargs)
    return df

#
# df = read_csv_generic(r"D:\paper3\xxx.csv")
# print(df.columns.tolist())
# print(df.head())

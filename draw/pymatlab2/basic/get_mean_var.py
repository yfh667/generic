import pandas as pd
import numpy as np
def calc_mean_std(df_two_cols: pd.DataFrame):
    """
    df_two_cols: 形如 [time, value] 的 DataFrame（两列）
    返回: (mean, std)，这里 std 用的是样本标准差（ddof=1）
    """
    # 第二列视为“计量值”
    values = pd.to_numeric(df_two_cols.iloc[:, 1], errors='coerce')
    values = values.dropna()

    mean_val = float(values.mean())
    std_val  = float(values.std(ddof=1))  # 如果想要总体标准差改成 ddof=0

    return mean_val, std_val
from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np

@dataclass
class StableInterval:
    interval: tuple[int, int]   # (start_t, end_t)
    edges: float                # 区间代表 link_count（平均/中位/起点）
    length: float               # 区间时长

def extract_stable_intervals(df: pd.DataFrame, rep_stat: str = "mean") -> List[StableInterval]:
    """
    输入：
        df：包含列 ['t', 'link_count', 'changed']
        rep_stat：区间代表link_count的计算方式 ('mean', 'median', 'first', 'last')
    输出：
        qujian: List[StableInterval]
    """
    t = df["t"].to_numpy()
    y = df["link_count"].to_numpy()
    c = df["changed"].to_numpy(int)
    dt = np.median(np.diff(t)) if len(t) >= 2 else 1.0

    qujian: List[StableInterval] = []
    start_i = 0
    for i in range(len(t)):
        if c[i] == 1 and i > start_i:
            seg_t = t[start_i:i]
            seg_y = y[start_i:i]
            if len(seg_t) > 0:
                start_t, end_t = seg_t[0], t[i]
                if rep_stat == "mean":
                    edges = float(np.mean(seg_y))
                elif rep_stat == "median":
                    edges = float(np.median(seg_y))
                elif rep_stat == "first":
                    edges = float(seg_y[0])
                elif rep_stat == "last":
                    edges = float(seg_y[-1])
                else:
                    raise ValueError("rep_stat 必须是 mean/median/first/last")
                qujian.append(StableInterval((int(start_t), int(end_t)), edges, end_t - start_t))
            start_i = i

    # 最后一个区间（无后续变化）
    if start_i < len(t):
        seg_t, seg_y = t[start_i:], y[start_i:]
        if len(seg_t) > 0:
            start_t, end_t = seg_t[0], seg_t[-1] + dt
            if rep_stat == "mean":
                edges = float(np.mean(seg_y))
            elif rep_stat == "median":
                edges = float(np.median(seg_y))
            elif rep_stat == "first":
                edges = float(seg_y[0])
            elif rep_stat == "last":
                edges = float(seg_y[-1])
            qujian.append(StableInterval((int(start_t), int(end_t)), edges, end_t - start_t))

    return qujian

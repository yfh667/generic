
from typing import Any, Dict, List, Optional, Tuple

def longest_zero_window_from_edges(edges: List[Optional[int]], t_min: int
                                  ) -> Optional[Tuple[int, int, int, float]]:
    """
    在给定 edges 序列中（按 t_min 偏移索引），计算 edge==0 的最长连续区间。
    - None 视为“断点”（不会计入连续段）
    返回:
        (start_t, end_t, dur_seconds, dur_minutes)；若不存在则返回 None
    """
    best_len = 0
    best_start_idx: Optional[int] = None

    cur_len = 0
    cur_start_idx: Optional[int] = None

    for i, val in enumerate(edges):
        if val == 0:  # 只统计严格等于 0
            if cur_len == 0:
                cur_start_idx = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start_idx = cur_start_idx
        else:
            cur_len = 0
            cur_start_idx = None

    if best_len > 0 and best_start_idx is not None:
        start_t = t_min + best_start_idx
        end_t = start_t + best_len - 1
        dur_s = best_len
        dur_min = dur_s / 60.0
        return start_t, end_t, dur_s, dur_min
    else:
        return None


def list_zero_windows(edges: List[Optional[int]], t_min: int
                     ) -> List[Tuple[int, int, int]]:
    """
    枚举所有 edge==0 的连续窗口（None 视为断点）。
    返回: [(start_t, end_t, length_seconds), ...]
    """
    windows: List[Tuple[int, int, int]] = []
    cur_start_idx = None
    cur_len = 0

    for i, v in enumerate(edges):
        if v == 0:
            if cur_len == 0:
                cur_start_idx = i
            cur_len += 1
        else:
            if cur_len > 0 and cur_start_idx is not None:
                start_t = t_min + cur_start_idx
                end_t = start_t + cur_len - 1
                windows.append((start_t, end_t, cur_len))
            cur_start_idx = None
            cur_len = 0

    # 收尾
    if cur_len > 0 and cur_start_idx is not None:
        start_t = t_min + cur_start_idx
        end_t = start_t + cur_len - 1
        windows.append((start_t, end_t, cur_len))

    return windows


def average_zero_window_duration(edges: List[Optional[int]], t_min: int
                                ) -> Optional[Tuple[float, float, int]]:
    """
    计算所有 zero-edge 连续窗口的平均时长。
    返回: (avg_seconds, avg_minutes, count_windows)
          若没有窗口则返回 None
    """
    windows = list_zero_windows(edges, t_min)
    if not windows:
        return None
    total = sum(L for _, _, L in windows)
    count = len(windows)
    avg_s = total / count
    return avg_s, avg_s / 60.0, count
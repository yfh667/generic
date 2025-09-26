from typing import Any, Dict, List, Optional, Tuple

def analyze_time_continuity(mapping: Dict[Any, Any]) -> Tuple[int, int, List[int], List[Tuple[int, int]]]:
    """
    输入:
        mapping: 由 df.set_index('time')['pending_edges'].to_dict() 得到的 {time -> edge} 映射
    处理:
        - 将 time 强制为 int，edge 尽量转为 int, 无法转的置为 None
        - 排序后检测相邻 time 是否连续，找出所有缺口区间 [a+1, b-1]
    返回:
        t_min, t_max, times(升序不重复), gaps(缺口段列表)
    """
    # 1) 规范化
    norm: Dict[int, Optional[int]] = {}
    for k, v in mapping.items():
        try:
            t = int(k)
        except Exception:
            continue
        if v is None:
            norm[t] = None
        else:
            try:
                norm[t] = int(v)
            except Exception:
                try:
                    norm[t] = int(float(v))
                except Exception:
                    norm[t] = None

    if not norm:
        raise ValueError("mapping 为空或无法规范化出有效的 time。")

    # 2) 连续性检测
    times = sorted(norm.keys())
    gaps: List[Tuple[int, int]] = []
    for a, b in zip(times, times[1:]):
        if b - a > 1:
            gaps.append((a + 1, b - 1))

    t_min, t_max = times[0], times[-1]
    return t_min, t_max, times, gaps


def build_edges_from_mapping(mapping: Dict[Any, Any], fill_missing: Optional[int] = None
                            ) -> Tuple[List[Optional[int]], int, int]:
    """
    将 {time -> edge} 转为长度为 (t_max-t_min+1) 的列表 edges（偏移索引），并返回 t_min, t_max
    参数:
        fill_missing:
            - None: 缺秒保持 None（当作断点）
            - 0 等整数: 用该值填补缺秒
    返回:
        edges, t_min, t_max
    """
    t_min, t_max, times, _ = analyze_time_continuity(mapping)

    # 先做一次规范化，得到 {int time -> Optional[int edge]}
    norm: Dict[int, Optional[int]] = {}
    for k, v in mapping.items():
        try:
            t = int(k)
        except Exception:
            continue
        if v is None:
            norm[t] = None
        else:
            try:
                norm[t] = int(v)
            except Exception:
                try:
                    norm[t] = int(float(v))
                except Exception:
                    norm[t] = None

    size = t_max - t_min + 1
    edges: List[Optional[int]] = ([fill_missing] * size) if fill_missing is not None else ([None] * size)

    for t, e in norm.items():
        idx = t - t_min
        edges[idx] = e if e is not None else (edges[idx] if fill_missing is None else fill_missing)

    return edges, t_min, t_max

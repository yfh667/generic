

from typing import Dict, Tuple, Optional

from typing import Dict, Tuple, Optional, List, Set


def calc_envelope_for_group(
        group_data: Dict[int, dict],
        interval: Tuple[int, int],
        groupid: int,
        P: int,
        N: int
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
    """
    计算指定时间区间内、指定 group 的包络矩形。
    使用智能分割算法处理跨 x=0 和 x=P-1 边界的情况。

    返回统一为 (left_box, right_box)：
      - 不跨缝：left_box = 单个矩形, right_box = None
      - 跨缝：  left_box, right_box = 各自的矩形
      - 没有数据：返回 (None, None)
    """
    t0, t1 = interval
    times = [t for t in sorted(group_data.keys()) if t0 <= t <= t1]
    if not times:
        return None, None

    # --- 收集该群组在时间段内所有出现过的点 ---
    all_sats: Set[int] = set()
    for t in times:
        sats = group_data.get(t, {}).get('groups', {}).get(groupid, set())
        all_sats.update(sats)

    if not all_sats:
        return None, None

    all_x = [sid // N for sid in all_sats]
    all_y = [sid % N for sid in all_sats]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # --- 检查是否跨缝 ---
    # 条件：群组中同时存在 x=0 和 x=P-1 的点
    left_seam = 0 in all_x
    right_seam = (P - 1) in all_x

    # --- 情况1：不跨缝，直接返回一个大矩形 ---
    if not (left_seam and right_seam):
        return (x_min, x_max, y_min, y_max), None

    # --- 情况2：跨缝，需要进行智能分割 ---

    # 1. 获取所有唯一的、排序后的x坐标
    unique_x = sorted(list(set(all_x)))

    # 如果跨缝但只有一列或两列，也视为一个整体
    if len(unique_x) <= 2:
        return (x_min, x_max, y_min, y_max), None

    # 2. 计算所有x坐标之间的间隙，找到最大的那个
    gaps = []
    for i in range(len(unique_x) - 1):
        gaps.append((unique_x[i + 1] - unique_x[i], i))  # (间隙大小, 间隙前一个点的索引)

    # 计算从最后一个点绕回第一个点的跨缝间隙
    wraparound_gap = (unique_x[0] + P) - unique_x[-1]
    gaps.append((wraparound_gap, len(unique_x) - 1))

    max_gap, split_index = max(gaps)

    # 3. 根据最大间隙，将x坐标分为两组
    # right_x_values 对应于原图中右侧的独立小群组 (例如 [23])
    # left_x_values 对应于原图中左侧的主体大群组 (例如 [0, 1, 2, ..., 8])
    right_x_values = set(unique_x[split_index + 1:])
    left_x_values = set(unique_x[:split_index + 1])

    # 4. 根据x坐标分组，为两组点分别计算包络矩形
    def _get_box_from_points(points: Set[int]) -> Optional[Tuple[int, int, int, int]]:
        if not points:
            return None
        xs = [sid // N for sid in points]
        ys = [sid % N for sid in points]
        return (min(xs), max(xs), min(ys), max(ys))

    left_points = {sid for sid in all_sats if sid // N in left_x_values}
    right_points = {sid for sid in all_sats if sid // N in right_x_values}

    left_box = _get_box_from_points(left_points)
    right_box = _get_box_from_points(right_points)

    # 保证包含 x=0 的一定是 left_box
    if left_box and 0 not in range(left_box[0], left_box[1] + 1):
        left_box, right_box = right_box, left_box

    return left_box, right_box
#
#
# def calc_envelope_for_group(
#     group_data: Dict[int, dict],
#     interval: Tuple[int, int],
#     groupid: int,
#     P: int,
#     N: int
# ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
#     """
#     计算指定时间区间内、指定 group 的包络矩形。
#
#     返回统一为 (left_box, right_box)：
#       - 不跨缝：left_box = 单个矩形, right_box = None
#       - 跨缝：   left_box, right_box = 各自的矩形
#       - 没有数据：返回 (None, None)
#     """
#
#     t0, t1 = interval
#     times = [t for t in sorted(group_data.keys()) if t0 <= t <= t1]
#     if not times:
#         return None, None
#
#     left_seam, right_seam = False, False
#     x_min = y_min = None
#     x_max = y_max = None
#
#     # --- 检查是否跨缝 ---
#     for t in times:
#         g = group_data.get(t, {})
#         groups = g.get('groups', {})
#         sats = groups.get(groupid, set())
#         if not sats:
#             continue
#
#         xs = [sid // N for sid in sats]
#         ys = [sid % N for sid in sats]
#
#         if 0 in xs:
#             left_seam = True
#         if (P - 1) in xs:
#             right_seam = True
#
#         _xmin, _xmax = min(xs), max(xs)
#         _ymin, _ymax = min(ys), max(ys)
#
#         x_min = _xmin if x_min is None else min(x_min, _xmin)
#         x_max = _xmax if x_max is None else max(x_max, _xmax)
#         y_min = _ymin if y_min is None else min(y_min, _ymin)
#         y_max = _ymax if y_max is None else max(y_max, _ymax)
#
#     # --- 不跨缝：只返回一个矩形 ---
#     if not (left_seam and right_seam):
#         if x_min is None:
#             return None, None
#         return (x_min, x_max, y_min, y_max), None
#
#     # --- 跨缝：分两块 ---
#     split_plane = P // 2
#     left_box  = [None, None, None, None]
#     right_box = [None, None, None, None]
#
#     def _update(box, x_val, y_val):
#         box[0] = x_val if box[0] is None else min(box[0], x_val)
#         box[1] = x_val if box[1] is None else max(box[1], x_val)
#         box[2] = y_val if box[2] is None else min(box[2], y_val)
#         box[3] = y_val if box[3] is None else max(box[3], y_val)
#
#     for t in times:
#         sats = group_data.get(t, {}).get('groups', {}).get(groupid, set())
#         for sid in sats:
#             x_sid, y_sid = divmod(sid, N)
#             if x_sid >= split_plane:
#                 _update(right_box, x_sid, y_sid)
#             else:
#                 _update(left_box, x_sid, y_sid)
#
#     def _finalize(box):
#         if box[0] is None:
#             return None
#         return tuple(int(v) for v in box)
#
#     return _finalize(left_box), _finalize(right_box)


#t1,t2=get_rectangular_size_interval.calc_envelope_for_group(rev_group_data,[start_ts,end_ts],4,P,N)


import matplotlib.pyplot as plt
import pandas as pd

def get_envelope_traces(
    group_data,
    groupid,
    P,
    N,
    t0=None,
    t1=None
):
    """
    返回：DataFrame，每行是
    time, left_xmin, left_xmax, left_ymin, left_ymax, right_xmin, right_xmax, right_ymin, right_ymax
    没有的地方是 nan
    """
    times = sorted(group_data.keys())
    if t0 is not None:
        times = [t for t in times if t >= t0]
    if t1 is not None:
        times = [t for t in times if t <= t1]
    records = []
    for t in times:
        g = group_data[t]
        sats = g.get('groups', {}).get(groupid, set())
        if not sats:
            records.append([t]+[None]*8)
            continue

        xs = [sid // N for sid in sats]
        ys = [sid % N for sid in sats]
        left_seam = 0 in xs
        right_seam = (P - 1) in xs
        if not (left_seam and right_seam):
            l = (min(xs), max(xs), min(ys), max(ys))
            record = [t, *l, None, None, None, None]
        else:
            split_plane = P // 2
            lx = [sid // N for sid in sats if sid // N < split_plane]
            ly = [sid % N for sid in sats if sid // N < split_plane]
            rx = [sid // N for sid in sats if sid // N >= split_plane]
            ry = [sid % N for sid in sats if sid // N >= split_plane]
            l = (min(lx), max(lx), min(ly), max(ly)) if lx else (None, None, None, None)
            r = (min(rx), max(rx), min(ry), max(ry)) if rx else (None, None, None, None)
            record = [t, *l, *r]
        records.append(record)
    df = pd.DataFrame(records, columns=[
        'time',
        'left_xmin','left_xmax','left_ymin','left_ymax',
        'right_xmin','right_xmax','right_ymin','right_ymax'
    ])
    return df

# --- 使用示例 ---
# df = get_envelope_traces(group_data, groupid=4, P=18, N=36, t0=0, t1=1203)

def plot_envelope_traces(df, title='Group Envelope Evolution'):
    fig, axs = plt.subplots(2, 1, figsize=(12,7), sharex=True)
    # 左包络
    axs[0].set_title("Left Envelope")
    axs[0].plot(df['time'], df['left_xmin'], label='xmin')
    axs[0].plot(df['time'], df['left_xmax'], label='xmax')
    axs[0].plot(df['time'], df['left_ymin'], label='ymin', linestyle='--')
    axs[0].plot(df['time'], df['left_ymax'], label='ymax', linestyle='--')
    axs[0].legend()
    axs[0].set_ylabel("Plane / Node")

    # 右包络
    axs[1].set_title("Right Envelope")
    axs[1].plot(df['time'], df['right_xmin'], label='xmin')
    axs[1].plot(df['time'], df['right_xmax'], label='xmax')
    axs[1].plot(df['time'], df['right_ymin'], label='ymin', linestyle='--')
    axs[1].plot(df['time'], df['right_ymax'], label='ymax', linestyle='--')
    axs[1].legend()
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Plane / Node")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# # 1. 先算表
# df = get_envelope_traces(group_data, groupid=4, P=18, N=36, t0=0, t1=1203)
#
# # 2. 可直接保存df到csv
# df.to_csv('envelope_trace_group4.csv', index=False)
#
# # 3. 画随时间变化的包络范围
# plot_envelope_traces(df)

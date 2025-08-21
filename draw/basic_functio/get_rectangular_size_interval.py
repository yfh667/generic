

from typing import Dict, Tuple, Optional

def calc_envelope_for_group(
    group_data: Dict[int, dict],
    interval: Tuple[int, int],
    groupid: int,
    P: int,
    N: int
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
    """
    计算指定时间区间内、指定 group 的包络矩形。

    返回统一为 (left_box, right_box)：
      - 不跨缝：left_box = 单个矩形, right_box = None
      - 跨缝：   left_box, right_box = 各自的矩形
      - 没有数据：返回 (None, None)
    """

    t0, t1 = interval
    times = [t for t in sorted(group_data.keys()) if t0 <= t <= t1]
    if not times:
        return None, None

    left_seam, right_seam = False, False
    x_min = y_min = None
    x_max = y_max = None

    # --- 检查是否跨缝 ---
    for t in times:
        g = group_data.get(t, {})
        groups = g.get('groups', {})
        sats = groups.get(groupid, set())
        if not sats:
            continue

        xs = [sid // N for sid in sats]
        ys = [sid % N for sid in sats]

        if 0 in xs:
            left_seam = True
        if (P - 1) in xs:
            right_seam = True

        _xmin, _xmax = min(xs), max(xs)
        _ymin, _ymax = min(ys), max(ys)

        x_min = _xmin if x_min is None else min(x_min, _xmin)
        x_max = _xmax if x_max is None else max(x_max, _xmax)
        y_min = _ymin if y_min is None else min(y_min, _ymin)
        y_max = _ymax if y_max is None else max(y_max, _ymax)

    # --- 不跨缝：只返回一个矩形 ---
    if not (left_seam and right_seam):
        if x_min is None:
            return None, None
        return (x_min, x_max, y_min, y_max), None

    # --- 跨缝：分两块 ---
    split_plane = P // 2
    left_box  = [None, None, None, None]
    right_box = [None, None, None, None]

    def _update(box, x_val, y_val):
        box[0] = x_val if box[0] is None else min(box[0], x_val)
        box[1] = x_val if box[1] is None else max(box[1], x_val)
        box[2] = y_val if box[2] is None else min(box[2], y_val)
        box[3] = y_val if box[3] is None else max(box[3], y_val)

    for t in times:
        sats = group_data.get(t, {}).get('groups', {}).get(groupid, set())
        for sid in sats:
            x_sid, y_sid = divmod(sid, N)
            if x_sid >= split_plane:
                _update(right_box, x_sid, y_sid)
            else:
                _update(left_box, x_sid, y_sid)

    def _finalize(box):
        if box[0] is None:
            return None
        return tuple(int(v) for v in box)

    return _finalize(left_box), _finalize(right_box)


#t1,t2=get_rectangular_size_interval.calc_envelope_for_group(rev_group_data,[start_ts,end_ts],4,P,N)

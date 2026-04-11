from collections import defaultdict


def _find_components_by_neighbors(sats: set, config):
    """
    用并查集按邻接关系把 sats 划分为连通块。
    邻接：同 x 的 (y±1)%N；以及 x< P-1 时的右邻 (x+1,y)。
    返回: [set(sid), ...]，按规模降序。
    """
    P,N=config.P,config.N
    if not sats:
        return []

    sats = set(sats)
    parent = {sid: sid for sid in sats}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for sid in sats:
        x, y = divmod(sid, N)

        # 竖直相邻（y 环绕）
        up = x * N + ((y + 1) % N)
        down = x * N + ((y - 1 + N) % N)
        if up in sats: union(sid, up)
        if down in sats: union(sid, down)

        # 水平相邻（x 不环绕）
        if x < P - 1:
            right = (x + 1) * N + y
            if right in sats: union(sid, right)

    comps = defaultdict(set)
    for sid in sats:
        comps[find(sid)].add(sid)

    return sorted(comps.values(), key=len, reverse=True)


def _offset_from_component_y(sids: set, config) -> int:
    """
    在主簇的 y 值上找“最大的环形间隙”，
    取该间隙后面的 y 作为 start_y，并返回 offset=start_y。
    这样 y_new = (y - offset + N - 1) % N 会把 start_y 卷到最上面(N-1)。
    """
    N =config.N
    ys = sorted({sid % N for sid in sids})

    m = len(ys)
    if m == 0:
        return 0
    if m == 1:
        return ys[0]  # 只有一个点，自己就是起点

    # 计算环上的相邻间隙（含 wrap 间隙）
    # 找到环上的最大间隙，记下间隙后的位置
    max_gap = -1
    idx_after_gap = 0
    for i in range(m - 1):
        g = ys[i + 1] - ys[i]
        if g > max_gap:
            max_gap = g
            idx_after_gap = i + 1
    wrap_gap = (ys[0] + N) - ys[-1]
    if wrap_gap > max_gap:
        idx_after_gap = 0  # 最大间隙在 ys[-1] 与 ys[0] 之间

    # 注意：主簇的“最后一个 y”是最大间隙的前一个元素
    offset = ys[idx_after_gap - 1]  # 关键改动：不是 ys[idx_after_gap]
    return offset


def modify_group_data(group_data: dict,config,  base_groupid: int = 4):
    """
    先把 base_groupid 的点集按邻接分成连通块，
    选点数最多的“主簇”，在其 y 上找最大环形间隙确定 offset，
    再用统一 offset 对所有组做 y 平移： y_new = (y - offset + N - 1) % N
    """
    P, N = config.P, config.N
    min_keep = 0.6
    new_group_data = {}
    off_sets = {}

    prev_comp = None  # 仅记录上一次“被选中的蔟”
    for step in sorted(group_data.keys()):

        raw_groups = group_data[step]['groups']
        new_group_data[step] = {'groups': {}, 'all_mentioned': set()}

        base_sats = raw_groups.get(base_groupid, set())
        comps = _find_components_by_neighbors(base_sats, config)

        # if step ==13605:
        #     print(1)
        # if step==13606:
        #     print(1)

        if comps:
            cand = comps[:max(1, 2)]

            if prev_comp:
                # 选与 prev_comp 重叠比例最大的候选
                best_c, best_ratio = cand[0], -1.0
                for c in cand:
                    inter = len(prev_comp & c)
                    ratio = inter / len(prev_comp) if len(prev_comp) else 0.0
                    if ratio > best_ratio:
                        best_c, best_ratio = c, ratio
                chosen_comp = best_c if best_ratio >= min_keep else cand[0]
            else:
                chosen_comp = cand[0]

            chosen_offset = _offset_from_component_y(chosen_comp, config)
            prev_comp = chosen_comp  # 只记蔟，不记 offset
        else:
            # 本帧无蔟：沿用“上一次蔟”计算的 offset；若还没有任何蔟，置 0
            chosen_offset = _offset_from_component_y(prev_comp, config) if prev_comp else 0

        off_sets[step] = chosen_offset

        # 统一 offset 平移所有组
        for gid, sats in raw_groups.items():
            tgt = new_group_data[step]['groups'].setdefault(gid, set())
            for sid in sats:
                x, y = divmod(sid, N)
                y_new = (y - chosen_offset + N - 1) % N
                new_sid = x * N + y_new
                tgt.add(new_sid)
                new_group_data[step]['all_mentioned'].add(new_sid)

    return new_group_data, off_sets


def modify_data(time, number, off_sets,config):
    N= config.N
    x = number // N
    y = number % N
    y_new = (y - off_sets[time] + N - 1) % N
    new_sid = x * N + y_new
    return new_sid


def rev_modify_group_data(group_data, off_sets, config):
    N = config.N
    new_group_data = {}

    for step, raw_step_dict in group_data.items():  # ← 改这里
        raw_groups = raw_step_dict['groups']
        new_group_data[step] = {'groups': {}, 'all_mentioned': set()}

        for gid, sats in raw_groups.items():
            tgt_set = new_group_data[step]['groups'].setdefault(gid, set())
            for sid in sats:
                y = sid % N
                x = sid // N

                y_new = (y + off_sets[step] + 1) % N
                new_sid = x * N + y_new
                tgt_set.add(new_sid)
                new_group_data[step]['all_mentioned'].add(new_sid)

    return new_group_data


def rev_modify_data(time, number, off_sets, config):
    N = config.N
    y = number % N
    x = number // N

    y_new = (y + off_sets[time] + 1) % N
    new_sid = x * N + y_new

    return new_sid
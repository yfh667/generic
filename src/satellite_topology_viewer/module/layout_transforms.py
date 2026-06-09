from __future__ import annotations

"""Data-preparation helpers.

Viewer classes do not call these transforms. If a topology needs a shifted
layout, prepare shifted nodes, edges, and group data before creating the viewer.
"""


def _find_components_by_neighbors(sats: set[int], p_count: int, y_count: int) -> list[set[int]]:
    if not sats:
        return []

    sats = set(int(x) for x in sats)
    parent = {sid: sid for sid in sats}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for sid in sats:
        x, y = divmod(int(sid), int(y_count))
        up = x * y_count + ((y + 1) % y_count)
        down = x * y_count + ((y - 1 + y_count) % y_count)
        if up in sats:
            union(sid, up)
        if down in sats:
            union(sid, down)
        if x < int(p_count) - 1:
            right = (x + 1) * y_count + y
            if right in sats:
                union(sid, right)

    comps: dict[int, set[int]] = {}
    for sid in sats:
        comps.setdefault(find(sid), set()).add(sid)
    return sorted(comps.values(), key=len, reverse=True)


def _offset_from_component_y(sids: set[int] | None, y_count: int) -> int:
    if not sids:
        return 0
    ys = sorted({int(sid) % int(y_count) for sid in sids})
    if len(ys) == 1:
        return int(ys[0])

    max_gap = -1
    idx_after_gap = 0
    for idx in range(len(ys) - 1):
        gap = int(ys[idx + 1]) - int(ys[idx])
        if gap > max_gap:
            max_gap = gap
            idx_after_gap = idx + 1
    wrap_gap = int(ys[0]) + int(y_count) - int(ys[-1])
    if wrap_gap > max_gap:
        idx_after_gap = 0
    return int(ys[idx_after_gap - 1])


def build_rev_group_offsets(
    group_data: dict,
    p_count: int,
    y_count: int,
    base_groupid: int,
) -> dict[int, int]:
    min_keep = 0.6
    offsets: dict[int, int] = {}
    prev_comp: set[int] | None = None

    for step in sorted(int(x) for x in group_data.keys()):
        current = group_data.get(step, {}) if group_data else {}
        groups = current.get("groups", {}) if isinstance(current, dict) else {}
        base_sats = set(int(x) for x in (groups.get(int(base_groupid), set()) or set()))
        comps = _find_components_by_neighbors(base_sats, int(p_count), int(y_count))

        if comps:
            candidates = comps[:2]
            if prev_comp:
                best_comp = candidates[0]
                best_ratio = -1.0
                for comp in candidates:
                    ratio = len(prev_comp & comp) / len(prev_comp) if prev_comp else 0.0
                    if ratio > best_ratio:
                        best_comp = comp
                        best_ratio = ratio
                chosen_comp = best_comp if best_ratio >= min_keep else candidates[0]
            else:
                chosen_comp = candidates[0]
            chosen_offset = _offset_from_component_y(chosen_comp, int(y_count))
            prev_comp = chosen_comp
        else:
            chosen_offset = _offset_from_component_y(prev_comp, int(y_count)) if prev_comp else 0

        offsets[int(step)] = int(chosen_offset)

    return offsets

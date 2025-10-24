# --- 常量（WGS-84） ---
import json, math
from pathlib import Path

from pathlib import Path
import json
from typing import Iterable, Optional, Union, Dict, Any, Tuple, List

_WGS84_A = 6378137.0           # 半长轴 a (m)
_WGS84_E2 = 6.69437999014e-3   # 第一偏心率平方 e^2
_WGS84_B = _WGS84_A * math.sqrt(1 - _WGS84_E2)

def _ecef_to_lla(x, y, z):
    """
    将 ECEF (m) 转为 (lon, lat, alt) = (经度°, 纬度°, 海拔m)，WGS-84。
    采用常用的 Bowring 近似，精度满足可视化需求。
    """
    a = _WGS84_A
    e2 = _WGS84_E2
    b = _WGS84_B

    # 经度
    lon = math.degrees(math.atan2(y, x))

    # 中间量
    p = math.hypot(x, y)
    if p < 1e-9:  # 极点附近的稳健性
        lat = 90.0 if z >= 0 else -90.0
        N = a / math.sqrt(1 - e2 * math.sin(math.radians(lat))**2)
        alt = abs(z) - (a*(1 - e2))  # 近似
        return lon, lat, alt

    # Bowring 公式
    theta = math.atan2(z * a, p * b)
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    lat = math.atan2(z + (e2 * b) * (sin_t**3),
                     p - (e2 * a) * (cos_t**3))
    sin_lat = math.sin(lat)
    N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - N

    lat = math.degrees(lat)
    return lon, lat, alt



def _coerce_link_item(item) -> Dict[str, Any]:
    """
    将 link 条目规范化为 {"source": int, "target": int}
    支持 (s, t), [s, t], {"source": s, "target": t}
    其他字段（如 status/type）由外层处理。
    """
    if isinstance(item, (tuple, list)) and len(item) == 2:
        s, t = item
        return {"source": int(s), "target": int(t)}
    if isinstance(item, dict):
        if "source" in item and "target" in item:
            return {"source": int(item["source"]), "target": int(item["target"])}
    raise ValueError(f"Unrecognized link item: {item!r}")

def _split_links_by_status(
    links: Optional[Iterable[Union[Tuple[int, int], Dict[str, Any]]]]
) -> Tuple[List[Dict[str, int]], List[Dict[str, int]]]:
    """
    将混合 links（可能带 status/type 字段）拆成 (active_list, pending_list)。
    未带状态的条目默认为 active。
    """
    act, pend = [], []
    if not links:
        return act, pend
    for it in links:
        norm = _coerce_link_item(it)
        status = None
        if isinstance(it, dict):
            status = (it.get("status") or it.get("type") or "").strip().lower()
        if status in ("pending", "building", "setup"):
            pend.append(norm)
        else:
            act.append(norm)
    return act, pend

def _uniq_and_filter_existing(
    links: Iterable[Dict[str, int]], valid_ids: set
) -> List[Dict[str, int]]:
    """
    去重（无向/有向按你的需求选择，这里按有向去重），并过滤掉不在卫星集合里的端点。
    如需无向去重，把 key 改成 tuple(sorted((s,t))).
    """
    seen = set()
    out = []
    for lk in links:
        s, t = int(lk["source"]), int(lk["target"])
        if s not in valid_ids or t not in valid_ids:
            # 丢弃指向不存在卫星的链路
            continue
        key = (s, t)  # 如果你的链路是无向的，改成 tuple(sorted((s, t)))
        if key in seen:
            continue
        seen.add(key)
        out.append({"source": s, "target": t})
    return out




def export_satellites_snapshot_to_json(
    SatelliteManager,
    time_idx: int,
    out_path: Union[str, Path],
    *,
    ecef_in_km: bool = False,
    alt_unit: str = 'km',
    links: Optional[Iterable[Union[Tuple[int, int], Dict[str, Any]]]] = None,
    links_active: Optional[Iterable[Union[Tuple[int, int], Dict[str, Any]]]] = None,
    links_pending: Optional[Iterable[Union[Tuple[int, int], Dict[str, Any]]]] = None,
) -> str:
    """
    导出指定时间步的卫星 (lng, lat, alt) 和链路到 JSON。

    链路输入有两种用法（二选一或混用）：
    - 方式 A：分别传 links_active 与 links_pending；
    - 方式 B：传 links（元素可带 status/type='active'|'pending'，未标注默认为 active）。

    输出 JSON schema:
    {
      "satellites": [
        {"id": "12", "lng": 110.0, "lat": 30.0, "alt": 550.0}, ...
      ],
      "links_active":  [ {"source": 1, "target": 2}, ... ],
      "links_pending": [ {"source": 3, "target": 4}, ... ]
    }
    """
    sats = []
    scale = 1000.0 if ecef_in_km else 1.0

    # --- 导出卫星位置 ---
    for sid, sat in SatelliteManager.satellites.items():
        if time_idx >= len(sat.trajectory):
            raise IndexError(
                f"satellite {sid} has trajectory length {len(sat.trajectory)}, but time_idx={time_idx}"
            )
        p = sat.trajectory[time_idx]
        x, y, z = p.x * scale, p.y * scale, p.z * scale
        lon, lat, alt_m = _ecef_to_lla(x, y, z)
        alt_out = alt_m / 1000.0 if alt_unit == 'km' else alt_m
        sats.append({
            "id": str(int(sid)),
            "lng": float(lon),
            "lat": float(lat),
            "alt": float(alt_out)
        })

    valid_ids = {int(sid) for sid in SatelliteManager.satellites.keys()}

    # --- 归并两种输入形式的链路 ---
    act_from_single, pend_from_single = _split_links_by_status(links)
    act_from_two   = [_coerce_link_item(x) for x in (links_active or [])]
    pend_from_two  = [_coerce_link_item(x) for x in (links_pending or [])]

    links_active_all  = act_from_single + act_from_two
    links_pending_all = pend_from_single + pend_from_two

    # --- 去重 + 过滤不存在的卫星 ID ---
    links_active_all  = _uniq_and_filter_existing(links_active_all,  valid_ids)
    links_pending_all = _uniq_and_filter_existing(links_pending_all, valid_ids)

    payload = {
        "satellites": sats,
        "links_active": links_active_all,
        "links_pending": links_pending_all
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return str(out_path)


# usage :
# active = [(0,1), (1,2), {"source":2, "target":3}]
# pending = [(3,4), {"source":5, "target":6}]
# export_satellites_snapshot_to_json(SatelliteManager, 100, "out/snapshot_100.json",
#                                    links_active=active, links_pending=pending)


# def export_satellites_snapshot_to_json(SatelliteManager, time_idx, out_path, *,
#                                        ecef_in_km=False, alt_unit='km', links=None):
#     """
#     从 SatelliteManager 在指定时间步导出所有卫星的 (lng, lat, alt) 到 JSON。
#     新增:
#       - links: list[{"source": id, "target": id}]，不传则默认为 []
#     """
#     sats = []
#     scale = 1000.0 if ecef_in_km else 1.0
#
#     for sid, sat in SatelliteManager.satellites.items():
#         if time_idx >= len(sat.trajectory):
#             raise IndexError(f"satellite {sid} has trajectory length {len(sat.trajectory)}, "
#                              f"but time_idx={time_idx}")
#         p = sat.trajectory[time_idx]
#         x, y, z = p.x * scale, p.y * scale, p.z * scale
#         lon, lat, alt_m = _ecef_to_lla(x, y, z)
#         alt_out = alt_m / 1000.0 if alt_unit == 'km' else alt_m
#         sats.append({
#             "id": str(sid),
#             "lng": float(lon),
#             "lat": float(lat),
#             "alt": float(alt_out)
#         })
#
#     payload = { "satellites": sats, "links": (links or []) }
#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     with out_path.open("w", encoding="utf-8") as f:
#         json.dump(payload, f, ensure_ascii=False, indent=2)
#
#     return str(out_path)


def export_highlighted_to_json(group_dict, selected_groups, color_map, out_path):
    """
    根据分组字典和选定区域，导出高亮卫星分组信息到 JSON 文件。

    :param group_dict: dict, 包含所有卫星分组及其卫星 ID 的字典，格式如 {'group_id': {satellite_ids}}
    :param selected_groups: list, 指定的区域（例如 [1, 2]），表示需要导出的分组
    :param color_map: dict, 为每个分组指定颜色，格式如 {group_id: color}
    :param out_path: str, 输出的 JSON 文件路径
    """
    highlighted_data = []

    # 遍历选择的分组
    for group_id in selected_groups:
        # 获取分组对应的卫星 ID
        satellites = list(group_dict['groups'].get(group_id, []))
        # 获取该分组的颜色，若没有颜色，则使用默认颜色
        color = color_map.get(group_id, '#ff5722')  # 默认为橙色

        # 创建分组信息字典
        group_info = {
            "group": f"fenzu{group_id}",  # 使用 "fenzu" + group_id 构建分组名称
            "satellites": satellites,
            "color": color
        }

        highlighted_data.append(group_info)

    # 将生成的高亮数据写入 JSON 文件
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(highlighted_data, f, ensure_ascii=False, indent=2)

    return str(out_path)

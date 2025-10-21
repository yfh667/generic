import basicSa.fileread.readstation as readstation
import basicSa.fileread.readsatellite as readsatellite
import basicSa.simulator.nodemanager as nodemanager

import basicSa.postsimmulation.snapshot as snapshot

import numpy as np
from basicSa.los import  Sat2Gnd
from basicSa.satellite_stk_alpha.core.angularvector2 import SatelliteVector
import basicSa.simulator.neighbor as neighbor
import basicSa.linkalgorith.postallpath as postallpath
import xml.etree.ElementTree as ET
import draw.step1.function.satellite_database as satellite_database
import json, math
from pathlib import Path

# --- 常量（WGS-84） ---
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
#
#
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

def build_same_orbit_links(P, N, SatelliteManager, as_str=True):
    """
    为每个平面 x，给 (x, y) 连接到 (x, (y+1) % N)。
    返回 [{source: id, target: id}, ...]
    - as_str=True：把 id 转成字符串，和你导出的 satellites.id 保持一致
    说明：这里假设 SatelliteManager.satellites 以 0..P*N-1 的整数为 key，
    且 j = x*N + y 能访问到对应卫星（你现有代码就是这么用的）。
    """
    links = []
    for x in range(P):
        for y in range(N):
            curr = x * N + y
            nxt  = x * N + ((y + 1) % N)
            # 防守式判断，确保字典里确有这些编号的卫星
            if curr in SatelliteManager.satellites and nxt in SatelliteManager.satellites:
                s = str(curr) if as_str else curr
                t = str(nxt)  if as_str else nxt
                links.append({"source": s, "target": t})
    return links


def save_to_xml(output_file, station_visible_data):
    """保存每个时间步的地面站可见卫星到XML"""
    root = ET.Element("snapshots")

    for time_step, stations_data in station_visible_data.items():
        time_elem = ET.SubElement(root, "time")
        time_elem.set("step", str(time_step))  # 时间步标记

        stations_elem = ET.SubElement(time_elem, "stations")

        for station_id, sat_list in stations_data.items():
            station_elem = ET.SubElement(stations_elem, "station")
            station_elem.set("id", str(station_id))  # 地面站ID

            for sat_id in sat_list:
                sat_elem = ET.SubElement(station_elem, "basicSa")
                sat_elem.set("id", str(sat_id))  # 卫星ID

    # 让XML缩进（Python 3.9+必备）
    ET.indent(root, space="    ")

    tree = ET.ElementTree(root)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)

import draw.step1.function.linkjson as linkjson

def main():

    # —— 配置 —— #
    station_dir = r'C:\usrspace\mywork\generic\data\sta20'
    station_min_elev_deg = 20
    sat_dir = r'C:\usrspace\mywork\generic\data\648qianfan'
    P, N = 18, 36
    sat_angle = 45
    track_angle = 89
    base_raan_increment = 18
    sim_steps = 101

    # —— 构建数据库 —— #
    db, SatelliteManager, stations_info = satellite_database.build_time_database(
        station_dir, station_min_elev_deg,
        sat_dir, P, N,
        sat_angle=sat_angle,
        track_angle=track_angle,
        base_raan_increment=base_raan_increment,
        sim_steps=sim_steps,
        lenthpropority=20,
        stationsnaplength=20,
        include_stations=True,       # 只看卫星可设为 False
        compute_linkflags=False,     # 需要角速度/链路时再开
    )



 # them here ,we could easily do our job
    # —— 在 main() 最后，XML 写完后追加：导出 2D 可视化用的 JSON —— #
    # —— 导出 t=100 的快照，并附带“同轨链接” —— #
    time_idx = 100
    out_json = r"C:\usrspace\mywork\generic\data\snapshot_100.json"

    # 构建同轨（同平面）邻接： (x,y) ↔ (x,(y+1)%N)
    same_orbit_links = build_same_orbit_links(P, N, SatelliteManager, as_str=True)

    linkjson.export_satellites_snapshot_to_json(
        SatelliteManager,
        time_idx=time_idx,
        out_path=out_json,
        ecef_in_km=False,   # 按你的坐标单位调整
        alt_unit='km',
        links=same_orbit_links
    )

    print(f"Snapshot @ {time_idx} with same-orbit links exported to: {out_json}")




#  save_to_xml("/home/yfh/Desktop/Data/station_visible_satellites_648.xml", station_visible_data)

if __name__ == "__main__":
    main()



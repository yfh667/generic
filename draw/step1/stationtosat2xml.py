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


import xml.etree.ElementTree as ET

def save_satellites_db_to_xml(
    db,
    out_file_path: str,
    *,
    by: str = "satellite",      # 'satellite' 或 'time'
    include_lla: bool = True,   # 是否同时输出 lon/lat/alt（由 x/y/z 转换）
    ecef_in_km: bool = False,   # 若 db 里的 x/y/z 是千米，则设 True
    float_fmt: str = ".6f"      # 数字格式
):
    """
    基于 TimeDatabase(db) 将卫星坐标（x, y, z 和 lon/lat/alt）写入 XML。
    - db.snapshots[t].satellite_dsnapshot 形状: (sat_count, >=6)
      其中 [0:6] = (x, y, z)
    - by='satellite'：<SatellitesData><Satellite id=""><Time .../></Satellite>...
    - by='time'：     <SatellitesData><Time step=""><Satellite id=""/></Time>...
    """
    scale = 1000.0 if ecef_in_km else 1.0
    sim_steps = len(db.snapshots)
    sat_count = db.satellitenum

    def _fmt(v):
        return format(float(v), float_fmt)

    root = ET.Element("SatellitesData")
    root.set("layout", by)

    if by == "satellite":
        # 每颗卫星一块
        for sid in range(sat_count):
            sat_elem = ET.SubElement(root, "Satellite", id=str(sid))
            for t in range(sim_steps):
                snap = db.snapshots[t]
                mat = snap.satellite_dsnapshot
                if sid >= len(mat):
                    continue
                row = mat[sid]
                x, y, z = row[0], row[1], row[2]
                # 若无效（-1），就跳过该条
                if x == -1 or y == -1 or z == -1:
                    continue

                time_elem = ET.SubElement(sat_elem, "Time", step=str(t))
                pos = ET.SubElement(time_elem, "ECEF")
                ET.SubElement(pos, "X").text = _fmt(x)
                ET.SubElement(pos, "Y").text = _fmt(y)
                ET.SubElement(pos, "Z").text = _fmt(z)

                # 可选：写经纬高
                if include_lla:
                    lon, lat, alt_m = _ecef_to_lla(x*scale, y*scale, z*scale)
                    lla = ET.SubElement(time_elem, "LLA")
                    ET.SubElement(lla, "LonDeg").text = _fmt(lon)
                    ET.SubElement(lla, "LatDeg").text = _fmt(lat)
                    ET.SubElement(lla, "AltM").text   = _fmt(alt_m)

    elif by == "time":
        # 每个时间步一块
        for t in range(sim_steps):
            snap = db.snapshots[t]
            mat = snap.satellite_dsnapshot
            time_elem = ET.SubElement(root, "Time", step=str(t))
            for sid in range(min(sat_count, len(mat))):
                row = mat[sid]
                x, y, z = row[0], row[1], row[2]
                if x == -1 or y == -1 or z == -1:
                    continue
                sat_elem = ET.SubElement(time_elem, "Satellite", id=str(sid))
                pos = ET.SubElement(sat_elem, "ECEF")
                ET.SubElement(pos, "X").text = _fmt(x)
                ET.SubElement(pos, "Y").text = _fmt(y)
                ET.SubElement(pos, "Z").text = _fmt(z)

                # 可选：写经纬高
                if include_lla:
                    lon, lat, alt_m = _ecef_to_lla(x*scale, y*scale, z*scale)
                    lla = ET.SubElement(sat_elem, "LLA")
                    ET.SubElement(lla, "LonDeg").text = _fmt(lon)
                    ET.SubElement(lla, "LatDeg").text = _fmt(lat)
                    ET.SubElement(lla, "AltM").text   = _fmt(alt_m)
    else:
        raise ValueError("by 必须是 'satellite' 或 'time'")

    # 格式化缩进并写文件（Py3.9+）
    ET.indent(root, space="  ", level=0)
    tree = ET.ElementTree(root)
    out_path = Path(out_file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)

    print(f"Saved satellites data to: {out_file_path}")

def main():

    dirpath = r'C:\usrspace\mywork\generic\data\sta20'  # raw string for file path
    stationangle = 20
    StationManager = readstation.readstation_path(dirpath, stationangle)
    stations =  StationManager.stations
    _ground_cache = []
    for sim_time_step in range(len(stations)):
         _ground_cache.append(stations[sim_time_step].trajectory[0])
    stationsnum = len(_ground_cache)


    SatelliteManager = nodemanager.SatelliteManager()

    sat_dir_path = r'C:\usrspace\mywork\generic\data\648qianfan'  # raw string for file path
    P = 18
    N = 36
    #sat_dir_path = '/home/yfh/Desktop/Data/onehun_ecef'
    satangle = 45
    track_angle = 89

    BaseRAAN_INCREMENT = 18
    lenthpropority = 20
    stationsnaplength = 20
    simulatationtime = 101
    readsatellite.readsatellite(SatelliteManager,sat_dir_path, satangle, track_angle, P, N, BaseRAAN_INCREMENT)



    print(1)



#  save_to_xml("/home/yfh/Desktop/Data/station_visible_satellites_648.xml", station_visible_data)

if __name__ == "__main__":
    main()



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

def export_satellites_snapshot_to_json(SatelliteManager, time_idx, out_path, *,
                                       ecef_in_km=False, alt_unit='km'):
    """
    从 SatelliteManager 在指定时间步导出所有卫星的 (lng, lat, alt) 到 JSON。
    - ecef_in_km: 若轨迹坐标是千米，则设为 True（会自动换算为米再做转换）
    - alt_unit: 'km' 或 'm'，输出 JSON 中 alt 的单位
    JSON 结构: { "satellites": [{id,lng,lat,alt}], "links": [] }
    """
    sats = []
    scale = 1000.0 if ecef_in_km else 1.0

    for sid, sat in SatelliteManager.satellites.items():
        # 时间边界检查
        if time_idx >= len(sat.trajectory):
            raise IndexError(f"satellite {sid} has trajectory length {len(sat.trajectory)}, "
                             f"but time_idx={time_idx}")

        p = sat.trajectory[time_idx]
        # ECEF → LLA
        x, y, z = p.x * scale, p.y * scale, p.z * scale
        lon, lat, alt_m = _ecef_to_lla(x, y, z)

        alt_out = alt_m / 1000.0 if alt_unit == 'km' else alt_m
        # 生成条目；保持你的前端字段命名
        sats.append({
            "id": str(sid),
            "lng": float(lon),
            "lat": float(lat),
            "alt": float(alt_out)
        })

    payload = { "satellites": sats, "links": [] }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return str(out_path)

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
  #  print("finish")
    _position_cache = {}

    db = snapshot.TimeDatabase()

    db.stationnum = stationsnum
    db.satellitenum = P*N

## here we get the satellite data into the db
    for sim_time_step in range(simulatationtime):
        #basicSa
        _position_cache.clear()
        for sat in SatelliteManager.satellites.values():
            # 使用traj_idx替代内部变量i，避免名称冲突
            idx = next(
                (traj_idx for traj_idx, p in enumerate(sat.trajectory)
                 if p.time >= sim_time_step),  # 正确比较轨迹时间与仿真步长
                0
            )
            # 确保idx+1不超过轨迹长度
            next_idx = min(idx + 1, len(sat.trajectory) - 1)
            _position_cache[sat.id] = (
                sat.trajectory[idx],
                sat.trajectory[next_idx]  # 安全访问下一时刻
            )
        sv = SatelliteVector(P, N)
        sv.load_from_3dview(_position_cache)

        linkflags = sv.calculate_angular_velocity_7_all(sim_time_step)


        # 正确初始化方法：使用np.full直接填充-1
        satellitesnap = np.full(shape=(P * N, lenthpropority), fill_value=-1, dtype=float)  # 浮点型矩阵
        stationsnap = np.full(shape=(stationsnum, stationsnaplength), fill_value=-1, dtype=float)

       # stationsnap = np.full((stationsnum, stationsnaplength), -1)  # 形状 (stationsnum行, stationsnaplength列)，全-1
        for j in range(P*N):
            planid = j // N

            satellite = SatelliteManager.satellites[j]

            satellitesnap[j][0] =satellite.trajectory[sim_time_step].x
            satellitesnap[j][1] = satellite.trajectory[sim_time_step].y
            satellitesnap[j][2] = satellite.trajectory[sim_time_step].z
            satellitesnap[j][3] = satellite.trajectory[sim_time_step].angle
            satellitesnap[j][4] = satellite.trajectory[sim_time_step].trackangle
            satellitesnap[j][5] = satellite.trajectory[sim_time_step].RAAN

        #station:
        for j in range(stationsnum):
            stationsnap[j][0] = _ground_cache[j].x
            stationsnap[j][1] = _ground_cache[j].y
            stationsnap[j][2] = _ground_cache[j].z
            stationsnap[j][3] = _ground_cache[j].angle

            m = 4
            for k in range(P*N):
                staellite = SatelliteManager.satellites[k].trajectory[sim_time_step]
                visibility = Sat2Gnd.Ground_Sat(_ground_cache[j], staellite)
                if visibility>0:
                    stationsnap[j][m] = k
                    m =m+1

        timesnap = snapshot.TimeSnapshot(
            timestamp_str=sim_time_step,
            station_snapshot=stationsnap,
            satellite_dsnapshot=satellitesnap
        )
        db.snapshots.append(timesnap)

        # print(f"time is {sim_time_step}")
  #  print("we finish the raw data")



 # them here ,we could easily do our job
    # —— 在 main() 最后，XML 写完后追加：导出 2D 可视化用的 JSON —— #
    time_idx = 100  # 你要看的“100s”的那个离散时间步（与你的轨迹数组索引一致）
    out_json = r"C:\usrspace\mywork\generic\data\snapshot_100.json"

    # 如果你的 sat.trajectory[*].x/y/z 是“千米”，把 ecef_in_km=True；
    # 如果是“米”，保持 False：
    export_satellites_snapshot_to_json(
        SatelliteManager,
        time_idx=time_idx,
        out_path=out_json,
        ecef_in_km=False,   # ← 根据你的数据单位改 True/False
        alt_unit='km'       # 输出 alt 用 km，和你前端例子一致
    )

    print(f"Snapshot @ {time_idx} exported to: {out_json}")



#  save_to_xml("/home/yfh/Desktop/Data/station_visible_satellites_648.xml", station_visible_data)

if __name__ == "__main__":
    main()



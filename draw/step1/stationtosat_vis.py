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
    simulatationtime = 100
    readsatellite.readsatellite(SatelliteManager,sat_dir_path, satangle, track_angle, P, N, BaseRAAN_INCREMENT)
  #  print("finish")
    _position_cache = {}

    db = snapshot.TimeDatabase()

    db.stationnum = stationsnum
    db.satellitenum = P*N

## here we calculate all the status of the constellation
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
           #  if planid<=P-2:
           #      satellitesnap[j][6] =  linkflags[0][j]
           #      satellitesnap[j][7] =  linkflags[1][j]
           #      satellitesnap[j][8] =  linkflags[2][j]
           #      satellitesnap[j][9] =  linkflags[3][j]
           #  if planid <= P - 3:
           #      satellitesnap[j][10] =  linkflags[4][j]
           #      satellitesnap[j][11] =  linkflags[5][j]
           # # satellitesnap[j][12] = basicSa.trajectory[sim_time_step].linkflags[4][j]


            # if planid>=1:
            #     _,_,ID = neighbor._get_left_neighbor2(j,N,0)
            #     satellitesnap[j][12] = linkflags[3][ID]
            #
            #     _,_,ID = neighbor._get_left_neighbor2(j,N,1)
            #     satellitesnap[j][13] = linkflags[2][ID]
            #
            #     _,_,ID = neighbor._get_left_neighbor2(j,N,2)
            #     satellitesnap[j][14] = linkflags[1][ID]
            #
            #     _,_,ID = neighbor._get_left_neighbor2(j,N,3)
            #     satellitesnap[j][15] = linkflags[0][ID]

            #
            # if planid>=2:
            #     _, _, ID = neighbor._get_left_neighbor2(j, N, 4)
            #     satellitesnap[j][16] = linkflags[5][ID]
            #
            #     _, _, ID = neighbor._get_left_neighbor2(j, N, 5)
            #     satellitesnap[j][17] = linkflags[4][ID]

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




    station_visible_data = {}  # 存储所有时间步数据

    # 主代码：计算卫星连接数

    for i in range(simulatationtime):  # 假设是2个snapshot
        snap = db.snapshots[i]
        stationsnap = snap.stations
        sats_access_ground = [0] * P * N

        current_step_data = {}  # 当前时间步的地面站数据

        for k in range(stationsnum):
            visible_sats = [sat_id for sat_id in stationsnap[k][4:] if sat_id != -1]
            current_step_data[k] = visible_sats  # 记录地面站-卫星关系

        station_visible_data[i] = current_step_data  # 存储此时间步




    # 输出到XML

    save_to_xml(r"C:\usrspace\mywork\generic\data\test100.xml", station_visible_data)


#  save_to_xml("/home/yfh/Desktop/Data/station_visible_satellites_648.xml", station_visible_data)

if __name__ == "__main__":
    main()



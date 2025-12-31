import basicSa.fileread.readstation as readstation
import basicSa.fileread.readsatellite as readsatellite
import basicSa.simulator.nodemanager as nodemanager

import basicSa.postsimmulation.snapshot as snapshot
import bisect
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
                sat_elem = ET.SubElement(station_elem, "satellite")
                sat_elem.set("id", str(sat_id))  # 卫星ID

    # 让XML缩进（Python 3.9+必备）
    ET.indent(root, space="    ")

    tree = ET.ElementTree(root)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)


def main():
   # dirpath = r'C:\usrspace\mywork\data\sta20'  # raw string for file path
    dirpath = r'C:\usrspace\mywork\data_paper2\stations'  # raw string for file path




    stationangle = 20
    StationManager = readstation.readstation_path(dirpath, stationangle)
    stations =  StationManager.stations
    _ground_cache = []
    for sim_time_step in range(len(stations)):
         _ground_cache.append(stations[sim_time_step].trajectory[0])
    stationsnum = len(_ground_cache)


    SatelliteManager = nodemanager.SatelliteManager()

   # sat_dir_path = r'C:\usrspace\mywork\generic\data\648qianfan1d'  # raw string for file path
  #  sat_dir_path = r'C:\usrspace\mywork\data\648qianfan1d_xml'  # raw string for file path
    sat_dir_path = r"C:\usrspace\mywork\data_paper2\position_modify\baseRaan_0_xml"  # raw string for file path


    # sat_dir_path = r"C:\usrspace\mywork\data_paper2\position_modify\g60_xml"  # raw string for file path

    P = 18
    N = 36
    #sat_dir_path = '/home/yfh/Desktop/Data/onehun_ecef'
    satangle = 45
    track_angle = 89

    BaseRAAN_INCREMENT = 18
    lenthpropority = 20
    stationsnaplength = 20
    simulatationtime = 86400


    # simulatationtime = 5000




    readsatellite.readsatellite(SatelliteManager,sat_dir_path, satangle, track_angle, P, N, BaseRAAN_INCREMENT,t_start=0, t_end=simulatationtime)

    print("finish")
    _position_cache = {}

    db = snapshot.TimeDatabase()

    db.stationnum = stationsnum
    db.satellitenum = P*N

## here we calculate all the status of the constellation
    for sim_time_step in range(simulatationtime):
        print(f"time is {sim_time_step}")
        #basicSa
        _position_cache.clear()

        for sat in SatelliteManager.satellites.values():
            # 1) 取已排序的时间轴（可做一次缓存，避免每次都排序）
            if not hasattr(sat, "_sorted_times"):
                sat._sorted_times = sorted(sat.trajectory.keys())
            times = sat._sorted_times

            if not times:
                continue  # 这颗卫星没有轨迹

            # 2) 在时间轴上定位 sim_time_step 的插入位（左闭右开）
            i = bisect.bisect_left(times, sim_time_step)

            # 3) 夹取当前与下一时刻的索引，注意边界
            if i >= len(times):
                # 要的时间在最后一个采样点之后：退回到最后一个点，next 用自己
                idx = len(times) - 1
                next_idx = idx
            else:
                idx = i
                # 如果恰好命中某时刻，就用该点作为 idx；下一点尽量取 idx+1，若越界就等于 idx
                next_idx = min(idx + 1, len(times) - 1)

            t0 = times[idx]
            t1 = times[next_idx]
            node0 = sat.trajectory[t0]  # Node
            node1 = sat.trajectory[t1]  # Node

            # 4) 用 sat.sat_id 作为 key
            _position_cache[sat.sat_id] = (node0, node1)

        sv = SatelliteVector(P, N)
        sv.load_from_3dview(_position_cache)



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

    #save_to_xml(r"C:\usrspace\mywork\generic\data\station_visible_satellites_648_1d_test.xml", station_visible_data)
    save_to_xml(r"C:\usrspace\mywork\data_paper2\visibile_data\baseRaan_0\station_visible_satellites_baseRaan_0.xml", station_visible_data)


    # save_to_xml(r"C:\usrspace\mywork\data_paper2\visibile_data\G60\g60.xml",
    #             station_visible_data)

#  save_to_xml("/home/yfh/Desktop/Data/station_visible_satellites_648.xml", station_visible_data)

if __name__ == "__main__":
    main()



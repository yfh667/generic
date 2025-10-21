import basicSa.fileread.readstation as readstation
import basicSa.fileread.readsatellite as readsatellite
import basicSa.simulator.nodemanager as nodemanager
import basicSa.simulator.linkengine as linkengine
import basicSa.simulator.websocket as websocket
import time
import logging
import basicSa.simulator.sendtocesium as sendtocesium2
import basicSa.simulator.cesiumswitch.heightlight as heightlight
import xml.etree.ElementTree as ET
from xml.dom import minidom
import basicSa.postsimmulation.snapshot as snapshot
from basicSa.satellite_stk_alpha.contrib.route import SatelliteRouter

import basicSa.getthepath.checkconflic as  checkconflic
import basicSa.getthepath.timegraph as  Timegraph

import collections
import heapq
from math import inf
from collections import defaultdict
from collections import deque
import basicSa.getthepath.tiemsegment as tiemsegment
import basicSa.getthepath.buildTimegraph as buildTimegraph
import basicSa.getthepath.TImegraph_path as TImegraph_path
import collections
import numpy as np
from basicSa.los import  Sat2Gnd
from basicSa.satellite_stk_alpha.core.angularvector2 import SatelliteVector
import basicSa.simulator.neighbor as neighbor
import basicSa.linkalgorith.postallpath as postallpath
import xml.etree.ElementTree as ET


def save_to_xml(output_file, data):
    """Save basicSa access data to a structured XML file"""
    root = ET.Element("snapshots")  # 根节点

    for time, entries in data.items():
        time_elem = ET.SubElement(root, "time")
        time_elem.set("step", str(time))  # time step = i

        for sat, count in entries.items():
            sat_elem = ET.SubElement(time_elem, "sat")  # sat节点
            sat_elem.set("satid", str(sat))  # 卫星编号 (属性)
            sat_elem.set("connections", str(count))  # 连接数 (属性)

    # 让XML更易读（自动格式化 + 换行）
    ET.indent(root, space="    ")  # Python 3.9+

    tree = ET.ElementTree(root)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)





def main():
    dirpath  = '/home/yfh/Desktop/Data/sta20'
    stationangle = 20
    StationManager = readstation.readstation_path(dirpath, stationangle)
    stations =  StationManager.stations
    _ground_cache = []
    for sim_time_step in range(len(stations)):
         _ground_cache.append(stations[sim_time_step].trajectory[0])
    stationsnum = len(_ground_cache)


    SatelliteManager = nodemanager.SatelliteManager()

    sat_dir_path = '/home/yfh/Desktop/Data/648qianfan'
    satangle = 45
    track_angle = 89
    P = 18
    N = 36
    BaseRAAN_INCREMENT = 10.8
    lenthpropority = 20
    stationsnaplength = 20
    simulatationtime = 10
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
            if planid<=P-2:
                satellitesnap[j][6] =  linkflags[0][j]
                satellitesnap[j][7] =  linkflags[1][j]
                satellitesnap[j][8] =  linkflags[2][j]
                satellitesnap[j][9] =  linkflags[3][j]
            if planid <= P - 3:
                satellitesnap[j][10] =  linkflags[4][j]
                satellitesnap[j][11] =  linkflags[5][j]
           # satellitesnap[j][12] = basicSa.trajectory[sim_time_step].linkflags[4][j]


            if planid>=1:
                _,_,ID = neighbor._get_left_neighbor2(j,N,0)
                satellitesnap[j][12] = linkflags[3][ID]

                _,_,ID = neighbor._get_left_neighbor2(j,N,1)
                satellitesnap[j][13] = linkflags[2][ID]

                _,_,ID = neighbor._get_left_neighbor2(j,N,2)
                satellitesnap[j][14] = linkflags[1][ID]

                _,_,ID = neighbor._get_left_neighbor2(j,N,3)
                satellitesnap[j][15] = linkflags[0][ID]


            if planid>=2:
                _, _, ID = neighbor._get_left_neighbor2(j, N, 4)
                satellitesnap[j][16] = linkflags[5][ID]

                _, _, ID = neighbor._get_left_neighbor2(j, N, 5)
                satellitesnap[j][17] = linkflags[4][ID]

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

        print(f"time is {sim_time_step}")
  #  print("we finish the raw data")


    ## here we calculate all the raw data of the satellites

    #sats_access_ground = []
    #
    # for i in range(2):
    #     snap = db.snapshots[i]
    #     stationsnap = snap.stations
    #     sats_access_ground = [0] * P * N
    #     for j in range(len(stationsnap)):
    #         for k in range(4, len(stationsnap[j])):
    #             if stationsnap[j][k] != -1:
    #                 sats_access_ground[int(stationsnap[j][k])] += 1  # Convert to int
    #     print(f"time is {i}")
    #     for j in range(P * N):
    #         if sats_access_ground[j]!=0:
    #             print(f"sat {j} is {sats_access_ground[j]}")

    # 主代码：计算卫星连接数
    result_data = {}  # {time_step: {sat_id: connections}}

    for i in range(simulatationtime):  # 假设是2个snapshot
        snap = db.snapshots[i]
        stationsnap = snap.stations
        sats_access_ground = [0] * P * N

        # 统计卫星-地面站连接数
        for stn in stationsnap:
            for sat_idx in stn[4:]:  # 跳过前4个无用字段？
                if sat_idx != -1:
                    sats_access_ground[int(sat_idx)] += 1

        # 只记录 connections > 0 的卫星, key=sat_id, value=connections
        result_data[i] = {
            sat_id: count
            for sat_id, count in enumerate(sats_access_ground)
            if count > 0
        }

    # 输出到XML
    save_to_xml("/home/yfh/Desktop/Data/sat_access_stats.xml", result_data)



if __name__ == "__main__":
    main()

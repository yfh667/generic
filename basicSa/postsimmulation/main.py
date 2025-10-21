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

def main():
    dirpath  = '/home/yfh/Desktop/Data/station_qianfan_tongjin'
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
    simulatationtime = 1000
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




    allpaths = []
    for i in range(simulatationtime):
        snap = db.snapshots[i]
        paths = postallpath.getpaths(snap, N, P, stationsnum)

        allpaths.append(paths)

    from collections import defaultdict

    # 用于存储每个 path 的 intervals

    def cluster_path_intervals(allpaths):
        # 存储每个路径及其时间区间
        path_info = defaultdict(list)

        # 首先记录每个路径出现的所有时间点
        for time_step, paths in enumerate(allpaths):
            for path in paths:
                path_tuple = tuple(path)  # 转为可哈希的类型
                path_info[path_tuple].append(time_step)

        # 对每个路径的时间点进行区间合并
        result = []
        for path_tuple, times in path_info.items():
            times = sorted(times)
            intervals = []
            if not times:
                continue

            # 合并连续的时间点为区间
            start = times[0]
            prev = times[0]
            for current in times[1:]:
                if current == prev + 1:  # 连续
                    prev = current
                else:  # 不连续了，记录当前区间
                    intervals.append([start, prev])
                    start = current
                    prev = current
            intervals.append([start, prev])  # 添加最后一个区间

            result.append({
                "path": list(path_tuple),
                "time_intervals": intervals
            })

        return result

    # 测试

    # 使用方法
    paths = cluster_path_intervals(allpaths)
    chenfa = -60
    time_segments = tiemsegment.calculate_time_segments(paths)
    conflict_matrix = checkconflic.build_conflict_matrix(paths)
    graph = buildTimegraph.build_Time_graph(paths, time_segments,chenfa,conflict_matrix)
    print("图构建完成，使用数字索引路径:")
    print("节点ID格式: (时间段索引, 路径索引)")


  #  interval, total_weight, dijkstra_path = TImegraph_path.find_shortest_path(graph, (0, 1), 4, conflict_matrix,
 #                                                                           #  time_segments)
  #  print(interval)
    interval, total_weight, dijkstra_path = TImegraph_path.find_shortest_paths(graph, conflict_matrix, time_segments,paths)


    # for baseinterval in interval:
    #     print(baseinterval)
    # print(f"interval: {interval}")
    # print(f"total_weight: {total_weight}")
    # print(f"dijkstra_path: {dijkstra_path}")

    # for path_idx, intervals in interval.items():
    #     print(f"Path {paths[path_idx]['path']} -> Time Intervals: {intervals}")






if __name__ == "__main__":
    main()

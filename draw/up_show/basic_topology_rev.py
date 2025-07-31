from PyQt5.QtGui import QPainterPath
from PyQt5.QtWidgets import QGraphicsPathItem

import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import draw.read_snap_xml as read_snap_xml
import draw.basic_show.get_satellite_block_info as get_satellite_block_info
import draw.pyqt_draw.pyqt_main as pyqt_main
N = 36   # 每轨道卫星数
P = 18   # 轨道平面数
TOTAL_SATS = N * P

STATION_GROUPS = {
    0: {"name": "Group 0", "stations": list(range(0, 4))},
    1: {"name": "Group 1", "stations": list(range(4, 9))},
    2: {"name": "Group 2", "stations": [9]},
    3: {"name": "Group 3", "stations": [10]},
    4: {"name": "Group 4", "stations": list(range(11, 15))},
    5: {"name": "Group 5", "stations": list(range(15, 17))},
    6: {"name": "Group 6", "stations": list(range(17, 20))},
}

GROUP_COLORS = [
    '#FF0000',  # 红 (Group 0)
    '#00FF00',  # 绿 (Group 1)
    '#0000FF',  # 蓝 (Group 2)
    '#FFA500',  # 橙 (Group 3)
    '#800080',  # 紫 (Group 4)
    '#00FFFF',  # 青 (Group 5)
    '#FFFF00',  # 黄   (Group 6)
]



start_ts = 1215
# end_ts = 3540


# start_ts = 3541
# end_ts = 3668

# start_ts = 3669
# end_ts = 5950

# start_ts = 5951
end_ts = 6218
#




#xml_file = "E:\\Data\\station_visible_satellites_648_8_h.xml"
xml_file = "E:\\Data\\station_visible_satellites_648_1d_real.xml"
group_data = read_snap_xml.parse_xml_group_data(xml_file, start_ts, end_ts)
group_data,offset = read_snap_xml.modify_group_data(group_data, N=36, groupid=0)

def calc_envelope_for_group(group_data, times, groupid, N):
    x_min, x_max, y_min, y_max = -1, -1, -1, -1
    for t in times:
        sats = group_data[t]['groups'][groupid]
        xs = [sid // N for sid in sats]
        ys = [sid % N for sid in sats]
        if not xs or not ys:
            continue
        if x_min == -1 or x_min > min(xs):
            x_min = min(xs)
        if x_max == -1 or x_max < max(xs):
            x_max = max(xs)
        if y_min == -1 or y_min > min(ys):
            y_min = min(ys)
        if y_max == -1 or y_max < max(ys):
            y_max = max(ys)
    return x_min, x_max, y_min, y_max



if __name__ == "__main__":
    pg.setConfigOption('background', 'w')

    edges_by_step = {}


    for step in range(start_ts, end_ts):
        edges_by_step[step] = {}
        for i in range(P - 1):
            for j in range(14,32):
                nownode = i * N + j

                # if i<P-2:
                #     next_node1 = (i + 2) * N + j
                #     edges_by_step[step].setdefault(nownode, set()).add(next_node1)


                if i+1<=17 and j+2<=32:
                    next_node1 = (i + 1) * N + j+2
                    edges_by_step[step].setdefault(nownode, set()).add(next_node1)



                upnodes = i * N + (j + 1) % N
                edges_by_step[step].setdefault(nownode, set()).add(upnodes)
                downnodes = i * N + (j - 1 + N) % N
                edges_by_step[step].setdefault(nownode, set()).add(downnodes)


    for step in range(start_ts, end_ts):

        for i in range(P - 1):
            for j in range(0,14):
                nownode = i * N + j


                next_node1 = (i + 1) * N + j
                edges_by_step[step].setdefault(nownode, set()).add(next_node1)


    # times = sorted(group_data.keys())
    # groupid = 4
    # # --- 主程序修改 ---
    #
    # x_min, x_max, y_min, y_max = -1, -1, -1, -1
    # for t in times:
    #
    #     sats4 = group_data[t]['groups'][groupid]
    #     xs = [sid // N for sid in sats4]
    #     ys = [sid % N for sid in sats4]
    #     if x_min==-1:
    #         x_min = min(xs)
    #     if x_max==-1:
    #         x_max = max(xs)
    #     if y_min==-1:
    #         y_min = min(ys)
    #     if y_max==-1:
    #         y_max = max(ys)
    #
    #     if x_min>min(xs):
    #         x_min = min(xs)
    #     if x_max<max(xs):
    #         x_max = max(xs)
    #     if y_min>min(ys):
    #         y_min = min(ys)
    #     if y_max<max(ys):
    #         y_max = max(ys)
    #
    #
    #
    # print(x_min, x_max, y_min, y_max)
    #
    # groupid = 0
    # # --- 主程序修改 ---
    #
    # x_min1, x_max1, y_min1, y_max1 = -1, -1, -1, -1
    # for t in times:
    #
    #     sats4 = group_data[t]['groups'][groupid]
    #     xs = [sid // N for sid in sats4]
    #     ys = [sid % N for sid in sats4]
    #     if x_min1 == -1:
    #         x_min1 = min(xs)
    #     if x_max1 == -1:
    #         x_max1 = max(xs)
    #     if y_min1 == -1:
    #         y_min1 = min(ys)
    #     if y_max1 == -1:
    #         y_max1 = max(ys)
    #
    #     if x_min1 > min(xs):
    #         x_min1 = min(xs)
    #     if x_max1 < max(xs):
    #         x_max1 = max(xs)
    #     if y_min1 > min(ys):
    #         y_min1 = min(ys)
    #     if y_max1 < max(ys):
    #         y_max1= max(ys)
    intervals = [
        (1215, 3540),
        (3541, 3668),
        (3669, 5950),
        (5951, 6218)
    ]
    envelope_regions = {}

    for start_ts, end_ts in intervals:
        times = [t for t in group_data if start_ts <= t <= end_ts]
        subdict = {}
        for gid in [4, 0]:  # 如需支持更多group可自行添加
            x_min, x_max, y_min, y_max = calc_envelope_for_group(group_data, times, gid, N)
            subdict[gid] = (x_min, x_max, y_min, y_max)
        envelope_regions[(start_ts, end_ts)] = subdict

    # edges_by_step = {
    #     1202: {
    #         1: {73},  # 1连到73
    #
    #     }
    # }

    app = QtWidgets.QApplication([])
    viewer = pyqt_main.SatelliteViewer(group_data)
    # viewer.edges_by_step = edges_by_step
    viewer.envelopesflag  =1
    viewer.envelope_regions = envelope_regions

    # viewer.envelope_regions = {
    #     (start_ts,
    #     end_ts,):{      4: (x_min, x_max, y_min, y_max),
    #    0: (x_min1, x_max1, y_min1, y_max1),}
    #
    #
    # }
    viewer.setWindowTitle("Grouped Satellite Visibility - High Performance (PyQtGraph)")
    viewer.resize(1200, 700)
    viewer.show()
    sys.exit(app.exec_())

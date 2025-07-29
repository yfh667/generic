from PyQt5.QtGui import QPainterPath
from PyQt5.QtWidgets import QGraphicsPathItem

import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import draw.read_snap_xml as read_snap_xml

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

def preprocess_envelopes(group_data, groups_to_envelope, N, y_preset_map):
    envelope_regions = {gid: {} for gid in groups_to_envelope}
    steps = sorted(group_data.keys())
    old_x_min_map = {gid: -1 for gid in groups_to_envelope}
    for step in steps:
        for gid in groups_to_envelope:
            group_sats = group_data[step]["groups"].get(gid, set())
            if gid==4 or gid==0:
                x_coords = [sid // N for sid in group_sats]
                if len(x_coords) == 0:
                    envelope_regions[gid][step] = None
                    continue
                x_min, x_max = min(x_coords), max(x_coords)
                rect_x = x_min - 0.5
                rect_width = 7
                if old_x_min_map[gid] == -1:
                    old_x_min_map[gid] = x_min
                elif old_x_min_map[gid] + rect_width > x_max:
                    rect_x = old_x_min_map[gid] - 0.5
                else:
                    old_x_min_map[gid] = x_min
                rect_y, rect_height = y_preset_map.get(gid, (0, N))
                envelope_regions[gid][step] = (rect_x, rect_y, rect_width, rect_height)
            else:
                envelope_regions[gid][step] = None
    return envelope_regions

# start_ts = 1202
# end_ts = 3320

# start_ts = 53232
# end_ts = 55574

start_ts = 1215
end_ts = 3540
#xml_file = "E:\\Data\\station_visible_satellites_648_8_h.xml"
xml_file = "E:\\Data\\station_visible_satellites_648_1d_real.xml"
group_data = read_snap_xml.parse_xml_group_data(xml_file, start_ts, end_ts)
group_data,offset = read_snap_xml.modify_group_data(group_data, N=36, groupid=0)


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
    viewer.setWindowTitle("Grouped Satellite Visibility - High Performance (PyQtGraph)")
    viewer.resize(1200, 700)
    viewer.show()
    sys.exit(app.exec_())

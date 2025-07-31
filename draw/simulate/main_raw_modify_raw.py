from PyQt5.QtGui import QPainterPath
from PyQt5.QtWidgets import QGraphicsPathItem
import genaric2.tegnode as tegnode
import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import draw.read_snap_xml as read_snap_xml
import draw.pyqt_draw.adjacent2xml as adjacent2xml
import draw.pyqt_draw.pyqt_main as pyqt_main
N = 36   # 每轨道卫星数
P = 18   # 轨道平面数
TOTAL_SATS = N * P

start_ts = 1202
end_ts = 3320


# xml_file = "E:\\Data\\station_visible_satellites_648_8_h.xml"
xml_file = "E:\\Data\\station_visible_satellites_648_1d_real.xml"
group_data = read_snap_xml.parse_xml_group_data(xml_file, start_ts, end_ts)
modify_group_data,off_sets = read_snap_xml.modify_group_data(group_data, N=36, groupid=4)


if __name__ == "__main__":
    pg.setConfigOption('background', 'w')
    time_2_build = 60
    edges_by_step = {}



#### here we hard code our modify edges, this conclude the all the link for the transformed the chart
# first we need arragent the main inter-link region
    for step in range(start_ts, end_ts):
        edges_by_step[step] = {}
        for i in range(P - 1):
            for j in range(14,32):
                nownode = i * N + j
                if i+1<=17 and j+2<32:
                    next_node1 = (i + 1) * N + j+2
                    edges_by_step[step].setdefault(nownode, set()).add(next_node1)

                upnodes = i * N + (j + 1) % N
                edges_by_step[step].setdefault(nownode, set()).add(upnodes)
                downnodes = i * N + (j - 1 + N) % N
                edges_by_step[step].setdefault(nownode, set()).add(downnodes)

# second we need managet the region 0
    for step in range(start_ts, end_ts):

        for i in range(P - 1):
            for j in range(0,14):
                nownode = i * N + j


                upnodes = i * N + (j + 1) % N
                edges_by_step[step].setdefault(nownode, set()).add(upnodes)
                downnodes = i * N + (j - 1 + N) % N
                edges_by_step[step].setdefault(nownode, set()).add(downnodes)

                next_node1 = (i + 1) * N + j
                edges_by_step[step].setdefault(nownode, set()).add(next_node1)

# third we need managet the region 4

    for step in range(start_ts, end_ts):

        for i in range(P - 1):
            for j in range(32, 36):
                nownode = i * N + j

                upnodes = i * N + (j + 1) % N
                edges_by_step[step].setdefault(nownode, set()).add(upnodes)
                downnodes = i * N + (j - 1 + N) % N
                edges_by_step[step].setdefault(nownode, set()).add(downnodes)

                next_node1 = (i + 1) * N + j
                edges_by_step[step].setdefault(nownode, set()).add(next_node1)


    # raw_edges_by_step = {}
    # for step, edges in edges_by_step.items():  # step 是时间片
    #     for src, dsts in edges.items():  # src 是起点
    #         for dst in dsts:  # dst 是终点集合里的每一个
    #             print(step, src, dst)
                # 在这里做你要做的事，比如绘制、统计等

    # edges_by_step = {
    #     1202: {
    #         1: {73},  # 1连到73
    #
    #     }
    # }


## then we need transform the modify edges to the raw edges
    raw_edges_by_step = {}

    for step, edges in edges_by_step.items():  # step: 时间片
        raw_edges_by_step[step] = {}

        for src, dsts in edges.items():  # src: 起点id, dsts: 终点集合
            raw_src = read_snap_xml.rev_modify_data(step, src, off_sets)

            for dst in dsts:
                raw_dst = read_snap_xml.rev_modify_data(step, dst, off_sets)
                raw_edges_by_step[step].setdefault(raw_src, set()).add(raw_dst)

    # xml_file = "E:\\Data\\test_raw.xml"
    #
    # adjacent2xml.write_steps_to_xml(raw_edges_by_step, xml_file)
    # 我们接下来需要导出modify_edges_by_step，因为这个是转化后的拓扑形态
    app = QtWidgets.QApplication([])
    viewer = pyqt_main.SatelliteViewer(group_data)
    viewer.edges_by_step = raw_edges_by_step
    # viewer.pending_links_by_step = pending_links_by_step  # <<<<<<<< 新增
    # viewer.envelopesflag = 1
    viewer.setWindowTitle("Grouped Satellite Visibility - High Performance (PyQtGraph)")
    viewer.resize(1200, 700)
    viewer.show()
    sys.exit(app.exec_())
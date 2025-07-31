
from PyQt5.QtGui import QPainterPath
from PyQt5.QtWidgets import QGraphicsPathItem
import genaric2.tegnode as tegnode
import sys
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import draw.read_snap_xml as read_snap_xml
import draw.pyqt_draw.adjacent2xml as adjacent2xml
import draw.pyqt_draw.pyqt_main as pyqt_main
import draw.pyqt_draw.adjacent2xml as adjacent2xml
if __name__ == "__main__":
    start_ts = 1202
    end_ts = 3320
    N = 36
    groupid = 4


    xml_file = "E:\\Data\\test.xml"
    #xml_file = "E:\\Data\\grid.xml"

    modify_edges_by_step = adjacent2xml.read_steps_from_xml(xml_file)

    xml_file2 = "E:\\Data\\station_visible_satellites_648_8_h.xml"

    group_data = read_snap_xml.parse_xml_group_data(xml_file2, start_ts, end_ts)
    group_data, offset = read_snap_xml.modify_group_data(group_data, N=36, groupid=4)





    # 我们接下来需要导出modify_edges_by_step，因为这个是转化后的拓扑形态
    app = QtWidgets.QApplication([])
    viewer = pyqt_main.SatelliteViewer(group_data)
    viewer.edges_by_step = modify_edges_by_step
   # viewer.pending_links_by_step = modify_pending_links_by_step  # <<<<<<<< 新增
   #  viewer.envelopesflag = 1
    viewer.setWindowTitle("Grouped Satellite Visibility - High Performance (PyQtGraph)")
    viewer.resize(1200, 700)
    viewer.show()
    sys.exit(app.exec_())

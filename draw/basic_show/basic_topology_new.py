# 1) 必须已做过：%gui qt5

# 2) 只导入一次，避免重复导入 Qt 类与 sip 触发崩溃
import sys
if 'draw.pyqt_draw.pyqt_main2' in sys.modules:
    del sys.modules['draw.pyqt_draw.pyqt_main2']  # 防止你反复粘贴时热重载 Qt 类引起崩溃
from PyQt5 import QtWidgets
import pyqtgraph as pg

from draw.pyqt_draw.pyqt_main2 import SatelliteViewer
import draw.read_snap_xml as read_snap_xml

pg.setConfigOptions(antialias=True)          # 保险：不开 OpenGL
# pg.setConfigOptions(useOpenGL=False)       # 若你机器上有 OpenGL/驱动问题，可显式关掉

xml_file = r"E:\Data\station_visible_satellites_648_1d_real.xml"
start_ts, end_ts = 0, 3320

gd = read_snap_xml.parse_xml_group_data(xml_file, start_ts, end_ts)
rev_group_data,offset = read_snap_xml.modify_group_data(gd, N=36, groupid=0)

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

# 关键：给 viewer 找个全局引用，防止被 GC 回收引发 native 崩溃
if not hasattr(sys.modules[__name__], "_viewer_list"):
    _viewer_list = []
viewer = SatelliteViewer(gd)
viewer.setWindowTitle("Grouped Satellite Visibility - High Performance (PyQtGraph)")
viewer.resize(1200, 700)

#     viewer.edges_by_step = raw_edges_by_step
viewer.show()
_viewer_list.append(viewer)   # 保持引用

# 注意：在控制台里不要 app.exec_()，%gui qt5 已经驱动事件循环

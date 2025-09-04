from PyQt5 import QtWidgets
import pyqtgraph as pg
from draw.pyqt_draw.pyqt_main2 import SatelliteViewer
import draw.read_snap_xml as read_snap_xml
import draw.read_snap_xml as read_snap_xml
# 配置 pyqtgraph：开启抗锯齿，关闭 OpenGL（更稳定）
pg.setConfigOptions(antialias=True)

from config import DATA_DIR

# ====================== 基础参数 ======================
# 星座参数：每轨道卫星数 N，轨道平面数 P
N = 36
P = 18

# ====================== 读取数据 ======================
file_in = DATA_DIR
# xml_file = r"DATA_DIR\station_visible_satellites_648_1d_real.xml"
xml_file = DATA_DIR / "station_visible_satellites_648_1d_real.xml"


start_ts = 4094

# end_ts   = 86399
end_ts   = 6814
# 解析 XML 得到 group_data，结构：{time_step: {'groups': {...}}}
group_data = read_snap_xml.parse_xml_group_data(xml_file, start_ts, end_ts)

#下面是图变换的。
#
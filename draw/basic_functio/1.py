# ====================== 基础参数 ======================
# 星座参数：每轨道卫星数 N，轨道平面数 P

from PyQt5 import QtWidgets
import pyqtgraph as pg
from draw.pyqt_draw.pyqt_main2 import SatelliteViewer
import draw.read_snap_xml as read_snap_xml
import draw.basic_functio.revdata2rawdata as revdata2rawdata
N = 36
P = 18
# ====================== 读取数据 ======================
xml_file = r"E:\Data\station_visible_satellites_648_1d_real.xml"
start_ts = 1
# end_ts   = 86399
end_ts   = 1202
# 解析 XML 得到 group_data，结构：{time_step: {'groups': {...}}}
group_data = read_snap_xml.parse_xml_group_data(xml_file, start_ts, end_ts)
rev_group_data,offset = read_snap_xml.modify_group_data(group_data, N=36, groupid=4)

nodes = {}
import draw.basic_functio.motif as motif

motif.write_distinct_motif(0, 17, 0, 9, P, N, nodes, option=0)
adj = motif.transform_nodes_2_adjacent(nodes, P, N)
alledge = {}
for i in range(start_ts, end_ts):
    alledge[i] = adj
revdata2rawdata.revedge2rawedge(alledge,offset)
print(1)

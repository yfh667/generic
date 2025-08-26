# ====================== 基础参数 ======================
# 星座参数：每轨道卫星数 N，轨道平面数 P
import sys

from PyQt5 import QtWidgets
import pyqtgraph as pg
from draw.pyqt_draw.pyqt_main2 import SatelliteViewer
import draw.read_snap_xml as read_snap_xml
import draw.basic_functio.revdata2rawdata as revdata2rawdata
N = 36
P = 18
# ====================== 读取数据 ======================
xml_file = r"E:\Data\station_visible_satellites_648_1d_real.xml"
# ====================== 基础参数 ======================
# 星座参数：每轨道卫星数 N，轨道平面数 P
N = 36
P = 18

# ====================== 读取数据 ======================
xml_file = r"E:\Data\station_visible_satellites_648_1d_real.xml"
start_ts = 0
# end_ts   = 86399
end_ts   = 1024
# 解析 XML 得到 group_data，结构：{time_step: {'groups': {...}}}
group_data = read_snap_xml.parse_xml_group_data(xml_file, start_ts, end_ts)
rev_group_data,offset = read_snap_xml.modify_group_data(group_data, N=36, groupid=4)
# 同样是计算block尺寸
import draw.basic_functio.get_rectangular_size_interval as get_rectangular_size_interval
t1,t2=get_rectangular_size_interval.calc_envelope_for_group(rev_group_data,[start_ts,end_ts],0,P,N)
t3,t4=get_rectangular_size_interval.calc_envelope_for_group(rev_group_data,[start_ts,end_ts],4,P,N)

# 这里一般就是规划的地方了，这里进行手动规划，
nodes = {}

import draw.basic_functio.motif as motif
# 下面就是motif图案填充，重复图案填充。里面有多种选项。
# 1
# motif.write_distinct_motif(0, 17, 9, 35, P, N, nodes, option=0)
# motif.write_distinct_motif(0, 17, 0, 8, P, N, nodes, option=1)
motif.write_distinct_motif(0, 17, 9, 35, P, N, nodes, option=0)
motif.write_distinct_motif(0, 17, 0, 8, P, N, nodes, option=1)

rev_inter_edge = motif.transform_nodes_2_adjacent(nodes,P,N)

# 将同构图的拓扑设计映射到该时间段的同构拓扑设计上去，
all_rev_inter_edge = {}
for i in range(start_ts,end_ts):
    all_rev_inter_edge[i] = rev_inter_edge
# here the adj can be used for the qt5 to draw

# 注意上述我们是在同构拓扑序列上进行的，因此，我们还要将同构拓扑序列进行转化，同时，我们还要考虑到建链时间约束
import draw.basic_functio.revdata2rawdata as revdata2rawdata
# attention ,here  it just composed of the inter-link, intra_link hasn't benn conclued
raw_inter_edge = revdata2rawdata.revedge2rawedge(all_rev_inter_edge,offset)
import draw.basic_functio.conflict_link as conflict_link
# 建链时间约束
time_2_build = 60
import draw.basic_functio.get_rectangular_size_interval as get_rectangular_size_interval
t1,t2=get_rectangular_size_interval.calc_envelope_for_group(rev_group_data,[start_ts,end_ts],0,P,N)
t3,t4=get_rectangular_size_interval.calc_envelope_for_group(rev_group_data,[start_ts,end_ts],4,P,N)
rects = {
    0: (t1, t2),
    4: (t3, t4),
}

# 这里一般就是规划的地方了，这里进行手动规划，
nodes = {}


# ====================== 绘图初始化 ======================
# 1) QApplication 实例（全局唯一）
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

# 2) 确保 viewer 有全局引用，避免 GC 回收导致崩溃
if not hasattr(sys.modules[__name__], "_viewer_list"):
    _viewer_list = []
# 3) 下面是同构图设计，我们一般从同构图上设计出 motif，然后，再迁移到我们其他图的显示上去
# 3) 创建并配置 viewer
viewer = SatelliteViewer(group_data)
viewer.setWindowTitle("group_data")
viewer.resize(1200, 700)
viewer.edges_by_step =raw_inter_edge
# viewer.pending_links_by_step =
viewer.show()

_viewer_list.append(viewer)
# raw_edges_by_step = conflict_link.get_no_conflict_link(raw_inter_edge,offset,rects,start_ts,end_ts,time_2_build,N,P,group_data)
app.exec_()   # 关键点
print(1)



# ====================== 环境准备 ======================
# 1) 设定环境变量（必须在导入 matplotlib 之前执行）
import os
os.environ['QT_API'] = 'pyqt5'        # 指定使用 PyQt5 作为 Qt 绑定
os.environ['MPLBACKEND'] = 'QtAgg'    # 指定 Matplotlib 后端为 QtAgg（更推荐，替代 TkAgg）

# 2) 启动 Qt 事件循环（控制台模式下，让 Qt 窗口能实时响应）
%gui qt5

# 3) 检查 matplotlib backend
import matplotlib as mpl
mpl.rcParams.update({
    # 这里按系统常见字体给一串候选，存在则自动生效
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun",
                        "Noto Sans CJK SC", "Source Han Sans SC",
                        "Arial Unicode MS", "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,   # 负号用正常字符，避免被当作缺字形
})

print("backend (before pyplot):", mpl.get_backend())
# 如果不是 QtAgg，强制改为 QtAgg（注意：必须在导入 pyplot 前设置）
mpl.rcParams['backend'] = 'QtAgg'

# 4) 现在再导入 pyplot
import matplotlib.pyplot as plt
print("backend (after pyplot):", mpl.get_backend())

# ====================== 导入依赖 ======================
import sys
# 避免反复执行时 Qt 类重复导入导致崩溃：如果已加载，先删除再导入
if 'draw.pyqt_draw.pyqt_main2' in sys.modules:
    del sys.modules['draw.pyqt_draw.pyqt_main2']

from PyQt5 import QtWidgets
import pyqtgraph as pg
from draw.pyqt_draw.pyqt_main2 import SatelliteViewer
import draw.read_snap_xml as read_snap_xml

# 配置 pyqtgraph：开启抗锯齿，关闭 OpenGL（更稳定）
pg.setConfigOptions(antialias=True)
# pg.setConfigOptions(useOpenGL=False)   # 若驱动或 OpenGL 有问题可显式关闭
# ====================== 基础参数 ======================
# 星座参数：每轨道卫星数 N，轨道平面数 P
N = 36
P = 18


## 绘图一般参数
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


# ====================== 读取数据 ======================
xml_file = r"E:\Data\station_visible_satellites_648_1d_real.xml"
start_ts = 1
end_ts   = 86399

# 解析 XML 得到 group_data，结构：{time_step: {'groups': {...}}}
group_data = read_snap_xml.parse_xml_group_data(xml_file, start_ts, end_ts)

#下面是图变换的。
#rev_group_data,offset = read_snap_xml.modify_group_data(group_data, N=36, groupid=0)
# ====================== 计算 Block 尺寸 ======================
import draw.basic_show.get_satellite_block_info as get_satellite_block_info


times = sorted(group_data.keys())   # 时间序列（所有 step）

# 下述代码是为了计算某个区域的块分布
groupid = 4  # 目标分组 ID

# 存储 Block1/Block2 的长宽随时间变化
widths1, heights1, widths2, heights2 = [], [], [], []
for t in times:
    sats4 = group_data[t]['groups'][groupid]
    # 返回：Block1 (长度, 宽度)，Block2 (长度, 宽度)
    (l1, w1), (l2, w2) = get_satellite_block_info.get_satellite_block_info(sats4, P, N)
    widths1.append(l1); heights1.append(w1)
    widths2.append(l2); heights2.append(w2)

# ====================== 可视化 ======================
# 创建上下两个子图：Block1 和 Block2
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(12, 8), sharex=True,
    gridspec_kw={'height_ratios': [1, 1]}
)
# -------- Block1 曲线 --------
ax1.plot(times, widths1,  label='Block1 Length', color='tab:blue')
ax1.plot(times, heights1, label='Block1 Width',  color='tab:orange')
ax1.set_ylabel('Block1 Size')
ax1.set_title('Block1（一般为主包络）长宽随时间变化')
ax1.legend()
ax1.grid(alpha=0.3)
# -------- Block2 线 --------
ax2.plot(times, widths2,  label='Block2 Length', color='tab:green')
ax2.plot(times, heights2, label='Block2 Width',  color='tab:red')
ax2.set_ylabel('Block2 Size')
ax2.set_title('Block2（有时不存在）长宽随时间变化')
ax2.legend()
ax2.grid(alpha=0.3)
# 公共横轴
plt.xlabel('Time Step')
# 自动调整布局，避免文字重叠
plt.tight_layout()
# 显示图形（独立 Qt 窗口，不阻塞控制台）
plt.show()





# -*- coding: utf-8 -*-
import os
import sys
import argparse
from pathlib import Path

# ===== 解析命令行参数 =====
parser = argparse.ArgumentParser(
    description="Satellite Viewer — pass time-to-build (ttb) from CLI."
)
parser.add_argument("-t", "--ttb", type=int, required=True,
                    help="time-to-build (seconds), e.g. 30/40/.../140")
parser.add_argument("--basicSa", type=int, default=0,
                    help="global basicSa step (default: 0)")
parser.add_argument("--end", type=int, default=22005,
                    help="global end step (default: 22005)")
args = parser.parse_args()

# ===== 轻量环境设置（非 Jupyter）=====
os.environ.setdefault("QT_API", "pyqt5")  # 用 PyQt5
# 若你不使用 matplotlib，可注释掉下一行
os.environ.setdefault("MPLBACKEND", "QtAgg")

# ===== 依赖导入 =====
from PyQt5 import QtWidgets
import pyqtgraph as pg

# 你的项目内模块
from config import DATA_DIR, INPUT_DIR
import genaric2.tegnode as tegnode
import draw.read_snap_xml as read_snap_xml
import draw.basic_functio.write2xml as write2xml
import draw.basic_functio.inter_edge2nodes as inter_edge2nodes
from draw.pyqt_draw.pyqt_main2 import SatelliteViewer

# ===== 常量 / 参数 =====
N = 36
P = 18
TIME_2_BUILD = int(args.ttb)                  # ← 来自命令行
VERSION = os.getenv("TOPOLOGY_VERSION", f"topology_{TIME_2_BUILD}")
MODIFY_DIR = Path(INPUT_DIR) / VERSION / "modify"
FIGURE_DIR = Path(INPUT_DIR) / VERSION / "figure"

# 默认全局时段，可由命令行覆盖
RAW_START, RAW_END = int(args.start), int(args.end)

# ===== 准备目录 =====
MODIFY_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
if __name__ == "__main__":
    # main() 或直接放 argparse + app.exec_()

    # ===== 数据准备 =====
    xml_file = Path(DATA_DIR) / "station_visible_satellites_648_1d_real.xml"
    if not xml_file.exists():
        raise FileNotFoundError(f"XML not found: {xml_file}")

    def slice_group_data(raw_group_data, start, end):
        """裁剪时间区间 [basicSa, end)"""
        return {step: raw_group_data[step] for step in range(start, end) if step in raw_group_data}

    # 解析“组数据”
    print(f"[INFO] Parsing group XML: {xml_file}  range=[{RAW_START},{RAW_END}]")
    raw_group_data = read_snap_xml.parse_xml_group_data(xml_file, RAW_START, RAW_END)

    # 你现有的分片区间
    RANGES = [
        (0,1204),(1204,3669),(3669,4094),(4094,6814),(6814,8485),(8485,11640),
        (11640,13057),(13057,14065),(14065,16604),(16604,18396),
        (18396,19831),(19831,20814),(20814,22005)
    ]
    paths = [MODIFY_DIR / f"interplane_links_{s}_{e}.xml" for s, e in RANGES]

    # 加载 nodes（顺序方式更稳）
    print("[INFO] Loading nodes (sequential)…")
    totalnode = write2xml.load_all_nodes_sequential(paths, tegnode.tegnode_complete)

    # 派生 all_inter_edge 与 pending_edges
    print("[INFO] Building inter-edges & pending-edges…")
    all_inter_edge = inter_edge2nodes.trans_nodes2edges(totalnode, P, N)
    pending_edges  = inter_edge2nodes.trans_nodes2_pendingedges2(
        totalnode, RAW_START, RAW_END, TIME_2_BUILD, P, N
    )

    # 裁剪展示区间（如果你只想展示子区间，可以改 basicSa/end）
    group_data = slice_group_data(raw_group_data, RAW_START, RAW_END)

    # ===== Qt 与可视化 =====
    pg.setConfigOptions(antialias=True)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # 防止被 GC
    if not hasattr(sys.modules[__name__], "_viewer_refs"):
        _viewer_refs = []

    viewer = SatelliteViewer(group_data)
    viewer.setWindowTitle(f"raw behand — TTB={TIME_2_BUILD}")
    viewer.resize(1200, 700)
    viewer.edges_by_step = all_inter_edge
    viewer.pending_links_by_step = pending_edges
    viewer.show()

    _viewer_refs.append(viewer)  # 保持引用

    print("[INFO] Viewer started. Close the window to exit.")
    sys.exit(app.exec_())
    # python -m draw.pymatlab2.xianshi_topology_sequence -t 100
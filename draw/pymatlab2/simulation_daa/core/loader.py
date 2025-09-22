# simulation_daa/core/loader.py
from PyQt5 import QtCore
from pathlib import Path
from typing import Dict, Any
import traceback

from .config_runtime import discover_paths, XML_FILE, RAW_RANGE, P, N, TIME_2_BUILD
import draw.read_snap_xml as read_snap_xml
import draw.basic_functio.write2xml as write2xml
import draw.basic_functio.inter_edge2nodes as inter_edge2nodes
import genaric2.tegnode as tegnode
from utils.graph_utils import build_intra_edges_copies, make_edges_bidirectional

class DataBundle(Dict[str, Any]):
    """载入后的共享数据字典：
       group_data, total_nodes, inter_edges, pending_edges, all_edges, steps
    """

class DataLoader(QtCore.QObject):
    progress = QtCore.pyqtSignal(str)
    done     = QtCore.pyqtSignal(dict)   # DataBundle
    failed   = QtCore.pyqtSignal(str)

    @QtCore.pyqtSlot()
    def run(self):
        try:
            paths = discover_paths()
            if not paths:
                self.progress.emit("⚠ 未找到 interplane_links_*.xml，检查 MODIFY_DIR 与 VERSION")
            else:
                self.progress.emit(f"发现 {len(paths)} 个 XML 待载入")

            # 1) 载入 group_data（一次）
            self.progress.emit("解析 station 可见性 XML（group_data）…")
            start_raw, end_raw = RAW_RANGE
            group_data = read_snap_xml.parse_xml_group_data(XML_FILE, start_raw, end_raw)
            self.progress.emit("group_data OK")

            # 2) 读 nodes 并合并（顺序足够快 & 稳定）
            total_nodes = {}
            for i, fp in enumerate(paths, 1):
                self.progress.emit(f"[{i}/{len(paths)}] 读取: {fp.name}")
                dic = write2xml.xml_to_nodes2(fp, tegnode.tegnode_complete)
                total_nodes.update(dic)

            # 3) nodes → inter_edges + pending_edges
            self.progress.emit("nodes→inter_edges …")
            inter_edges = inter_edge2nodes.trans_nodes2edges(total_nodes, P, N)

            self.progress.emit("计算 pending_edges …")
            steps = sorted(inter_edges.keys())
            if steps:
                start_ts, end_ts = steps[0], steps[-1] + 1
            else:
                start_ts, end_ts = 0, 0

            pending_edges = inter_edge2nodes.trans_nodes2_pendingedges2(
                total_nodes, start_ts, end_ts, TIME_2_BUILD, P, N
            )

            # 4) 加同轨、双向 → all_edges
            self.progress.emit("合并同轨 + 双向化 …")
            intra = build_intra_edges_copies(start_ts, end_ts, P, N)
            # 先双向 inter，再合并
            for t in list(inter_edges.keys()):
                inter_edges[t] = make_edges_bidirectional(inter_edges[t])
            all_edges = {}
            for t in range(start_ts, end_ts):
                a = {}
                if t in intra:
                    for u, vs in intra[t].items():
                        a.setdefault(u, set()).update(vs)
                if t in inter_edges:
                    for u, vs in inter_edges[t].items():
                        a.setdefault(u, set()).update(vs)
                all_edges[t] = a

            bundle = DataBundle(
                group_data=group_data,
                total_nodes=total_nodes,
                inter_edges=inter_edges,
                pending_edges=pending_edges,
                all_edges=all_edges,
                steps=list(range(start_ts, end_ts))
            )
            self.progress.emit("✅ 数据准备完成")
            self.done.emit(bundle)

        except Exception as e:
            self.failed.emit(f"数据载入失败：{e}\n{traceback.format_exc()}")

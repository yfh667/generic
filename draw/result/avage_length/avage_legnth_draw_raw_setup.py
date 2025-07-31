
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
from genaric2.TopoSeqValidator import leftneighbor

if __name__ == "__main__":
    start_ts = 1202
    end_ts = 3320
    N = 36
    P=18
    groupid = 4
    time_2_build = 60

    xml_file = "E:\\Data\\test_raw.xml"
    #xml_file = "E:\\Data\\grid.xml"

    raw_edges_by_step = adjacent2xml.read_steps_from_xml(xml_file)


    xml_file2 = "E:\\Data\\station_visible_satellites_648_1d_real.xml"

    group_data = read_snap_xml.parse_xml_group_data(xml_file2, start_ts, end_ts)
    group_data_modify, off_sets = read_snap_xml.modify_group_data(group_data, N=36, groupid=4)




    # xml_file = "E:\\Data\\test_raw.xml"
    #
    # adjacent2xml.write_steps_to_xml(raw_edges_by_step, xml_file)

    ### then ,we need arange the link reconfiguration,that means,we need arrrange the restablish limitation for the
    nodes = {}
    for step, edges in raw_edges_by_step.items():
        for src, dsts in edges.items():
            x1 = src // N
            y1 = src % N
            for dst in dsts:
                x2 = dst // N
                y2 = dst % N
                # if step ==1231 and x1 ==1 and y1==26:
                #     print(1)
                if x2 == x1:
                    continue
                # 处理右邻居
                if (x1, y1, step) not in nodes:
                    nodes[(x1, y1, step)] = tegnode.tegnode(
                        asc_nodes_flag=False,
                        rightneighbor=(x2, y2, step),
                        leftneighbor=None,
                        state=-1,
                        importance=0,
                    )
                else:
                    nodes[(x1, y1, step)].rightneighbor = (x2, y2, step)

                # 处理左邻居
                if (x2, y2, step) not in nodes:
                    nodes[(x2, y2, step)] = tegnode.tegnode(
                        asc_nodes_flag=False,
                        rightneighbor=None,
                        leftneighbor=(x1, y1, step),
                        state=-1,
                        importance=0,
                    )
                else:
                    nodes[(x2, y2, step)].leftneighbor = (x1, y1, step)
    ### we find the change time and the link
    allchange = {}
    for step in range(start_ts, end_ts):
        changed = []
        for i in range(P - 1):
            for j in range(N):
                n1 = nodes.get((i, j, step))
                n2 = nodes.get((i, j, step + 1))
                if n1 and n2:
                    # 两个时刻都存在右邻居
                    if n1.rightneighbor and n2.rightneighbor:
                        n1_neighbor = (n1.rightneighbor[0], n1.rightneighbor[1])
                        n2_neighbor = (n2.rightneighbor[0], n2.rightneighbor[1])
                        if n1_neighbor != n2_neighbor:
                            changed.append((i, j, n1_neighbor, n2_neighbor))
                    # step时刻没有右邻居，step+1时刻有右邻居
                    elif not n1.rightneighbor and n2.rightneighbor:
                        n2_neighbor = (n2.rightneighbor[0], n2.rightneighbor[1],step)
                        linshi =  (n2.rightneighbor[0], n2.rightneighbor[1])
                        changed.append((i, j, None, linshi))
                        # and we need add the raw rightneighbor for the
                        n2_neighbor_node = nodes.get(n2_neighbor)

                        if n2_neighbor_node.leftneighbor:
                            n2_neighbor_node =  nodes.get(n2_neighbor)
                            forward_n2 = (n2_neighbor_node.leftneighbor[0], n2_neighbor_node.leftneighbor[1],step)
                            changed.append((forward_n2[0], forward_n2[1], n2, None))


        if changed:
            changed_str = ', '.join(
                f'({i},{j}) from {old} to {new}'
                for (i, j, old, new) in changed
            )
            print(f"{step + 1}: {changed_str}")
            allchange[step + 1] = changed  # 用 step+1 作为key

    # 如果我们要查看建链的时间点，只要看allchange的key即可
    # 既然我们有了冲突点，我们只需要提前60s终止冲突链路即可

    for step, changes in allchange.items():
        print(f"{step}:")
        newtime = step - time_2_build
        if newtime < start_ts:
            continue
        for i, j, old, new in changes:
            for k in range(newtime, step):
                src = i * N + j
                # 遍历当前时间k下src的所有目标（dsts 是个 set）
                dsts = raw_edges_by_step[k].get(src, set())
                # 生成需要删除的dst列表（横向链路，即目的节点横坐标和src不一样）
                to_remove = [dst for dst in dsts if dst // N != i]
                # 遍历删除
                for dst in to_remove:
                    raw_edges_by_step[k][src].remove(dst)
                    # 如果是 set()，用 discard(dst) 更安全（不存在不会报错）
                    # raw_edges_by_step[k][src].discard(dst)

    pending_links_by_step = {}  # key: step, value: dict: src -> set(dst)


    def xy_to_id(x, y, N):
        return x * N + y


    pending_links_by_step = {}
    for step, changes in allchange.items():
        newtime = step - time_2_build
        if newtime < start_ts:
            continue
        for i, j, old, new in changes:
            if  new:
                new_dst_id = xy_to_id(*new, N)
                src = i * N + j
                # 标记建链区间内该链路为“pending”
                for k in range(newtime, step):
                    if k not in pending_links_by_step:
                        pending_links_by_step[k] = {}
                    if src not in pending_links_by_step[k]:
                        pending_links_by_step[k][src] = set()
                    pending_links_by_step[k][src].add(new_dst_id)

    # 至此，我们完成了拆链建链动作了。
    # 但是由于我们演示是要用变化后的图去演示的，所以，在这里，我们又重新把变换后的图变换出来，这样我们看图就会看的更加清楚。
    #
    modify_edges_by_step = {}

    for step, edges in raw_edges_by_step.items():  # 一定要遍历raw_edges_by_step！
        modify_edges_by_step[step] = {}

        for src, dsts in edges.items():
            modify_src = read_snap_xml.modify_data(step, src, off_sets)
            for dst in dsts:
                modify_dst = read_snap_xml.modify_data(step, dst, off_sets)
                modify_edges_by_step[step].setdefault(modify_src, set()).add(modify_dst)

    # xml_file = "E:\\Data\\test.xml"
    #
    # adjacent2xml.write_steps_to_xml(modify_edges_by_step, xml_file)

    modify_pending_links_by_step = {}
    for step, src_to_dsts in pending_links_by_step.items():
        for src, dsts in src_to_dsts.items():  # dsts是个集合
            # 只取集合中的第一个（如果你确认每个集合只有一个dst的话）
            dst = next(iter(dsts))

            modify_src = read_snap_xml.modify_data(step, src, off_sets)
            modify_dst = read_snap_xml.modify_data(step, dst, off_sets)

            # 先初始化
            if step not in modify_pending_links_by_step:
                modify_pending_links_by_step[step] = {}
            modify_pending_links_by_step[step].setdefault(modify_src, set()).add(modify_dst)

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
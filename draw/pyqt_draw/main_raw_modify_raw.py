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


xml_file = "E:\\Data\\station_visible_satellites_648_8_h.xml"

group_data = read_snap_xml.parse_xml_group_data(xml_file, start_ts, end_ts)
modify_group_data,off_sets = read_snap_xml.modify_group_data(group_data, N=36, groupid=4)


if __name__ == "__main__":
    pg.setConfigOption('background', 'w')
    time_2_build = 60
    edges_by_step = {}




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

    raw_edges_by_step = {}

    for step, edges in edges_by_step.items():  # step: 时间片
        raw_edges_by_step[step] = {}

        for src, dsts in edges.items():  # src: 起点id, dsts: 终点集合
            raw_src = read_snap_xml.rev_modify_data(step, src, off_sets)

            for dst in dsts:
                raw_dst = read_snap_xml.rev_modify_data(step, dst, off_sets)
                raw_edges_by_step[step].setdefault(raw_src, set()).add(raw_dst)



    nodes = {}
    for step, edges in raw_edges_by_step.items():
        for src, dsts in edges.items():
            x1 = src // N
            y1 = src % N
            for dst in dsts:
                x2 = dst // N
                y2 = dst % N

                if x2==x1:
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

    allchange = {}
    for step in range(start_ts, end_ts):
        changed = []
        for i in range(P - 1):
            for j in range(N):
                n1 = nodes.get((i, j, step))
                n2 = nodes.get((i, j, step + 1))
                if n1 and n2 and n1.rightneighbor and n2.rightneighbor:
                    n1_neighbor = (n1.rightneighbor[0], n1.rightneighbor[1])
                    n2_neighbor = (n2.rightneighbor[0], n2.rightneighbor[1])
                    if n1_neighbor != n2_neighbor:
                        changed.append((i, j, n1_neighbor, n2_neighbor))
        if changed:
            changed_str = ', '.join(
                f'({i},{j}) from {old} to {new}'
                for (i, j, old, new) in changed
            )
            print(f"{step + 1}: {changed_str}")
            allchange[step + 1] = changed  # 用 step+1 作为key



    # 既然我们有了冲突点，我们只需要提前60s终止冲突链路即可

    for step, changes in allchange.items():
        print(f"{step}:")
        newtime = step - time_2_build
        if newtime<start_ts:
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




    modify_edges_by_step = {}

    for step, edges in raw_edges_by_step.items():  # 一定要遍历raw_edges_by_step！
        modify_edges_by_step[step] = {}

        for src, dsts in edges.items():
            modify_src = read_snap_xml.modify_data(step, src, off_sets)
            for dst in dsts:
                modify_dst = read_snap_xml.modify_data(step, dst, off_sets)
                modify_edges_by_step[step].setdefault(modify_src, set()).add(modify_dst)

    xml_file = "E:\\Data\\test.xml"

    adjacent2xml.write_steps_to_xml(modify_edges_by_step, xml_file)

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


#我们接下来需要导出modify_edges_by_step，因为这个是转化后的拓扑形态
    app = QtWidgets.QApplication([])
    viewer = pyqt_main.SatelliteViewer(modify_group_data)
    viewer.edges_by_step = modify_edges_by_step
    viewer.pending_links_by_step = modify_pending_links_by_step  # <<<<<<<< 新增
    viewer.envelopesflag = 1
    viewer.setWindowTitle("Grouped Satellite Visibility - High Performance (PyQtGraph)")
    viewer.resize(1200, 700)
    viewer.show()
    sys.exit(app.exec_())

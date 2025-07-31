
from PyQt5.QtGui import QPainterPath
from PyQt5.QtWidgets import QGraphicsPathItem
import genaric2.tegnode as tegnode
import sys
import matplotlib
matplotlib.use('TkAgg')

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
    P=18
    time_2_build = 60
    #xml_file = "E:\\Data\\test.xml"
    #xml_file = "E:\\Data\\grid.xml"
    xml_file = "E:\\Data\\test_raw.xml"
    modify_edges_by_step = adjacent2xml.read_steps_from_xml(xml_file)

    xml_file2 = "E:\\Data\\station_visible_satellites_648_8_h.xml"

    group_data = read_snap_xml.parse_xml_group_data(xml_file2, start_ts, end_ts)
    group_data_modify, offset = read_snap_xml.modify_group_data(group_data, N=36, groupid=4)

    ## then we need transform the modify edges to the raw edges
    raw_edges_by_step = {}

    for step, edges in modify_edges_by_step.items():  # step: 时间片
        raw_edges_by_step[step] = {}

        for src, dsts in edges.items():  # src: 起点id, dsts: 终点集合
            raw_src = read_snap_xml.rev_modify_data(step, src, offset)

            for dst in dsts:
                raw_dst = read_snap_xml.rev_modify_data(step, dst, offset)
                raw_edges_by_step[step].setdefault(raw_src, set()).add(raw_dst)

    ### then ,we need arange the link reconfiguration,that means,we need arrrange the restablish limitation for the
    nodes = {}
    for step, edges in raw_edges_by_step.items():
        for src, dsts in edges.items():
            x1 = src // N
            y1 = src % N
            for dst in dsts:
                x2 = dst // N
                y2 = dst % N

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
            new_dst_id = xy_to_id(*new, N)
            src = i * N + j
            # 标记建链区间内该链路为“pending”
            for k in range(newtime, step):
                if k not in pending_links_by_step:
                    pending_links_by_step[k] = {}
                if src not in pending_links_by_step[k]:
                    pending_links_by_step[k][src] = set()
                pending_links_by_step[k][src].add(new_dst_id)

    print(1)

    # 统计所有起点
    build_points = set()
    for t in pending_links_by_step:
        for src in pending_links_by_step[t].keys():
            build_points.add(src)

    plt.figure(figsize=(12, 8))
    # 背景（所有网格点，浅灰）
    for x in range(P):
        for y in range(N):
            plt.plot(x, y, 'o', color='#dddddd', markersize=3, zorder=1)

    # 高亮所有建链起点
    for node in build_points:
        x, y = node // N, node % N
        plt.plot(x, y, 'o', color='red', markersize=8, zorder=3)

    plt.xlabel('P (x)')
    plt.ylabel('N (y)')
    plt.title('Active Link Build Start Positions (all time)')

    plt.xlim(-0.5, P - 0.5)
    plt.ylim(-0.5, N - 0.5)
    plt.gca().set_xticks(np.arange(0, P, max(1, P // 12)))
    plt.gca().set_yticks(np.arange(0, N, max(1, N // 12)))

    # y轴正向，0在下，N-1在上
    plt.tight_layout()
    plt.show()

    # 假设 pending_links_by_step 已经有了
    # keys就是有建链的时刻
    # times = np.arange(start_ts, end_ts + 1)
    # counts = np.array([len(pending_links_by_step.get(t, [])) for t in times])
    #
    # plt.figure(figsize=(12, 4))
    # plt.bar(times, counts, width=1.0, color='tab:blue')
    # plt.xlabel('Time Step')
    # plt.ylabel('Number of Link Builds')
    # plt.title('Number of Link Builds per Time Step')
    # plt.tight_layout()
    # plt.show()



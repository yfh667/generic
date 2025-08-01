
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
    raw_edges_by_step = adjacent2xml.read_steps_from_xml(xml_file)

    xml_file2 = "E:\\Data\\station_visible_satellites_648_1d_real.xml"

    group_data = read_snap_xml.parse_xml_group_data(xml_file2, start_ts, end_ts)
    group_data_modify, offset = read_snap_xml.modify_group_data(group_data, N=36, groupid=4)


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
    # 假设 nodes 已经有部分节点
    for step in range(start_ts, end_ts + 1):
        for x in range(P):
            for y in range(N):
                key = (x, y, step)
                if key not in nodes:
                    nodes[key] = tegnode.tegnode(
                        asc_nodes_flag=False,
                        rightneighbor=None,
                        leftneighbor=None,
                        state=-1,
                        importance=0,
                    )

    ### we find the change time and the link
    allchange = {}
    for step in range(start_ts, end_ts):
        changed = []
        for i in range(P - 1):
            for j in range(N):
                n1 = nodes.get((i, j, step))
                n2 = nodes.get((i, j, step + 1))
                # 防御式判断
                # if step==1398 and i==1 and j==25:
                #     print(1)
                # 现在 n1 和 n2 一定都不是 None，才安全用属性
                if n1.rightneighbor and n2.rightneighbor:
                    n1_neighbor = (n1.rightneighbor[0], n1.rightneighbor[1])
                    n2_neighbor = (n2.rightneighbor[0], n2.rightneighbor[1])
                    if n1_neighbor != n2_neighbor:
                        changed.append((i, j, n1_neighbor, n2_neighbor))
                elif not n1.rightneighbor and n2.rightneighbor:
                    n2_neighbor = (n2.rightneighbor[0], n2.rightneighbor[1], step)
                    linshi = (n2.rightneighbor[0], n2.rightneighbor[1])
                    changed.append((i, j, None, linshi))
                    # and we need add the raw rightneighbor for the
                    n2_neighbor_node = nodes.get(n2_neighbor)
                    if n2_neighbor_node and n2_neighbor_node.leftneighbor:
                        his_rightneighbor = (n2.rightneighbor[0], n2.rightneighbor[1])
                        forward_n2 = (n2_neighbor_node.leftneighbor[0], n2_neighbor_node.leftneighbor[1], step)
                        changed.append((forward_n2[0], forward_n2[1], his_rightneighbor, None))
                # check leftneighbor
                if (n1.leftneighbor and n2.leftneighbor):
                    n1_neighbor = (n1.leftneighbor[0], n1.leftneighbor[1])
                    n2_neighbor = (n2.leftneighbor[0], n2.leftneighbor[1])
                    if n1_neighbor != n2_neighbor:
                        changed.append((n1.leftneighbor[0], n1.leftneighbor[1], (i, j), None))

        if changed:
            changed_str = ', '.join(
                f'({i},{j}) from {old} to {new}'
                for (i, j, old, new) in changed
            )
            print(f"{step + 1}: {changed_str}")
            allchange[step + 1] = changed  # 用 step+1 作为key

    # 如果我们要查看建链的时间点，只要看allchange的key即可
    # 既然我们有了冲突点，我们只需要提前60s终止冲突链路即可

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

    # # 1. 统计每个节点作为起点的次数
    # build_counter = Counter()
    # for t in pending_links_by_step:
    #     for src in pending_links_by_step[t].keys():
    #         build_counter[src] += 1
    #
    # # 2. 获取所有次数，准备做归一化上色
    # if build_counter:
    #     max_count = max(build_counter.values())
    # else:
    #     max_count = 1
    #
    # # 3. 绘制
    # plt.figure(figsize=(12, 8))
    # for x in range(P):
    #     for y in range(N):
    #         plt.plot(x, y, 'o', color='#dddddd', markersize=3, zorder=1)
    #
    # # 彩色高亮（次数越多，颜色越深）
    # for node, count in build_counter.items():
    #     x, y = node // N, node % N
    #     # 归一化：0~1
    #     colorval = count / max_count
    #     plt.plot(x, y, 'o', color=plt.cm.jet(colorval), markersize=8, zorder=3)
    #
    # plt.xlabel('P (x)')
    # plt.ylabel('N (y)')
    # plt.title('Node Frequency of Link Build (by color)')
    #
    # plt.xlim(-0.5, P - 0.5)
    # plt.ylim(-0.5, N - 0.5)
    # plt.gca().set_xticks(np.arange(0, P, max(1, P // 12)))
    # plt.gca().set_yticks(np.arange(0, N, max(1, N // 12)))
    # plt.tight_layout()
    #
    # # 加 colorbar（辅助说明颜色含义）
    # sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=1, vmax=max_count))
    # cbar = plt.colorbar(sm, shrink=0.7)
    # cbar.set_label('Number of Link Build Switches')
    #
    # plt.show()

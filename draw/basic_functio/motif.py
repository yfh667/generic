from jupyterlab.commands import ensure_dev

import genaric2.tegnode as tegnode



def write_distinct_motif(x_min, x_max, y_min, y_max, P, N, nodes, option=0):
    for i in range(x_min, x_max + 1):
        for j in range(y_min, y_max + 1):
            muban_define(i, j, P, N, nodes, x_min, x_max, y_min, y_max, option=option)




def muban_define(x,y,  P,N, nodes,x_min, x_max, y_min, y_max,option=0):
    # 1 2 3
    # 4 5 6
    # 7 8 9


    # 4-5
    if option == 0:
        nextnode_x = x+1
        nextnode_y = y

       #4-8
    elif option == 1:

        nextnode_x = x + 1
        nextnode_y = (y - 1 + N) % N

    #
    # 4-6
    elif option == 2:

        nextnode_x = x + 2
        nextnode_y = y


    if nextnode_x < x_min or nextnode_x > x_max:
        return None
    if nextnode_y < y_min or nextnode_y > y_max:
        return None

    setnode_node(x, y, nextnode_x, nextnode_y, nodes)
    return (nextnode_x,nextnode_y)


def setnode_node(start_x, start_y, end_x, end_y, nodes):
    s = (start_x, start_y, -1)
    e = (end_x,   end_y,   -1)

    # --- 确保两端节点对象存在 ---
    if s not in nodes:
        nodes[s] = tegnode.tegnode(
            asc_nodes_flag=False, rightneighbor=None, leftneighbor=None,
            state=-1, importance=0,
        )
    if e not in nodes:
        nodes[e] = tegnode.tegnode(
            asc_nodes_flag=False, rightneighbor=None, leftneighbor=None,
            state=-1, importance=0,
        )

    # --- 对称地断开旧的关联（非常关键） ---
    # s 之前指向的旧 end
    old_e = nodes[s].rightneighbor
    if old_e is not None and old_e in nodes and nodes[old_e].leftneighbor == s:
        nodes[old_e].leftneighbor = None
    # e 之前的旧 start
    old_s = nodes[e].leftneighbor
    if old_s is not None and old_s in nodes and nodes[old_s].rightneighbor == e:
        nodes[old_s].rightneighbor = None

    # --- 建立新的双向一致的关联 ---
    nodes[s].rightneighbor = e
    nodes[e].leftneighbor  = s

# def setnode_node(start_x, start_y, end_x, end_y,nodes):
#     # 首先，我们要先覆盖end_X 和end_Y的内容
#     # 处理左邻居
#     if (start_x, start_y)==(9,30) or (end_x, end_y)==(9,30):
#         print(1)
#     if ( end_x, end_y, -1) not in nodes:
#         nodes[( end_x, end_y, -1)] = tegnode.tegnode(
#             asc_nodes_flag=False,
#             rightneighbor=None,
#             leftneighbor=(start_x, start_y, -1) ,
#             state=-1,
#             importance=0,
#         )
#     else:
#         if  nodes[( end_x, end_y, -1)].leftneighbor:
#             left = nodes[( end_x, end_y, -1)].leftneighbor
#             nodes[left].rightneighbor = None
#         nodes[( end_x, end_y, -1)].leftneighbor =(start_x, start_y, -1)
#
#         nodes[( end_x, end_y, -1)].leftneighbor =(start_x, start_y, -1)
#
#
#     if (start_x, start_y, -1) not in nodes:
#         nodes[(start_x, start_y, -1)] = tegnode.tegnode(
#             asc_nodes_flag=False,
#             rightneighbor=( end_x, end_y, -1),
#             leftneighbor=None,
#             state=-1,
#             importance=0,
#         )
#     else:
#         nodes[(start_x, start_y, -1) ].rightneighbor =( end_x, end_y, -1)




from collections import defaultdict
from typing import Iterable, Mapping

def transform_nodes_2_adjacent(nodes,P,N):
    """
    nodes: dict[(p,s,step) -> tegnode]  或  dict[(p,s) -> tegnode]
    返回:  { (p,s): [(p2,s2), ...], ... }  # 无向邻接表，已去重
    只用 rightneighbor / leftneighbor 两个字段；为 None 的会跳过。
    """

    adj = {}
    for i in range(P - 1):
        for j in range(N):
            nowdes = nodes.get((i, j, -1))
            if nowdes is None or nowdes.rightneighbor is None:
                continue
            rightneighbor = nowdes.rightneighbor
            nowid = i * N + j
            rightneighbor_id = rightneighbor[0] * N + rightneighbor[1]
            # 无向边：两边都加
            adj.setdefault(nowid, set()).add(rightneighbor_id)
        #    adj.setdefault(rightneighbor_id, set()).add(nowid)
            # 你如果有 leftneighbor 也可以类似处理
            # if nowdes.leftneighbor is not None:
            #     leftneighbor = nowdes.leftneighbor
            #     leftneighbor_id = leftneighbor[0] * N + leftneighbor[1]
            #     adj.setdefault(nowid, set()).add(leftneighbor_id)
            #     adj.setdefault(leftneighbor_id, set()).add(nowid)
    return adj




def transform_nodes_2_rawedge(nodes, P, N, start_ts, end_ts):
    """
    nodes: dict[(p, s, step) -> tegnode]  或  dict[(p, s) -> tegnode]
    返回:  {step: { nowid: set([neighbor_id, ...]), ... }, ... }
    只用 rightneighbor/leftneighbor 两个字段；为 None 的会跳过。
    """
    edge_by_step = {}

    for step in range(start_ts, end_ts):
        for i in range(P):
            for j in range(N):
                nowdes = nodes.get((i, j, step))  # 或改成 (i, j, step) 按需
                if nowdes is None:
                    continue

                nowid = i * N + j

                # 处理右邻
                if nowdes.rightneighbor is not None:
                    rightneighbor = nowdes.rightneighbor
                    rightneighbor_id = rightneighbor[0] * N + rightneighbor[1]
                    edge_by_step.setdefault(step, {}).setdefault(nowid, set()).add(rightneighbor_id)
                    # 无向：对方也加自己
                    # edge_by_step.setdefault(step, {}).setdefault(rightneighbor_id, set()).add(nowid)



    return edge_by_step


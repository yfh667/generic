from copy import deepcopy
import copy

import genaric2.tegnode as tegnode
import  draw.pymatlab2.basic.assignlink as assignlink
from collections.abc import MutableMapping

def _clone_node(n):
    # 你的字段都是标量/三元组，浅拷贝就够；若后续加了可变字段再改成深拷贝
    return tegnode.tegnode_new(
        asc_nodes_region_id = getattr(n, "asc_nodes_region_id", -1),
        rightneighbor       = getattr(n, "rightneighbor", None),
        leftneighbor        = getattr(n, "leftneighbor", None),
        left_state          = getattr(n, "left_state", -1),
        right_state         = getattr(n, "right_state", -1),
        node_type           = getattr(n, "node_type", -1),
        timelast            = getattr(n, "timelast", -1),
    )

class COWNodes(MutableMapping):
    """
    Copy-On-Write 节点视图：
    - 读：优先 overlay；否则从 base 取 *克隆* 放入 overlay 再返回（避免改到 base）
    - 写：只写 overlay
    - base 可以是 dict，或另一层 COWNodes（允许多层叠加）
    """
    def __init__(self, base):
        self.base = base
        self.overlay = {}

    # --- 必需接口 ---
    def __getitem__(self, key):
        if key in self.overlay:
            return self.overlay[key]
        if key in self.base:
            v = _clone_node(self.base[key])
            self.overlay[key] = v
            return v
        # 不存在时，按你习惯返回一个“新节点”
        v = tegnode.tegnode_new()
        self.overlay[key] = v
        return v

    def get(self, key, default=None):
        if key in self.overlay:
            return self.overlay[key]
        if key in self.base:
            v = _clone_node(self.base[key])
            self.overlay[key] = v
            return v
        return default

    def __setitem__(self, key, value):
        self.overlay[key] = value

    def __contains__(self, key):
        return key in self.overlay or key in self.base

    def __delitem__(self, key):
        if key in self.overlay:
            del self.overlay[key]
        else:
            raise KeyError(key)

    def __len__(self):
        return len(set(self.overlay) | set(self.base))

    def __iter__(self):
        seen = set()
        for k in self.overlay:
            seen.add(k); yield k
        for k in self.base:
            if k not in seen:
                yield k

import math
# def get_no_conflict_link(raw_edges_by_step,start_ts,end_ts,time_2_build,N,P):
# ### then ,we need arange the link reconfiguration,that means,we need arrrange the restablish limitation for the
#
#     ### then ,we need arange the link reconfiguration,that means,we need arrrange the restablish limitation for the
#     nodes = {}
#     for step, edges in raw_edges_by_step.items():
#         for basicSa, dsts in edges.items():
#             x1 = basicSa // N
#             y1 = basicSa % N
#             for dst in dsts:
#                 x2 = dst // N
#                 y2 = dst % N
#                 # if step ==1231 and x1 ==1 and y1==26:
#                 #     print(1)
#                 if x2 == x1:
#                     continue
#                 # 处理右邻居
#                 if (x1, y1, step) not in nodes:
#                     nodes[(x1, y1, step)] = tegnode.tegnode(
#                         asc_nodes_flag=False,
#                         rightneighbor=(x2, y2, step),
#                         leftneighbor=None,
#                         state=-1,
#                         importance=0,
#                     )
#                 else:
#                     nodes[(x1, y1, step)].rightneighbor = (x2, y2, step)
#
#                 # 处理左邻居
#                 if (x2, y2, step) not in nodes:
#                     nodes[(x2, y2, step)] = tegnode.tegnode(
#                         asc_nodes_flag=False,
#                         rightneighbor=None,
#                         leftneighbor=(x1, y1, step),
#                         state=-1,
#                         importance=0,
#                     )
#                 else:
#                     nodes[(x2, y2, step)].leftneighbor = (x1, y1, step)
#     # 假设 nodes 已经有部分节点
#     for step in range(start_ts, end_ts + 1):
#         for x in range(P):
#             for y in range(N):
#                 key = (x, y, step)
#                 if key not in nodes:
#                     nodes[key] = tegnode.tegnode(
#                         asc_nodes_flag=False,
#                         rightneighbor=None,
#                         leftneighbor=None,
#                         state=-1,
#                         importance=0,
#                     )
#
#     ### we find the change time and the link
#     allchange = {}
#     for step in range(start_ts, end_ts):
#         changed = []
#         for i in range(P - 1):
#             for j in range(N):
#                 n1 = nodes.get((i, j, step))
#                 n2 = nodes.get((i, j, step + 1))
#                 # 防御式判断
#                 # if step==1398 and i==1 and j==25:
#                 #     print(1)
#                 # 现在 n1 和 n2 一定都不是 None，才安全用属性
#                 if n1.rightneighbor and n2.rightneighbor:
#                     n1_neighbor = (n1.rightneighbor[0], n1.rightneighbor[1])
#                     n2_neighbor = (n2.rightneighbor[0], n2.rightneighbor[1])
#                     if n1_neighbor != n2_neighbor:
#                         changed.append((i, j, n1_neighbor, n2_neighbor))
#                 elif not n1.rightneighbor and n2.rightneighbor:
#                     n2_neighbor = (n2.rightneighbor[0], n2.rightneighbor[1], step)
#                     linshi = (n2.rightneighbor[0], n2.rightneighbor[1])
#                     changed.append((i, j, None, linshi))
#                     # and we need add the raw rightneighbor for the
#                     n2_neighbor_node = nodes.get(n2_neighbor)
#                     if n2_neighbor_node and n2_neighbor_node.leftneighbor:
#                         his_rightneighbor = (n2.rightneighbor[0], n2.rightneighbor[1])
#                         forward_n2 = (n2_neighbor_node.leftneighbor[0], n2_neighbor_node.leftneighbor[1], step)
#                         changed.append((forward_n2[0], forward_n2[1], his_rightneighbor, None))
#                 # check leftneighbor
#                 if (n1.leftneighbor and n2.leftneighbor):
#                     n1_neighbor = (n1.leftneighbor[0], n1.leftneighbor[1])
#                     n2_neighbor = (n2.leftneighbor[0], n2.leftneighbor[1])
#                     if n1_neighbor != n2_neighbor:
#                         changed.append((n1.leftneighbor[0], n1.leftneighbor[1], (i, j), None))
#
#         if changed:
#             changed_str = ', '.join(
#                 f'({i},{j}) from {old} to {new}'
#                 for (i, j, old, new) in changed
#             )
#             print(f"{step + 1}: {changed_str}")
#             allchange[step + 1] = changed  # 用 step+1 作为key
#
#     # 如果我们要查看建链的时间点，只要看allchange的key即可
#     # 既然我们有了冲突点，我们只需要提前60s终止冲突链路即可
#
#     def xy_to_id(x, y, N):
#         return x * N + y
#
#
#     pending_links_by_step = {}
#     for step, changes in allchange.items():
#         newtime = step - time_2_build
#         if newtime < start_ts:
#             continue
#         for i, j, old, new in changes:
#             if  new:
#                 new_dst_id = xy_to_id(*new, N)
#                 basicSa = i * N + j
#                 # 标记建链区间内该链路为“pending”
#                 for k in range(newtime, step):
#                     if k not in pending_links_by_step:
#                         pending_links_by_step[k] = {}
#                     if basicSa not in pending_links_by_step[k]:
#                         pending_links_by_step[k][basicSa] = set()
#                     pending_links_by_step[k][basicSa].add(new_dst_id)
#
#     for step, changes in allchange.items():
#         print(f"{step}:")
#         newtime = step - time_2_build
#         if newtime < start_ts:
#             continue
#         for i, j, old, new in changes:
#             for k in range(newtime, step):
#                 basicSa = i * N + j
#                 # 遍历当前时间k下src的所有目标（dsts 是个 set）
#                 dsts = raw_edges_by_step[k].get(basicSa, set())
#                 # 生成需要删除的dst列表（横向链路，即目的节点横坐标和src不一样）
#                 to_remove = [dst for dst in dsts if dst // N != i]
#                 # 遍历删除
#                 for dst in to_remove:
#                     raw_edges_by_step[k][basicSa].remove(dst)
#                     # 如果是 set()，用 discard(dst) 更安全（不存在不会报错）
#                     # raw_edges_by_step[k][basicSa].discard(dst)
#
#
#     return raw_edges_by_step,pending_links_by_step

def region_in_communication(n1node,n1neighbor_node):

    if n1node.asc_nodes_region_id!=-1 and n1node.asc_nodes_region_id==n1neighbor_node.asc_nodes_region_id:
        return n1node.asc_nodes_region_id
    else:
        return -1




def adjust_link_nodes(i, j, step, nodes, time2setup, start_ts, end_ts, option=0):
    if option == 0 or option == 2:
        for k in range(time2setup):
            bias = step - k
            if bias >= start_ts:
                nownode = nodes[i, j, bias]
                neighbor_key = nownode.rightneighbor
                nownode.rightneighbor = None

                if neighbor_key is not None:
                    neighbor_node = nodes.get(neighbor_key)
                    if neighbor_node is not None:
                        neighbor_node.leftneighbor = None

    elif option == 1:
        for k in range(time2setup):
            bias = step + k
            if bias < end_ts:
                nownode = nodes[i, j, bias]
                neighbor_key = nownode.rightneighbor
                nownode.rightneighbor = None
                if neighbor_key is not None:
                    neighbor_node = nodes.get(neighbor_key)
                    if neighbor_node is not None:
                        neighbor_node.leftneighbor = None


def adjust_link_nodes_test(i, j, step, nodes, time2setup, start_ts, end_ts, option=0):
    # 前面预预建链
    if option == 0 :
        future_node = nodes[i, j, step+1]
        r = future_node.rightneighbor
        if r is  None:# 有可能存在后面已经不链节点了，因此，就不需要提前准备了
            return
        rx =r[0]
        ry = r[1]



        lowbias = step - time2setup+1


        if lowbias>=start_ts:
            # print("there")
            for k in range(time2setup):
                bias = step - k
                if bias >= start_ts:
                    # nownode = nodes[i, j, bias]

                    assignlink.assign_Link((i, j, bias), (rx,ry,bias), nodes, 0,
                                           1, k+1)


        else:     # 如果出现，建联时间不够呢？因此此处需要补偿这一行为
          # print("here")
          for k in range(time2setup):
              bias = start_ts +k
              if bias < end_ts:
                  # nownode = nodes[i, j, bias]

                  assignlink.assign_Link((i, j, bias), (rx, ry, bias), nodes, 0,
                                         1,time2setup-k)

# 后面预建链
    elif option == 1:
        future_node = nodes[i, j, step+1]
        r = future_node.rightneighbor
        if r is  None:# 有可能存在后面已经不链节点了，因此，就不需要提前准备了
            return
        rx =r[0]
        ry = r[1]
        for k in range(time2setup):
            bias = step +1+ k
            if bias < end_ts:
                assignlink.assign_Link((i, j, bias), (rx, ry, bias), nodes, 0,
                                       1, time2setup-k)





import draw.read_snap_xml as read_snap_xml

import draw.basic_functio.motif as motif
def _is_triplet(v):
    return isinstance(v, tuple) and len(v) == 3

def _xy(nei):
    """从 (x,y,z) 取 (x,y)；若 nei 非三元组返回 None"""
    if _is_triplet(nei):
        return nei[0], nei[1]
    return None

# we check ,whether (x,y,z)是否处于热点链接，

def hot_link_flag(x,y,time,time2setup,nodes,start_ts,end_ts ):
    flag=0
    for i in range(time2setup):
        nowtime = time -i
        if nowtime<start_ts:
            return  flag
        n1node = nodes[(x,y,nowtime)]
        if not n1node.rightneighbor:
            continue
        n1neighbor = n1node.rightneighbor
        n1neighbor_node = nodes[n1neighbor]
        if  region_in_communication(n1node,n1neighbor_node)!=-1:
            flag=1
            return flag

    return flag






def get_no_conflict_link_test(raw_edges_by_step,offsets,rects,start_ts,end_ts,time_2_build,N,P):
    nodes = {}
    hotspot_keys = set()
    for step in range(start_ts, end_ts):
        hotspot_keys = set()  # <--- 每个 step 单独新建！！
        # 1. 初始化热点区域
        for groupid, rect_tuple in rects.items():
            if not isinstance(rect_tuple, (list, tuple)):
                rect_tuple = (rect_tuple,)
            for rect in rect_tuple:
                if rect is None:
                    continue
                xmin, xmax, ymin, ymax = rect
                for x in range(xmin, xmax + 1):
                    for y in range(ymin, ymax + 1):
                        number = x * N + y
                        modify_number = read_snap_xml.rev_modify_data(step, number, offsets)
                        real_x = modify_number // N
                        real_y = modify_number % N
                        key = (real_x, real_y, step)
                        nodes[key] = tegnode.tegnode_new(
                            asc_nodes_region_id=groupid,
                            rightneighbor=None,
                            right_state=-1,
                            leftneighbor=None,
                            left_state=-1,
                            node_type=-1,
                            timelast=-1,
                        )
                        hotspot_keys.add(key)
        # 2. 补齐全图节点
        for x in range(P):
            for y in range(N):
                key = (x, y, step)
                if key not in hotspot_keys:
                    nodes[key] = tegnode.tegnode_new(
                        asc_nodes_region_id=-1,
                        rightneighbor=None,
                        right_state=-1,
                        leftneighbor=None,
                        left_state=-1,
                        node_type=-1,
                        timelast=-1,
                    )
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

                assignlink.assign_Link((x1, y1, step), (x2, y2, step), nodes, 1,
                                       1, 0)

    change_terminal = []
    consider_terminal_pair = []
    for step in range(start_ts, end_ts-1):

        for i in range(P - 1):
            for j in range(N):

                n1 = nodes.get((i, j, step))
                n2 = nodes.get((i, j, step + 1))

                if n1.rightneighbor and n2.rightneighbor:
                    #   if _is_triplet(n1.rightneighbor) and _is_triplet(n2.rightneighbor):

                    n1_neighbor = (n1.rightneighbor[0], n1.rightneighbor[1])
                    n2_neighbor = (n2.rightneighbor[0], n2.rightneighbor[1])

                    if n1_neighbor != n2_neighbor:
                        consider_terminal_pair.append([(i, j, step),(n2.rightneighbor[0], n2.rightneighbor[1],step)])
                elif not n1.rightneighbor and n2.rightneighbor:

                    consider_terminal_pair.append([(i, j, step),(n2.rightneighbor[0], n2.rightneighbor[1],step)])


    for terminals in consider_terminal_pair:
        start_terminal = terminals[0]
        target_terminal = terminals[1]
        flag1 =hot_link_flag(start_terminal[0],start_terminal[1],start_terminal[2],time_2_build,nodes,start_ts, end_ts)
        flag2 =hot_link_flag(target_terminal[0],target_terminal[1],target_terminal[2],time_2_build,nodes,start_ts, end_ts)
        if flag1 or flag2:
            # 说明了两个点其中有一个或者都是区域内部链接，因此，后面链路进行拖鞋
            change_terminal.append((start_terminal[0], start_terminal[1], start_terminal[2], 1))
        else:
            change_terminal.append((start_terminal[0], start_terminal[1], start_terminal[2], 0))



    for terminal in change_terminal:
         if terminal[3]==1:
            #后面是于预建联
            adjust_link_nodes_test(terminal[0], terminal[1],  terminal[2], nodes, time_2_build, start_ts, end_ts, option=1)

         else:
            # 前面预建链
            adjust_link_nodes_test(terminal[0], terminal[1],  terminal[2], nodes, time_2_build, start_ts, end_ts, option=0)



    return change_terminal,nodes

# def get_no_conflict_link_test(raw_edges_by_step,offsets,rects,start_ts,end_ts,time_2_build,N,P):
#     nodes = {}
#     hotspot_keys = set()
#     for step in range(start_ts, end_ts):
#         hotspot_keys = set()  # <--- 每个 step 单独新建！！
#         # 1. 初始化热点区域
#         for groupid, rect_tuple in rects.items():
#             if not isinstance(rect_tuple, (list, tuple)):
#                 rect_tuple = (rect_tuple,)
#             for rect in rect_tuple:
#                 if rect is None:
#                     continue
#                 xmin, xmax, ymin, ymax = rect
#                 for x in range(xmin, xmax + 1):
#                     for y in range(ymin, ymax + 1):
#                         number = x * N + y
#                         modify_number = read_snap_xml.rev_modify_data(step, number, offsets)
#                         real_x = modify_number // N
#                         real_y = modify_number % N
#                         key = (real_x, real_y, step)
#                         nodes[key] = tegnode.tegnode_new(
#                             asc_nodes_region_id=groupid,
#                             rightneighbor=None,
#                             right_state=-1,
#                             leftneighbor=None,
#                             left_state=-1,
#                             node_type=-1,
#                             timelast=-1,
#                         )
#                         hotspot_keys.add(key)
#         # 2. 补齐全图节点
#         for x in range(P):
#             for y in range(N):
#                 key = (x, y, step)
#                 if key not in hotspot_keys:
#                     nodes[key] = tegnode.tegnode_new(
#                         asc_nodes_region_id=-1,
#                         rightneighbor=None,
#                         right_state=-1,
#                         leftneighbor=None,
#                         left_state=-1,
#                         node_type=-1,
#                         timelast=-1,
#                     )
#     for step, edges in raw_edges_by_step.items():
#         for src, dsts in edges.items():
#             x1 = src // N
#             y1 = src % N
#             for dst in dsts:
#                 x2 = dst // N
#                 y2 = dst % N
#                 # if step ==1231 and x1 ==1 and y1==26:
#                 #     print(1)
#                 if x2 == x1:
#                     continue
#
#                 assignlink.assign_Link((x1, y1, step), (x2, y2, step), nodes, 1,
#                                        1, 0)
#
#     # cankao_nodes= copy.deepcopy(nodes)
#     #
#     for step in range(start_ts, end_ts-1):
#         change_terminal = []
#
#         for i in range(P - 1):
#             for j in range(N):
#                 # if i==9 and j==19 and step==3399:
#                 #     print(1)
#                 if step==12412 and (i,j)==(14,1):
#                     print(1)
#
#
#
#                 n1 = nodes.get((i, j, step))
#                 n2 = nodes.get((i, j, step + 1))
#                 # if (i, j, step) ==(5,32,2519):
#                 #     print(1)
#                 # if n1.right_state==0:
#                 #     continue
#                 # 防御式判断
#                 # 这个表明，某个点在step+1时，其链接改变了，此刻，我们需要直接修改
#                 if n1.rightneighbor and n2.rightneighbor:
#               #   if _is_triplet(n1.rightneighbor) and _is_triplet(n2.rightneighbor):
#
#                     n1_neighbor = (n1.rightneighbor[0], n1.rightneighbor[1])
#                     n2_neighbor = (n2.rightneighbor[0], n2.rightneighbor[1])
#                     if n1_neighbor != n2_neighbor:
#                         # 这里，我们就要判断，此刻step+1的链接和step的链接是否有一方是处于区域内部的链接
#                         # 如果不一致，我们就需要修改
#
#                         # 这里还是要注意，当前不重要不代表之前时刻不重要
#
#
#                         #
#                         n1_neighbor_node = nodes.get((n1.rightneighbor[0], n1.rightneighbor[1],step))
#                         n2_neighbor_node = nodes.get((n2.rightneighbor[0], n2.rightneighbor[1],step))
#                         n1_region_group_id = region_in_communication(n1,n1_neighbor_node)
#                         n2_region_group_id = region_in_communication(n2,n2_neighbor_node)
#                         if n1_region_group_id!=-1:
#                             change_terminal.append((i, j, step,1))
#                         else:
#                             change_terminal.append((i, j, step,0))
#
#
#                         # ##
#                         # if n1_region_group_id==-1 and n2_region_group_id==-1:
#                         #     # 最简单的，就是后面覆盖前面，前面的要断链路
#                         #     adjust_link_nodes_test(i, j, step, nodes, time_2_build,start_ts, end_ts, option=0)
#                         # elif n1_region_group_id!=-1 and n2_region_group_id==-1:
#                         #     # 说明前面是区域内部链路，后面是区域外部链路，后面的链路需要妥协
#                         #     adjust_link_nodes_test(i, j, step, nodes, time_2_build, start_ts, end_ts, option=1)
#                         # elif n1_region_group_id==-1 and n2_region_group_id !=-1:
#                         #     # 说明前是其余外部链路，后面是区域内部链路，前面链路需要进行断链为后面准备,后面覆盖前面的
#                         #     adjust_link_nodes_test(i, j, step, nodes, time_2_build, start_ts, end_ts, option=2)
#                 # the
#                 elif not n1.rightneighbor and n2.rightneighbor:
#                     n2_neighbor_node = nodes.get((n2.rightneighbor[0], n2.rightneighbor[1], step))
#                     #  we need check ,the link at time=step+1: n2--n2_neighbor_node
#                     n2_region_group_id = region_in_communication(n2, n2_neighbor_node)
#
#
#                     change_terminal.append((i, j, step, 0))
#                     # if n2_region_group_id != -1:
#                     #
#                     # else:
#                     #     change_terminal.append((i, j, step, 1))
#                 elif n1.rightneighbor and not n2.rightneighbor:
#                     # attention,存在一种可能性，那就是n1是重要区间，n2不是重要区间
#                     print(1)
#
#
#
#         for terminal in change_terminal:
#     #        adjust_link_nodes_test(terminal[0], terminal[1], step, nodes, time_2_build, start_ts, end_ts, option=0)
#
#             if terminal[3]==1:
#                 #     # 说明前面是区域内部链路，后面是区域外部链路，后面的链路需要妥协
#                 adjust_link_nodes_test(terminal[0], terminal[1], step, nodes, time_2_build, start_ts, end_ts, option=1)
#             else:
#                 #     # 最简单的，就是后面覆盖前面，前面的要断链路
#                 adjust_link_nodes_test(terminal[0], terminal[1], step, nodes, time_2_build, start_ts, end_ts, option=0)
#
#                 # changed.append((i, j, n1_neighbor, n2_neighbor))
#     #             elif not n1.rightneighbor and n2.rightneighbor:
#     #                 # 说明前面无邻居，后面又邻居，注意，一开始设计的时候，已经优先为区域内考虑了
#     #                 # 我们事实上，是要查看后面节点的
#     #           #   elif (not _is_triplet(n1.rightneighbor)) and _is_triplet(n2.rightneighbor):
#     #           #       if (i, j, step)==(5,32,2519):
#     #           #           print(2)
#     #
#     #                 # 我们要进行回溯，要查询前面的链路，是否存在区域内部链路
#     #                 n2_neighbor_node = nodes.get((n2.rightneighbor[0], n2.rightneighbor[1], step))
#     #                 n2_region_group_id = region_in_communication(n2, n2_neighbor_node)
#     #                # n1_region_group_id = -1
#     #
#     #                 if n2_region_group_id:
#     #                     # 最简单的，就是后面覆盖前面，前面的要断链路
#     #                     adjust_link_nodes_test(i, j, step, nodes, time_2_build, start_ts, end_ts, option=0)
#     #
#     #
#     #
#     #
#     # # 前面有neighbor 后面没neighbor，所以，就要考虑左邻居的问题
#     #             elif  n1.rightneighbor and not n2.rightneighbor:
#     #             # elif _is_triplet(n1.rightneighbor) and (not _is_triplet(n2.rightneighbor)):
#     #
#     #                 right_neighbor = n1.rightneighbor
#     #                 # 接下来，我们要注意了，对于n1.rightneighbor,我们要知道，right neighbor如果切换链路
#     #                 # 是会影响到当前的n1 以及之前的链路的，所以这个要注意
#     #                 # 如果是，我们就需要进行调整
#     #
#     #                 n1 = nodes.get((right_neighbor[0], right_neighbor[1], step))
#     #                 n2 = nodes.get((right_neighbor[0], right_neighbor[1], step+1))
#     #
#     #
#     #
#     #                 if n1.leftneighbor and n2.leftneighbor:
#     #                     n1_neighbor = (n1.leftneighbor[0], n1.leftneighbor[1])
#     #                     n2_neighbor = (n2.leftneighbor[0], n2.leftneighbor[1])
#     #                     if n1_neighbor != n2_neighbor:
#     #                         # 这里，我们就要判断，此刻step+1的链接和step的链接是否有一方是处于区域内部的链接
#     #                         # 如果不一致，我们就需要修改
#     #
#     #
#     #                         n1_neighbor_node = nodes.get((n1.leftneighbor[0], n1.leftneighbor[1],step))
#     #                         n2_neighbor_node = nodes.get((n2.leftneighbor[0], n2.leftneighbor[1],step+1))
#     #
#     #                         n1_region_group_id = region_in_communication(n1_neighbor_node,n1)
#     #                         n2_region_group_id = region_in_communication(n2_neighbor_node,n2)
#     #                         ##
#     #                         if n1_region_group_id==-1 and n2_region_group_id==-1:
#     #                             # 最简单的，就是后面覆盖前面，前面的要断链路
#     #                             adjust_link_nodes_test(i, j, step, nodes, time_2_build,start_ts, end_ts, option=0)
#     #                         elif n1_region_group_id!=-1 and n2_region_group_id==-1:
#     #                             # 说明前面是区域内部链路，后面是区域外部链路，后面的链路需要妥协
#     #                             adjust_link_nodes_test(n2.leftneighbor[0], n2.leftneighbor[1], step+1, nodes, time_2_build, start_ts, end_ts, option=1)
#     #
#     #
#     #                         elif n1_region_group_id==-1 and n2_region_group_id !=-1:
#     #                             # 说明前是其余外部链路，后面是区域内部链路，前面链路需要进行断链为后面准备,后面覆盖前面的
#     #                             adjust_link_nodes_test(i, j, step, nodes, time_2_build, start_ts, end_ts, option=2)
#
#
#     return nodes
#

# def get_no_conflict_link_test(raw_edges_by_step,offsets,rects,start_ts,end_ts,time_2_build,N,P):
#
#     nodes = {}
#     hotspot_keys = set()
#
#
#     for step in range(start_ts, end_ts):
#         hotspot_keys = set()  # <--- 每个 step 单独新建！！
#         # 1. 初始化热点区域
#         for groupid, rect_tuple in rects.items():
#             if not isinstance(rect_tuple, (list, tuple)):
#                 rect_tuple = (rect_tuple,)
#             for rect in rect_tuple:
#                 if rect is None:
#                     continue
#                 xmin, xmax, ymin, ymax = rect
#                 for x in range(xmin, xmax + 1):
#                     for y in range(ymin, ymax + 1):
#                         number = x * N + y
#                         modify_number = read_snap_xml.rev_modify_data(step, number, offsets)
#                         real_x = modify_number // N
#                         real_y = modify_number % N
#                         key = (real_x, real_y, step)
#                         nodes[key] = tegnode.tegnode_new(
#                             asc_nodes_region_id=groupid,
#                             rightneighbor=None,
#                             right_state=-1,
#                             leftneighbor=None,
#                             left_state=-1,
#                             node_type=-1,
#                             timelast=-1,
#                         )
#                         hotspot_keys.add(key)
#         # 2. 补齐全图节点
#         for x in range(P):
#             for y in range(N):
#                 key = (x, y, step)
#                 if key not in hotspot_keys:
#                     nodes[key] = tegnode.tegnode_new(
#                         asc_nodes_region_id=-1,
#                         rightneighbor=None,
#                         right_state=-1,
#                         leftneighbor=None,
#                         left_state=-1,
#                         node_type=-1,
#                         timelast=-1,
#                     )
#
#
#     for step, edges in raw_edges_by_step.items():
#         for src, dsts in edges.items():
#             x1 = src // N
#             y1 = src % N
#             for dst in dsts:
#                 x2 = dst // N
#                 y2 = dst % N
#                 # if step ==1231 and x1 ==1 and y1==26:
#                 #     print(1)
#                 if x2 == x1:
#                     continue
#
#                 assignlink.assign_Link((x1, y1, step), (x2, y2, step), nodes, 1,
#                                        1, 0)
#
#                 # nodes[(x1, y1, step)].rightneighbor = (x2, y2, step)
#                 # nodes[(x1, y1, step)].right_state = 1
#                 # nodes[(x1, y1, step)].node_type = 1
#                 # nodes[(x2, y2, step)].timelast = 0
#                 #
#                 #
#                 # nodes[(x2, y2, step)].leftneighbor = (x1, y1, step)
#                 # nodes[(x2, y2, step)].left_state =1
#
#
#
#
#     for step in range(start_ts, end_ts-1):
#
#         for i in range(P - 1):
#             for j in range(N):
#                 if i==9 and j==19 and step==3399:
#                     print(1)
#                 n1 = nodes.get((i, j, step))
#                 n2 = nodes.get((i, j, step + 1))
#                 # if (i, j, step) ==(5,32,2519):
#                 #     print(1)
#                 # if n1.right_state==0:
#                 #     continue
#                 # 防御式判断
#                 # 这个表明，某个点在step+1时，其链接改变了，此刻，我们需要直接修改
#                 if n1.rightneighbor and n2.rightneighbor:
#               #   if _is_triplet(n1.rightneighbor) and _is_triplet(n2.rightneighbor):
#
#                     n1_neighbor = (n1.rightneighbor[0], n1.rightneighbor[1])
#                     n2_neighbor = (n2.rightneighbor[0], n2.rightneighbor[1])
#                     if n1_neighbor != n2_neighbor:
#                         # 这里，我们就要判断，此刻step+1的链接和step的链接是否有一方是处于区域内部的链接
#                         # 如果不一致，我们就需要修改
#
#
#                         n1_neighbor_node = nodes.get((n1.rightneighbor[0], n1.rightneighbor[1],step))
#                         n2_neighbor_node = nodes.get((n2.rightneighbor[0], n2.rightneighbor[1],step))
#                         n1_region_group_id = region_in_communication(n1,n1_neighbor_node)
#                         n2_region_group_id = region_in_communication(n2,n2_neighbor_node)
#                         ##
#                         if n1_region_group_id==-1 and n2_region_group_id==-1:
#                             # 最简单的，就是后面覆盖前面，前面的要断链路
#                             adjust_link_nodes_test(i, j, step, nodes, time_2_build,start_ts, end_ts, option=0)
#                         elif n1_region_group_id!=-1 and n2_region_group_id==-1:
#                             # 说明前面是区域内部链路，后面是区域外部链路，后面的链路需要妥协
#                             adjust_link_nodes_test(i, j, step, nodes, time_2_build, start_ts, end_ts, option=1)
#                         elif n1_region_group_id==-1 and n2_region_group_id !=-1:
#                             # 说明前是其余外部链路，后面是区域内部链路，前面链路需要进行断链为后面准备,后面覆盖前面的
#                             adjust_link_nodes_test(i, j, step, nodes, time_2_build, start_ts, end_ts, option=2)
#
#                         # changed.append((i, j, n1_neighbor, n2_neighbor))
#                 elif not n1.rightneighbor and n2.rightneighbor:
#                     # 说明前面无邻居，后面又邻居，注意，一开始设计的时候，已经优先为区域内考虑了
#                     # 我们事实上，是要查看后面节点的
#               #   elif (not _is_triplet(n1.rightneighbor)) and _is_triplet(n2.rightneighbor):
#               #       if (i, j, step)==(5,32,2519):
#               #           print(2)
#
#                     # 我们要进行回溯，要查询前面的链路，是否存在区域内部链路
#                     n2_neighbor_node = nodes.get((n2.rightneighbor[0], n2.rightneighbor[1], step))
#                     n2_region_group_id = region_in_communication(n2, n2_neighbor_node)
#                    # n1_region_group_id = -1
#
#                     if n2_region_group_id:
#                         # 最简单的，就是后面覆盖前面，前面的要断链路
#                         adjust_link_nodes_test(i, j, step, nodes, time_2_build, start_ts, end_ts, option=0)
#
#                         # for k in range(1,time_2_build+1):
#                         #     bias = step-k
#                         #     if bias<start_ts:
#                         #         break
#                         #     node_rev = nodes.get((i, j, bias))
#                         #     assignlink.assign_Link((i, j, bias), (rx, ry, bias), nodes, 0,
#                         #                            1, k + 1)
#                         #
#                         #     if node_rev is None:
#                         #         print(f"节点缺失: {(i, j, bias)}")
#                         #
#                         #     if node_rev.rightneighbor:
#                         #         if node_rev.rightneighbor == n2.rightneighbor:
#                         #             continue
#                         #         node_rev_neighbor_id =  node_rev.rightneighbor
#                         #         node_rev_neighbor_node = nodes.get((node_rev_neighbor_id[0],node_rev_neighbor_id[1],bias))
#                         #         n1_region_group_id = region_in_communication(node_rev, node_rev_neighbor_node)
#                         #
#                         #
#                         #         # if  n1_region_group_id:
#                         #             # 说明前面有区域内部链路,因此，我们需要开始为这部分进行准备
#                         #
#                         #         if n1_region_group_id == -1 and n2_region_group_id == -1:
#                         #             # 最简单的，就是后面覆盖前面，前面的要断链路
#                         #             adjust_link_nodes_test(i, j, step, nodes, time_2_build, start_ts, end_ts, option=0)
#                         #         elif n1_region_group_id != -1 and n2_region_group_id == -1:
#                         #             # 说明前面是区域内部链路，后面是区域外部链路，后面的链路需要妥协,注意，我们只要妥协bias的即可
#                         #             adjust_link_nodes_test(i, j, bias, nodes, time_2_build, start_ts, end_ts, option=1)
#                         #         elif n1_region_group_id == -1 and n2_region_group_id != -1:
#                         #             # 说明前是其余外部链路，后面是区域内部链路，前面链路需要进行断链为后面准备,后面覆盖前面的
#                         #             adjust_link_nodes_test(i, j, step, nodes, time_2_build, start_ts, end_ts, option=2)
#
#
#     # 前面有neighbor 后面没neighbor，所以，就要考虑左邻居的问题
#                 elif  n1.rightneighbor and not n2.rightneighbor:
#                 # elif _is_triplet(n1.rightneighbor) and (not _is_triplet(n2.rightneighbor)):
#
#                     right_neighbor = n1.rightneighbor
#                     # 接下来，我们要注意了，对于n1.rightneighbor,我们要知道，right neighbor如果切换链路
#                     # 是会影响到当前的n1 以及之前的链路的，所以这个要注意
#                     # 如果是，我们就需要进行调整
#
#                     n1 = nodes.get((right_neighbor[0], right_neighbor[1], step))
#                     n2 = nodes.get((right_neighbor[0], right_neighbor[1], step+1))
#
#
#
#                     if n1.leftneighbor and n2.leftneighbor:
#                         n1_neighbor = (n1.leftneighbor[0], n1.leftneighbor[1])
#                         n2_neighbor = (n2.leftneighbor[0], n2.leftneighbor[1])
#                         if n1_neighbor != n2_neighbor:
#                             # 这里，我们就要判断，此刻step+1的链接和step的链接是否有一方是处于区域内部的链接
#                             # 如果不一致，我们就需要修改
#
#
#                             n1_neighbor_node = nodes.get((n1.leftneighbor[0], n1.leftneighbor[1],step))
#                             n2_neighbor_node = nodes.get((n2.leftneighbor[0], n2.leftneighbor[1],step+1))
#
#                             n1_region_group_id = region_in_communication(n1_neighbor_node,n1)
#                             n2_region_group_id = region_in_communication(n2_neighbor_node,n2)
#                             ##
#                             if n1_region_group_id==-1 and n2_region_group_id==-1:
#                                 # 最简单的，就是后面覆盖前面，前面的要断链路
#                                 adjust_link_nodes_test(i, j, step, nodes, time_2_build,start_ts, end_ts, option=0)
#                             elif n1_region_group_id!=-1 and n2_region_group_id==-1:
#                                 # 说明前面是区域内部链路，后面是区域外部链路，后面的链路需要妥协
#                                 adjust_link_nodes_test(n2.leftneighbor[0], n2.leftneighbor[1], step+1, nodes, time_2_build, start_ts, end_ts, option=1)
#
#
#                             elif n1_region_group_id==-1 and n2_region_group_id !=-1:
#                                 # 说明前是其余外部链路，后面是区域内部链路，前面链路需要进行断链为后面准备,后面覆盖前面的
#                                 adjust_link_nodes_test(i, j, step, nodes, time_2_build, start_ts, end_ts, option=2)
#
#
#     return nodes

def get_no_conflict_link(raw_edges_by_step,offsets,rects,start_ts,end_ts,time_2_build,N,P):

    nodes = {}
    hotspot_keys = set()


    for step in range(start_ts, end_ts):
        hotspot_keys = set()  # <--- 每个 step 单独新建！！
        # 1. 初始化热点区域
        for groupid, rect_tuple in rects.items():
            if not isinstance(rect_tuple, (list, tuple)):
                rect_tuple = (rect_tuple,)
            for rect in rect_tuple:
                if rect is None:
                    continue
                xmin, xmax, ymin, ymax = rect
                for x in range(xmin, xmax + 1):
                    for y in range(ymin, ymax + 1):
                        number = x * N + y
                        modify_number = read_snap_xml.rev_modify_data(step, number, offsets)
                        real_x = modify_number // N
                        real_y = modify_number % N
                        key = (real_x, real_y, step)
                        nodes[key] = tegnode.tegnode_new1(
                            asc_nodes_region_id=groupid,
                            rightneighbor=None,
                            leftneighbor=None,
                            state=-1,
                            importance=0,
                        )
                        hotspot_keys.add(key)
        # 2. 补齐全图节点
        for x in range(P):
            for y in range(N):
                key = (x, y, step)
                if key not in hotspot_keys:
                    nodes[key] = tegnode.tegnode_new1(
                        asc_nodes_region_id=-1,
                        rightneighbor=None,
                        leftneighbor=None,
                        state=-1,
                        importance=0,
                    )
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

                nodes[(x1, y1, step)].rightneighbor = (x2, y2, step)
                nodes[(x2, y2, step)].leftneighbor = (x1, y1, step)

    for step in range(start_ts, end_ts-1):

        for i in range(P - 1):
            for j in range(N):
                # if i==10 and j==32 and step==891:
                #     print(1)
                n1 = nodes.get((i, j, step))
                n2 = nodes.get((i, j, step + 1))
                # 防御式判断
                # 这个表明，某个点在step+1时，其链接改变了，此刻，我们需要直接修改
                if n1.rightneighbor and n2.rightneighbor:
                    n1_neighbor = (n1.rightneighbor[0], n1.rightneighbor[1])
                    n2_neighbor = (n2.rightneighbor[0], n2.rightneighbor[1])
                    if n1_neighbor != n2_neighbor:
                        # 这里，我们就要判断，此刻step+1的链接和step的链接是否有一方是处于区域内部的链接
                        # 如果不一致，我们就需要修改


                        n1_neighbor_node = nodes.get((n1.rightneighbor[0], n1.rightneighbor[1],step))
                        n2_neighbor_node = nodes.get((n2.rightneighbor[0], n2.rightneighbor[1],step))
                        n1_region_group_id = region_in_communication(n1,n1_neighbor_node)
                        n2_region_group_id = region_in_communication(n2,n2_neighbor_node)
                        ##
                        if n1_region_group_id==-1 and n2_region_group_id==-1:
                            # 最简单的，就是后面覆盖前面，前面的要断链路
                            adjust_link_nodes(i, j, step, nodes, time_2_build,start_ts, end_ts, option=0)
                        elif n1_region_group_id!=-1 and n2_region_group_id==-1:
                            # 说明前面是区域内部链路，后面是区域外部链路，后面的链路需要妥协
                            adjust_link_nodes(i, j, step, nodes, time_2_build, start_ts, end_ts, option=1)
                        elif n1_region_group_id==-1 and n2_region_group_id !=-1:
                            # 说明前是其余外部链路，后面是区域内部链路，前面链路需要进行断链为后面准备,后面覆盖前面的
                            adjust_link_nodes(i, j, step, nodes, time_2_build, start_ts, end_ts, option=2)

                        # changed.append((i, j, n1_neighbor, n2_neighbor))
                elif not n1.rightneighbor and n2.rightneighbor:
                    # 我们要进行回溯，要查询前面的链路，是否存在区域内部链路
                    n2_neighbor_node = nodes.get((n2.rightneighbor[0], n2.rightneighbor[1], step))
                    n2_region_group_id = region_in_communication(n2, n2_neighbor_node)
                    n1_region_group_id = -1
                    for k in range(1,time_2_build+1):
                        bias = step-k
                        if bias<start_ts:
                            break
                        node_rev = nodes.get((i, j, bias))

                        if node_rev is None:
                            print(f"节点缺失: {(i, j, bias)}")

                        if node_rev.rightneighbor:
                            if node_rev.rightneighbor == n2.rightneighbor:
                                continue
                            node_rev_neighbor_id =  node_rev.rightneighbor
                            node_rev_neighbor_node = nodes.get((node_rev_neighbor_id[0],node_rev_neighbor_id[1],bias))
                            n1_region_group_id = region_in_communication(node_rev, node_rev_neighbor_node)


                            if  n1_region_group_id:
                                # 说明前面有区域内部链路,因此，我们需要开始为这部分进行准备

                                if n1_region_group_id == -1 and n2_region_group_id == -1:
                                    # 最简单的，就是后面覆盖前面，前面的要断链路
                                    adjust_link_nodes(i, j, step, nodes, time_2_build, start_ts, end_ts, option=0)
                                elif n1_region_group_id != -1 and n2_region_group_id == -1:
                                    # 说明前面是区域内部链路，后面是区域外部链路，后面的链路需要妥协,注意，我们只要妥协bias的即可
                                    adjust_link_nodes(i, j, bias, nodes, time_2_build, start_ts, end_ts, option=1)
                                elif n1_region_group_id == -1 and n2_region_group_id != -1:
                                    # 说明前是其余外部链路，后面是区域内部链路，前面链路需要进行断链为后面准备,后面覆盖前面的
                                    adjust_link_nodes(i, j, step, nodes, time_2_build, start_ts, end_ts, option=2)

                                break
    # 前面有neighbor 后面没neighbor，所以，就要考虑左邻居的问题
                elif  n1.rightneighbor and not n2.rightneighbor:
                    right_neighbor = n1.rightneighbor
                    # 接下来，我们要注意了，对于n1.rightneighbor,我们要知道，right neighbor如果切换链路
                    # 是会影响到当前的n1 以及之前的链路的，所以这个要注意
                    # 如果是，我们就需要进行调整

                    n1 = nodes.get((right_neighbor[0], right_neighbor[1], step))
                    n2 = nodes.get((right_neighbor[0], right_neighbor[1], step+1))



                    if n1.leftneighbor and n2.leftneighbor:
                        n1_neighbor = (n1.leftneighbor[0], n1.leftneighbor[1])
                        n2_neighbor = (n2.leftneighbor[0], n2.leftneighbor[1])
                        if n1_neighbor != n2_neighbor:
                            # 这里，我们就要判断，此刻step+1的链接和step的链接是否有一方是处于区域内部的链接
                            # 如果不一致，我们就需要修改


                            n1_neighbor_node = nodes.get((n1.leftneighbor[0], n1.leftneighbor[1],step))
                            n2_neighbor_node = nodes.get((n2.leftneighbor[0], n2.leftneighbor[1],step+1))

                            n1_region_group_id = region_in_communication(n1_neighbor_node,n1)
                            n2_region_group_id = region_in_communication(n2_neighbor_node,n2)
                            ##
                            if n1_region_group_id==-1 and n2_region_group_id==-1:
                                # 最简单的，就是后面覆盖前面，前面的要断链路
                                adjust_link_nodes(i, j, step, nodes, time_2_build,start_ts, end_ts, option=0)
                            elif n1_region_group_id!=-1 and n2_region_group_id==-1:
                                # 说明前面是区域内部链路，后面是区域外部链路，后面的链路需要妥协
                                adjust_link_nodes(n2.leftneighbor[0], n2.leftneighbor[1], step+1, nodes, time_2_build, start_ts, end_ts, option=1)


                            elif n1_region_group_id==-1 and n2_region_group_id !=-1:
                                # 说明前是其余外部链路，后面是区域内部链路，前面链路需要进行断链为后面准备,后面覆盖前面的
                                adjust_link_nodes(i, j, step, nodes, time_2_build, start_ts, end_ts, option=2)


    edges_by_step = motif.transform_nodes_2_rawedge(nodes, P, N, start_ts, end_ts)

    pendingnodes = {}
    for step in range(start_ts, end_ts-1):

        for i in range(P - 1):
            for j in range(N):
                n1 = nodes.get((i, j, step))
                n2 = nodes.get((i, j, step + 1))
                # 防御式判断
                # 这个表明，某个点在step+1时，其链接改变了，此刻，我们需要直接修改
                if not n1.rightneighbor and n2.rightneighbor:
                    #  说明是建链完成了，因此，我们要反向将建链的链路给加进来
                    right_neighbor = n2.rightneighbor
                    for k in range(time_2_build):
                        bias = step-k
                        if bias<start_ts:
                            break
                        pendingnodes[i, j,bias] = tegnode.tegnode_new1(
                            asc_nodes_region_id=-1,
                            rightneighbor=right_neighbor,
                            leftneighbor=None,
                            state=-1,
                            importance=0,
                        )

    pending_edges = motif.transform_nodes_2_rawedge(pendingnodes, P, N, start_ts, end_ts)

    return edges_by_step,pending_edges



# 仅仅是读取node，不做别的操作
# 我们下面的合并是有问题的，但是呢，我们暂时不去解决，因为影响并不大，以后再回头解决，有问题的，不要用
# def get_no_conflict_link_nodes(nodes, start_ts, end_ts, time_2_build, N, P):
#     nownodes = deepcopy(nodes)
#     for step in range(start_ts, end_ts):
#         for x in range(P):
#             for y in range(N):
#                 key = (x, y, step)
#                 if key not in nownodes:
#                     nownodes[key] = tegnode.tegnode_new(
#                         asc_nodes_region_id=-1,
#                         rightneighbor=None,
#                         leftneighbor=None,
#                         state=-1,
#                         importance=0,
#                     )
#     for step in range(start_ts, end_ts-1):
#         for i in range(P - 1):
#             for j in range(N):
#                 # if i==10 and j==32 and step==891:
#                 #     print(1)
#                 n1 = nownodes.get((i, j, step))
#                 n2 = nownodes.get((i, j, step + 1))
#                 # 防御式判断
#                 if n1.rightneighbor and n2.rightneighbor:
#                     n1_neighbor = (n1.rightneighbor[0], n1.rightneighbor[1])
#                     n2_neighbor = (n2.rightneighbor[0], n2.rightneighbor[1])
#                     if n1_neighbor != n2_neighbor:
#
#                         adjust_link_nodes(i, j, step, nownodes, time_2_build, start_ts, end_ts, option=0)
#
#                 elif not n1.rightneighbor and n2.rightneighbor:
#                     # 我们要进行回溯，要查询前面的链路，是否存在区域内部链路
#                     n2_neighbor_node = nownodes.get((n2.rightneighbor[0], n2.rightneighbor[1], step))
#                   #  n2_region_group_id = region_in_communication(n2, n2_neighbor_node)
#                     n1_region_group_id = -1
#                     for k in range(1,time_2_build+1):
#                         bias = step-k
#                         if bias<start_ts:
#                             break
#                         node_rev = nownodes.get((i, j, bias))
#
#                         if node_rev is None:
#                             print(f"节点缺失: {(i, j, bias)}")
#
#                         if node_rev.rightneighbor:
#                             if node_rev.rightneighbor == n2.rightneighbor:
#                                 continue
#
#                             # n1_region_group_id = region_in_communication(node_rev, node_rev_neighbor_node)
#                             adjust_link_nodes(i, j, step, nownodes, time_2_build, start_ts, end_ts, option=0)
#                             break
#
#     # 前面有neighbor 后面没neighbor，所以，就要考虑左邻居的问题
#                 elif  n1.rightneighbor and not n2.rightneighbor:
#                     right_neighbor = n1.rightneighbor
#                     # 接下来，我们要注意了，对于n1.rightneighbor,我们要知道，right neighbor如果切换链路
#                     # 是会影响到当前的n1 以及之前的链路的，所以这个要注意
#                     # 如果是，我们就需要进行调整
#
#                     n1 = nownodes.get((right_neighbor[0], right_neighbor[1], step))
#                     n2 = nownodes.get((right_neighbor[0], right_neighbor[1], step+1))
#
#                     if n1.leftneighbor and n2.leftneighbor:
#                         n1_neighbor = (n1.leftneighbor[0], n1.leftneighbor[1])
#                         n2_neighbor = (n2.leftneighbor[0], n2.leftneighbor[1])
#                         if n1_neighbor != n2_neighbor:
#
#                             adjust_link_nodes(i, j, step, nownodes, time_2_build, start_ts, end_ts, option=0)
#
#     edges_by_step = motif.transform_nodes_2_rawedge(nownodes, P, N, start_ts, end_ts)
#
#     pendingnodes = {}
#     for step in range(start_ts, end_ts-1):
#         for i in range(P - 1):
#             for j in range(N):
#                 n1 = nownodes.get((i, j, step))
#                 n2 = nownodes.get((i, j, step + 1))
#                 # 防御式判断
#                 # 这个表明，某个点在step+1时，其链接改变了，此刻，我们需要直接修改
#                 if not n1.rightneighbor and n2.rightneighbor:
#                     #  说明是建链完成了，因此，我们要反向将建链的链路给加进来
#                     right_neighbor = n2.rightneighbor
#                     for k in range(time_2_build):
#                         bias = step-k
#                         if bias<start_ts:
#                             break
#                         pendingnodes[i, j,bias] = tegnode.tegnode_new(
#                             asc_nodes_region_id=-1,
#                             rightneighbor=right_neighbor,
#                             leftneighbor=None,
#                             state=-1,
#                             importance=0,
#                         )
#
#     pending_edges = motif.transform_nodes_2_rawedge(pendingnodes, P, N, start_ts, end_ts)
#
#     return edges_by_step,pending_edges
#
#
#
#
# # 我们下面的合并是有问题的，但是呢，我们暂时不去解决，因为影响并不大，以后再回头解决
# def get_no_conflict_link_nodes2(nodes: dict[tuple[int, int, int], tegnode.tegnode_complete], start_ts, end_ts, time_2_build, N, P):
#
#
#
#     nownodes = deepcopy(nodes)
#     for step in range(start_ts, end_ts):
#         for x in range(P):
#             for y in range(N):
#                 key = (x, y, step)
#                 if key not in nownodes:
#                     nownodes[key] = tegnode.tegnode_complete(
#                         asc_nodes_region_id=-1,
#                         rightneighbor=None,
#                         leftneighbor=None,
#                         left_state=-1,
#                         right_state=-1,
#                     )
#
#
#     for step in range(start_ts, end_ts-1):
#         for i in range(P - 1):
#             for j in range(N):
#                 # if i==10 and j==32 and step==891:
#                 #     print(1)
#                 n1 = nownodes.get((i, j, step))
#                 n2 = nownodes.get((i, j, step + 1))
#                 # 防御式判断
#                 if n1.rightneighbor and n2.rightneighbor:
#                     n1_neighbor = (n1.rightneighbor[0], n1.rightneighbor[1])
#                     n2_neighbor = (n2.rightneighbor[0], n2.rightneighbor[1])
#                     if n1_neighbor != n2_neighbor:
#
#                         adjust_link_nodes(i, j, step, nownodes, time_2_build, start_ts, end_ts, option=0)
#
#                 elif not n1.rightneighbor and n2.rightneighbor:
#                     # 我们要进行回溯，要查询前面的链路，是否存在区域内部链路
#                     n2_neighbor_node = nownodes.get((n2.rightneighbor[0], n2.rightneighbor[1], step))
#                   #  n2_region_group_id = region_in_communication(n2, n2_neighbor_node)
#                     n1_region_group_id = -1
#                     for k in range(1,time_2_build+1):
#                         bias = step-k
#                         if bias<start_ts:
#                             break
#                         node_rev = nownodes.get((i, j, bias))
#
#                         if node_rev is None:
#                             print(f"节点缺失: {(i, j, bias)}")
#
#                         if node_rev.rightneighbor:
#                             if node_rev.rightneighbor == n2.rightneighbor:
#                                 continue
#
#                             # n1_region_group_id = region_in_communication(node_rev, node_rev_neighbor_node)
#                             adjust_link_nodes(i, j, step, nownodes, time_2_build, start_ts, end_ts, option=0)
#                             break
#
#     # 前面有neighbor 后面没neighbor，所以，就要考虑左邻居的问题
#                 elif  n1.rightneighbor and not n2.rightneighbor:
#                     right_neighbor = n1.rightneighbor
#                     # 接下来，我们要注意了，对于n1.rightneighbor,我们要知道，right neighbor如果切换链路
#                     # 是会影响到当前的n1 以及之前的链路的，所以这个要注意
#                     # 如果是，我们就需要进行调整
#
#                     n1 = nownodes.get((right_neighbor[0], right_neighbor[1], step))
#                     n2 = nownodes.get((right_neighbor[0], right_neighbor[1], step+1))
#
#                     if n1.leftneighbor and n2.leftneighbor:
#                         n1_neighbor = (n1.leftneighbor[0], n1.leftneighbor[1])
#                         n2_neighbor = (n2.leftneighbor[0], n2.leftneighbor[1])
#                         if n1_neighbor != n2_neighbor:
#
#                             adjust_link_nodes(i, j, step, nownodes, time_2_build, start_ts, end_ts, option=0)
#
#
#     # 接下来就要考虑断代的事情，这个其实很简单的，就是我们查看谁覆盖谁的问题
#
#     for step in range(start_ts, end_ts-1):
#         for i in range(P - 1):
#             for j in range(N):
#                 n1 = nownodes.get((i, j, step))
#
#                 n2 = nownodes.get((i, j, step + 1))
#                 if n1.rightneighbor and not n2.rightneighbor and not n2.leftneighbor:
#                     right_neighbor = n1.rightneighbor
#                     offset = 1
#                     while(1):
#                         time = step+offset
#                         if time >= end_ts:
#                             break
#                         nodes_next = nownodes.get((i, j, time))
#
#
#
#                         if  nodes_next.leftneighbor or  nodes_next.rightneighbor:
#                             break
#
#                         n1_neighbor_next_node = nownodes.get((right_neighbor[0], right_neighbor[1], time))
#
#                         if  n1_neighbor_next_node.leftneighbor or  n1_neighbor_next_node.rightneighbor:
#                             break
#
#
#                         offset = offset + 1
#
#                     baias = offset-time_2_build
#                     for k in range(1,baias):
#                         time = step+k
#                         nodes_next = nownodes.get((i, j, time))
#                         nodes_next.rightneighbor =(right_neighbor[0],right_neighbor[1],time)
#                         n1_neighbor_next_node=nownodes.get((right_neighbor[0],right_neighbor[1],time))
#                         n1_neighbor_next_node.leftneighbor = (i,j,time)
#
#
#
#
#
#
#
#
#
#
#     edges_by_step = motif.transform_nodes_2_rawedge(nownodes, P, N, start_ts, end_ts)
#
#     pendingnodes = {}
#     for step in range(start_ts, end_ts-1):
#         for i in range(P - 1):
#             for j in range(N):
#                 n1 = nownodes.get((i, j, step))
#                 n2 = nownodes.get((i, j, step + 1))
#                 # 防御式判断
#                 # 这个表明，某个点在step+1时，其链接改变了，此刻，我们需要直接修改
#                 if not n1.rightneighbor and n2.rightneighbor:
#                     #  说明是建链完成了，因此，我们要反向将建链的链路给加进来
#                     right_neighbor = n2.rightneighbor
#                     for k in range(time_2_build):
#                         bias = step-k
#                         if bias<start_ts:
#                             break
#                         pendingnodes[i, j,bias] = tegnode.tegnode_new(
#                             asc_nodes_region_id=-1,
#                             rightneighbor=right_neighbor,
#                             leftneighbor=None,
#                             state=-1,
#                             importance=0,
#                         )
#
#     pending_edges = motif.transform_nodes_2_rawedge(pendingnodes, P, N, start_ts, end_ts)
#
#     return edges_by_step,pending_edges
#
#
#

from typing import Dict, Tuple

#
import traceback

# 想重点观察的坐标（可加多个）

class NodeProxy:
    __slots__ = ("_obj", "_key")
    def __init__(self, obj, key):
        object.__setattr__(self, "_obj", obj)   # 真正的节点对象
        object.__setattr__(self, "_key", key)   # (i,j,t)

    # 读取转发
    def __getattr__(self, name):
        return getattr(self._obj, name)

    # 写入拦截（只关注左右邻）
    def __setattr__(self, name, value):
        if name in ("rightneighbor", "leftneighbor"):
            old = getattr(self._obj, name, None)
            if old != value:
                print(f"[WRITE] {self._key}.{name}: {old} -> {value}")
                # 打印调用栈最后几层，定位是哪个分支/函数写的
                for f in traceback.format_stack(limit=6):
                    print("   ", f.strip())
        setattr(self._obj, name, value)



def get_no_conflict_link_nodes3(
    nodes: Dict[Tuple[int, int, int], tegnode.tegnode_complete],
    start_ts: int, end_ts: int, time_2_build: int, N: int, P: int
):
    """
    优化版：
    - 不再 deepcopy + 预填所有节点，而是“按需创建”缺失节点（lazy ensure）
    - 大量局部绑定与一次遍历，减少 Python 层开销
    - 语义与原逻辑一致
    """
    # ---- 局部绑定，减少查找开销 ----
    TC = tegnode.tegnode_complete
    TN = tegnode.tegnode_new
    get = nodes.get

    # 工作字典：浅拷贝映射即可（不复制对象），按需创建新节点
    nownodes: Dict[Tuple[int, int, int], tegnode.tegnode_complete] = dict(nodes)
   # nownodes = deepcopy(nodes)
    nget = nownodes.get
    # WATCH = {(15, 26, 1203)}  # 也可以加  (15,26,1204)、(15,26,1233) 等

    # 按需创建的默认节点（每次必须新建实例，不能复用同一个）
    def _mk_empty():
        return TC(
            asc_nodes_region_id=-1,
            rightneighbor=None,
            leftneighbor=None,
            left_state=-1,
            right_state=-1,
        )

    # 确保 (i,j,t) 存在，若无则创建
    def ensure(i: int, j: int, t: int):
        k = (i, j, t)
        n = nget(k)
        if n is None:
            n = _mk_empty()
            nownodes[k] = n
        # 关键：若在观察名单内且还不是代理，则包一层
        # if k in WATCH and not isinstance(n, NodeProxy):
        #     n = NodeProxy(n, k)
        #     nownodes[k] = n
        return n
    # print(nownodes[15,26,1203])
    # ============== 第一轮：处理“切换/建链/断链”的冲突与回溯 ==============
    # 仅遍历必要范围（end_ts-1，因为我们总是看 t 与 t+1）
    e1 = end_ts - 1
    for step in range(start_ts, e1):
        # 只处理 i ∈ [0, P-2]（与你原代码一致）
        # actually，我们应该考虑的是以时间片为层级的
        # if step==1233:
        #     print(1)
        for i in range(P - 1):
            for j in range(N):
                # if (i,j)==(15,26):
                #     print(1)
                n1 = ensure(i, j, step)
                n2 = ensure(i, j, step + 1)

                rn1 = n1.rightneighbor
                rn2 = n2.rightneighbor

                # 情况 A：两步都有 rightneighbor，但目标不同 -> 触发调整
                if rn1 and rn2:
                    if (rn1[0], rn1[1]) != (rn2[0], rn2[1]):
                        adjust_link_nodes(i, j, step, nownodes, time_2_build, start_ts, end_ts, option=0)

                # 情况 B：前一步没有、后一步有 -> 需要回溯检查与修正
                elif (not rn1) and rn2:
                    # 先保证“对方节点在 step 的快照”存在
                    _ = ensure(rn2[0], rn2[1], step)

                    # 回溯 time_2_build 帧
                    for k in range(1, time_2_build + 1):
                        bias = step - k
                        if bias < start_ts:
                            break

                        node_rev = ensure(i, j, bias)
                        node_neighbor_rev = ensure(rn2[0], rn2[1], bias)

                        # 如果历史上自己在某帧已经有 rightneighbor 且不是 rn2，则认为冲突，触发调整
                        if node_rev.rightneighbor:
                            if node_rev.rightneighbor == rn2:
                                continue
                            adjust_link_nodes(i, j, step, nownodes, time_2_build, start_ts, end_ts, option=0)
                            break

                        # 如果对方在某帧有 leftneighbor，则把那条旧左邻断开
                        ln = node_neighbor_rev.leftneighbor
                        # 只在“对端左邻 == 我这条 (i,j,bias)”时，才允许拆
                        if ln and ln[0] == i and ln[1] == j and ln[2] == bias:
                            ln_node = ensure(i, j, bias)

                            # 可选：再加一道保险——仅当我这边的 rightneighbor 的确不是目标 rn2 时才清
                            if ln_node.rightneighbor and (ln_node.rightneighbor[0], ln_node.rightneighbor[1]) != (
                                    rn2[0], rn2[1]):
                                ln_node.rightneighbor = None

                            node_neighbor_rev.leftneighbor = None

                        # ln = node_neighbor_rev.leftneighbor
                        # if ln:
                        #     ln_node = ensure(ln[0], ln[1], bias)
                        #     ln_node.rightneighbor = None
                        #     node_neighbor_rev.leftneighbor = None

                # 情况 C：前一步有、后一步没有 -> 看对方左邻在 t/t+1 是否切换，若切则调整
                elif rn1 and (not rn2):
                    rx, ry = rn1[0], rn1[1]
                    nb1 = ensure(rx, ry, step)
                    nb2 = ensure(rx, ry, step + 1)

                    ln1 = nb1.leftneighbor
                    ln2 = nb2.leftneighbor
                    if ln1 and ln2:
                        if (ln1[0], ln1[1]) != (ln2[0], ln2[1]):
                            adjust_link_nodes(i, j, step, nownodes, time_2_build, start_ts, end_ts, option=0)

    # print(nownodes[15,26,1203])

    # ============== 第二轮：“断代”覆盖处理（把空白期回填为同一 rightneighbor） ==============
    for step in range(start_ts, e1):
        for i in range(P - 1):
            for j in range(N):
                n1 = ensure(i, j, step)
                n2 = ensure(i, j, step + 1)

                rn1 = n1.rightneighbor
                if rn1 and (not n2.rightneighbor) and (not n2.leftneighbor):
                    rx, ry = rn1[0], rn1[1]
                    # 向后找“下一个非空时刻”或结束
                    offset = 1
                    while True:
                        t = step + offset
                        if t >= end_ts:
                            break
                        node_next = ensure(i, j, t)
                        if node_next.leftneighbor or node_next.rightneighbor:
                            break

                        neigh_next = ensure(rx, ry, t)
                        if neigh_next.leftneighbor or neigh_next.rightneighbor:
                            break

                        offset += 1

                    # 回填区间：(step, step+baias) 使其都连向 (rx,ry,t)
                    baias = offset - time_2_build
                    for k in range(1, baias):
                        t = step + k
                        node_next = ensure(i, j, t)
                        node_next.rightneighbor = (rx, ry, t)

                        neigh_next = ensure(rx, ry, t)
                        neigh_next.leftneighbor = (i, j, t)

    # ============== 导出边（已建链） ==============
    edges_by_step = motif.transform_nodes_2_rawedge(nownodes, P, N, start_ts, end_ts)

    # ============== 生成 pendingnodes 并导出待建边 ==============
    pendingnodes: Dict[Tuple[int, int, int], tegnode.tegnode_new] = {}
    pset = pendingnodes  # 局部别名，少写字典名
    for step in range(start_ts, e1):
        for i in range(P - 1):
            for j in range(N):
                n1 = ensure(i, j, step)
                n2 = ensure(i, j, step + 1)
                # 只有在“前无后有”时，把过去 time_2_build 帧回填为 pending
                if (not n1.rightneighbor) and n2.rightneighbor:
                    rn = n2.rightneighbor
                    rx, ry = rn[0], rn[1]
                    for k in range(time_2_build):
                        bias = step - k
                        if bias < start_ts:
                            break
                        pset[(i, j, bias)] = TN(
                            asc_nodes_region_id=-1,
                            rightneighbor=(rx, ry, bias),
                            leftneighbor=None,
                            state=-1,
                            importance=0,
                        )

    pending_edges = motif.transform_nodes_2_rawedge(pendingnodes, P, N, start_ts, end_ts)
    return edges_by_step, pending_edges, nownodes

# def assign_Link(start,end,nodes,state,type,timelast):
#     if nodes[start].rightneighbor:
#         rightneighbor = nodes[start].rightneighbor
#         xr = rightneighbor[0]
#         yr = rightneighbor[1]
#         zr = rightneighbor[2]
#         nodes[(xr,yr,zr)].leftneighbor = -1
#         nodes[(xr, yr, zr)].left_state = -1
#     if nodes[end].leftneighbor:
#         leftneighbor = nodes[end].leftneighbor
#         xl = leftneighbor[0]
#         yl = leftneighbor[1]
#         zl = leftneighbor[2]
#         nodes[(xl,yl,zl)].rightneighbor = -1
#         nodes[(xl,yl,zl)].right_state = -1
#         nodes[(xl,yl,zl)].type = -1
#         nodes[(xl, yl, zl)].timelast = -1
#     nodes[start].rightneighbor = end
#     nodes[start].right_state = state
#     nodes[start].type = type
#     nodes[start].timelast = timelast
#     nodes[end].leftneighbor = start
#     nodes[end].left_state = state
#
# UNSET = -1
# def _is_triplet(val) -> bool:
#     return isinstance(val, tuple) and len(val) == 3
#
# def _ensure(nodes, key):
#     node = nodes.get(key)
#     if node is None:
#         node = tegnode.tegnode_new(
#                             asc_nodes_region_id=-1,
#                             rightneighbor=-1,
#                             leftneighbor=-1,
#                             right_state=-1,
#             left_state=-1,node_type=-1,
#
#                             timelast=-1,
#                         )    # 你的类默认字段均为 -1
#         nodes[key] = node
#     return node
#
#
# def assign_Link(start, end, nodes, state, type, timelast):
#     """
#     把 start 的 rightneighbor 连接到 end，并保持双向一致。
#     缺节点时自动创建默认节点；未设置邻居(-1/None)时不做清理。
#     """
#
#     # 0) 确保两端节点存在
#     s = _ensure(nodes, start)
#     e = _ensure(nodes, end)
#
#     # 如果本来就连的是同一端，只更新状态即可
#     if _is_triplet(s.rightneighbor) and s.rightneighbor == end:
#         s.right_state = state
#         s.type = type
#         s.timelast = timelast
#         e.leftneighbor = start
#         e.left_state = state
#         return
#
#     # 1) 断开 start 原来的右邻（若存在且确实指回 start）
#     rn = s.rightneighbor
#     if _is_triplet(rn):
#         rn_node = nodes.get(rn)
#         if rn_node and rn_node.leftneighbor == start:
#             rn_node.leftneighbor = -1
#             rn_node.left_state   = -1
#             # 如需一并清理，可按需解除注释：
#             # rn_node.type     = UNSET
#             # rn_node.timelast = UNSET
#
#     # 2) 断开 end 原来的左邻（若存在且确实指向 end）
#     ln = e.leftneighbor
#     if _is_triplet(ln):
#         ln_node = nodes.get(ln)
#         if ln_node and ln_node.rightneighbor == end:
#             ln_node.rightneighbor = UNSET
#             ln_node.right_state   = UNSET
#             ln_node.node_type          = UNSET
#             ln_node.timelast      = UNSET
#
#     # 3) 建立 start→end 与 end←start
#     s.rightneighbor = end
#     s.right_state   = state
#     s.node_type          = type
#     s.timelast      = timelast
#
#     e.leftneighbor  = start
#     e.left_state    = state
from itertools import groupby

def group_by_y_then_x(triples, dedup=True):
    """
    triples: [(x, y, z), ...]
    返回：[(y, [(x, [(x,y,z)...]), ...]), ...]，y 组按降序，组内 x 组按升序
    """
    data = set(triples) if dedup else list(triples)
    # 排序：确保 groupby 连续分组 —— y 降序，其次 x 升序，最后 z 升序
    ordered = sorted(data, key=lambda t: (-t[1], t[0], t[2]))

    out = []
    for y, items_y in groupby(ordered, key=itemgetter(1)):
        items_y = list(items_y)                 # 此时已按 x 升序
        x_groups = []
        for x, items_x in groupby(items_y, key=itemgetter(0)):
            x_groups.append((x, list(items_x))) # 同一 (x,y) 的所有 (x,y,z)
        out.append((y, x_groups))
    return out

def flatten_groups(groups):
    """把上面的分组结构扁平化为排序后的列表"""
    return [item for _, xgs in groups for _, items in xgs for item in items]


from operator import itemgetter
import  draw.pymatlab2.basic.assignlink as assignlink

def get_no_conflict_link_nodes4(
    nodes: Dict[Tuple[int, int, int], tegnode.tegnode_new],
    start_ts: int, end_ts: int, time_2_build: int, N: int, P: int,ratio,ig_endtime,flag
):
    """
    优化版：
    - 不再 deepcopy + 预填所有节点，而是“按需创建”缺失节点（lazy ensure）
    - 大量局部绑定与一次遍历，减少 Python 层开销
    - 语义与原逻辑一致
    """
    # ---- 局部绑定，减少查找开销 ----
    TC = tegnode.tegnode_new




    # # 工作字典：浅拷贝映射即可（不复制对象），按需创建新节点
    # nownodes: Dict[Tuple[int, int, int], tegnode.tegnode_new] = dict(nodes)
    # 原来：nownodes = dict(nodes)  # 仍共享对象，改了会污染 base
    if flag==1:
        nownodes: Dict[Tuple[int, int, int], tegnode.tegnode_new] = dict(nodes)
    else:
        nownodes = COWNodes(nodes)      # ✅ 改为写时拷贝视图
   # nownodes = deepcopy(nodes)

    nget = nownodes.get
    # WATCH = {(15, 26, 1203)}  # 也可以加  (15,26,1204)、(15,26,1233) 等

    # 按需创建的默认节点（每次必须新建实例，不能复用同一个）
    def _mk_empty():
        return TC(
            asc_nodes_region_id=-1,
            rightneighbor=None,
            leftneighbor=None,
            left_state=-1,
            right_state=-1,
        )

    # 确保 (i,j,t) 存在，若无则创建
    def ensure(i: int, j: int, t: int):
        k = (i, j, t)
        n = nget(k)
        if n is None:
            n = _mk_empty()
            nownodes[k] = n
        # 关键：若在观察名单内且还不是代理，则包一层
        # if k in WATCH and not isinstance(n, NodeProxy):
        #     n = NodeProxy(n, k)
        #     nownodes[k] = n
        return n
    # print(nownodes[15,26,1203])
    # ============== 第一轮：处理“切换/建链/断链”的冲突与回溯 ==============
    # 仅遍历必要范围（end_ts-1，因为我们总是看 t 与 t+1）
    # e1 = end_ts - 1
    if flag==1:
        test_nodes = copy.copy(nownodes)
    else:
        test_nodes = COWNodes(nownodes)  # ✅ 再叠一层，专门给本次批处理试验

    # for i in range(P - 1):
    #     for j in range(N):
    #         n1 = ensure(i, j, ig_endtime)
    #         if n1.right_state==0:
    #             neighbor=n1.rightneighbor
    #             rx=neighbor[0]
    #             ry=neighbor[1]
    #             for k in range(ig_endtime,ig_endtime+time_2_build):
    #                 assignlink.assign_Link((i,j,k), (rx,ry,k), test_nodes, 1, -1, 0)

   #

    step = ig_endtime-1

    # for step in range(start_ts, e1):
        # 只处理 i ∈ [0, P-2]（与你原代码一致）
        # actually，我们应该考虑的是以时间片为层级的
        # if step==1233:
        #     print(1)
    change_link_terminal = []
    mantan_link_terminal = []

    delete_link_terminal = []
    for i in range(P - 1):
        for j in range(N):
            if (i,j)==(5,33):
                print(1)
            n1 = ensure(i, j, step)
            n2 = ensure(i, j, step + 1)

            rn1 = n1.rightneighbor
            rn2 = n2.rightneighbor

            ln1 = n1.leftneighbor
            ln2 = n2.leftneighbor

            if  n2.right_state == 0:
                continue



            # 情况 A：两步都有 rightneighbor，但目标不同 -> 触发调整
            if rn1 and rn2:

                if (rn1[0], rn1[1]) != (rn2[0], rn2[1])  :
                    change_link_terminal.append((i, j, step))
                    # continue
               #     print((i, j, step))
                # 如果是相等的，但是状态不一样，实际上这个时候完全不需要简练，直接将相关点进行覆盖即可
                elif ((rn1[0], rn1[1]) == (rn2[0], rn2[1])) and (n1.right_state==1) and (n2.right_state==0):
                    mantan_link_terminal.append((i, j, step))

            # 如果前面没有，后面有，很显然要调整
            elif not rn1 and rn2:
                change_link_terminal.append((i, j, step))
            elif rn1 and not rn2:
                delete_link_terminal.append((i, j, step))

            # if ln1 and ln2:
            #     if (ln1[0], ln1[1]) != (ln2[0], ln2[1]):
            #         change_link_terminal.append((ln2[0], ln2[1], step))

               #     print((ln2[0], ln2[1], step))
                  #  adjust_link_nodes(i, j, step, nownodes, time_2_build, start_ts, end_ts, option=0)
    # here we get the change link terminal groups,next stage ,we
    # we begin do our algorithm2 steps


    #by_y = sorted(change_link_terminal, key=itemgetter(1), reverse=True)
    groups = group_by_y_then_x(set(change_link_terminal), dedup=True)

    by_y = flatten_groups(groups)

    endtiime = step
    n = len(by_y)
    if n == 0:
        pass
    else:
        chunk_size = max(int(n * ratio), 1)

        for batch_idx, start in enumerate(range(0, n, chunk_size), start=1):
            batch = by_y[start: start + chunk_size]
            print(f"# batch {batch_idx} (items {start}..{start + len(batch) - 1})")
            # firstly ,we the setup   start
            # print(batch)


            setup_start = endtiime-(batch_idx)*time_2_build+1
            work_start = setup_start+time_2_build

            for item in batch:
                x = item[0]
                y = item[1]

                future_neighbor = test_nodes[x, y, endtiime + 1].rightneighbor
                future_x = future_neighbor[0]
                future_y = future_neighbor[1]

                # firstly ,we setup the setup period
                for k in range(time_2_build):
                    setup_time_index =setup_start+k
                    assignlink.assign_Link((x,y,setup_time_index), (future_x,future_y,setup_time_index), test_nodes, 0, 0, time_2_build-k)

                # then ,we arrage the working period

                for k in range(work_start,endtiime+1):
                    assignlink.assign_Link((x,y,k), (future_x,future_y,k), test_nodes, 1, 0, 0)


    edges_by_step,pending_edge,iG_edge = motif.transform_nodes_2_rawedge_test(test_nodes, P, N, start_ts, end_ts)

    return edges_by_step,pending_edge,iG_edge,by_y








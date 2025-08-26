from copy import deepcopy
import genaric2.tegnode as tegnode



# def get_no_conflict_link(raw_edges_by_step,start_ts,end_ts,time_2_build,N,P):
# ### then ,we need arange the link reconfiguration,that means,we need arrrange the restablish limitation for the
#
#     ### then ,we need arange the link reconfiguration,that means,we need arrrange the restablish limitation for the
#     nodes = {}
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
#                 src = i * N + j
#                 # 标记建链区间内该链路为“pending”
#                 for k in range(newtime, step):
#                     if k not in pending_links_by_step:
#                         pending_links_by_step[k] = {}
#                     if src not in pending_links_by_step[k]:
#                         pending_links_by_step[k][src] = set()
#                     pending_links_by_step[k][src].add(new_dst_id)
#
#     for step, changes in allchange.items():
#         print(f"{step}:")
#         newtime = step - time_2_build
#         if newtime < start_ts:
#             continue
#         for i, j, old, new in changes:
#             for k in range(newtime, step):
#                 src = i * N + j
#                 # 遍历当前时间k下src的所有目标（dsts 是个 set）
#                 dsts = raw_edges_by_step[k].get(src, set())
#                 # 生成需要删除的dst列表（横向链路，即目的节点横坐标和src不一样）
#                 to_remove = [dst for dst in dsts if dst // N != i]
#                 # 遍历删除
#                 for dst in to_remove:
#                     raw_edges_by_step[k][src].remove(dst)
#                     # 如果是 set()，用 discard(dst) 更安全（不存在不会报错）
#                     # raw_edges_by_step[k][src].discard(dst)
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




import draw.read_snap_xml as read_snap_xml

import draw.basic_functio.motif as motif

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
                        nodes[key] = tegnode.tegnode_new(
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
                    nodes[key] = tegnode.tegnode_new(
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
                        pendingnodes[i, j,bias] = tegnode.tegnode_new(
                            asc_nodes_region_id=-1,
                            rightneighbor=right_neighbor,
                            leftneighbor=None,
                            state=-1,
                            importance=0,
                        )

    pending_edges = motif.transform_nodes_2_rawedge(pendingnodes, P, N, start_ts, end_ts)

    return edges_by_step,pending_edges

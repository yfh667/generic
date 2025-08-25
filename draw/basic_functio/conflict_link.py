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
def region_edge_state_special(rects, N, id1, id2, id3):
    """
    1. id1 不在任何区域，返回0
    2. id1 在某块区域里，如果 (id1, id2) 在这块、(id1, id3) 不在这块，返回1
    3. 其它情况返回2
    """
    def nodeid_to_xy(nodeid, N):
        return nodeid // N, nodeid % N

    def in_rect(x, y, rect):
        if rect is None:
            return False
        xmin, xmax, ymin, ymax = rect
        return xmin <= x <= xmax and ymin <= y <= ymax

    x1, y1 = nodeid_to_xy(id1, N)
    x2, y2 = nodeid_to_xy(id2, N)
    x3, y3 = nodeid_to_xy(id3, N)

    for groupid, rect_tuple in rects.items():
        if not isinstance(rect_tuple, (list, tuple)):
            rect_tuple = (rect_tuple,)
        for rect in rect_tuple:
            if rect is None:
                continue
            if in_rect(x1, y1, rect):
                # id1 在这块，才继续
                in_pair12 = in_rect(x2, y2, rect)
                in_pair13 = in_rect(x3, y3, rect)
                if in_pair12 and not in_pair13:
                    return 1
                else:
                    return 2
    # id1 不在任何区域
    return 0


import draw.read_snap_xml as read_snap_xml
def get_no_conflict_link(raw_edges_by_step,group_data,offsets,rects,start_ts,end_ts,time_2_build,N,P):
### then ,we need arange the link reconfiguration,that means,we need arrrange the restablish limitation for the

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
    newnodes =deepcopy(nodes)

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

                # 这个表明，某个点在step+1时，其链接改变了，此刻，我们需要直接修改
                if n1.rightneighbor and n2.rightneighbor:
                    n1_neighbor = (n1.rightneighbor[0], n1.rightneighbor[1])
                    n2_neighbor = (n2.rightneighbor[0], n2.rightneighbor[1])
                    if n1_neighbor != n2_neighbor:
                        # 这里，我们就要判断，此刻step+1的链接和step的链接是否有一方是处于区域内部的链接
                        # 如果不一致，我们就需要修改
                        nownode_id = i*N+j
                        nowtime_rightneighbor= n1_neighbor[0]*N+n1_neighbor[1]
                        nexttime_rightneighbor= n2_neighbor[0]*N+n2_neighbor[1]

                        # we all need check whether the link for nownode_id-nowtime_rightneighbor and nownode_id-nexttime_rightneighbor
                        # is in the same rec group
                        #attention ,rect group is more than  group, it is a rectangular group ,it is more

                        modify_nownode_id =  read_snap_xml.modify_data(step, nownode_id, offsets)
                        modify_nowtime_rightneighbor=  read_snap_xml.modify_data(step, nowtime_rightneighbor, offsets)
                        modify_nexttime_rightneighbor= read_snap_xml.modify_data(step, nexttime_rightneighbor, offsets)

                        flag =  region_edge_state_allgroups(rects, N, modify_nownode_id, modify_nowtime_rightneighbor, modify_nexttime_rightneighbor)

                        if flag==0:









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


    return raw_edges_by_step,pending_links_by_step
    # 至此，我们完成了拆链建链动作了。
    # 但是由于我们演示是要用变化后的图去演示的，所以，在这里，我们又重新把变换后的图变换出来，这样我们看图就会看的更加清楚。
    #
    # modify_edges_by_step = {}
    #
    # for step, edges in raw_edges_by_step.items():  # 一定要遍历raw_edges_by_step！
    #     modify_edges_by_step[step] = {}
    #
    #     for src, dsts in edges.items():
    #         modify_src = read_snap_xml.modify_data(step, src, off_sets)
    #         for dst in dsts:
    #             modify_dst = read_snap_xml.modify_data(step, dst, off_sets)
    #             modify_edges_by_step[step].setdefault(modify_src, set()).add(modify_dst)

    # xml_file = "E:\\Data\\test.xml"
    #
    # adjacent2xml.write_steps_to_xml(modify_edges_by_step, xml_file)

    # modify_pending_links_by_step = {}
    # for step, src_to_dsts in pending_links_by_step.items():
    #     for src, dsts in src_to_dsts.items():  # dsts是个集合
    #         # 只取集合中的第一个（如果你确认每个集合只有一个dst的话）
    #         dst = next(iter(dsts))
    #
    #         modify_src = read_snap_xml.modify_data(step, src, off_sets)
    #         modify_dst = read_snap_xml.modify_data(step, dst, off_sets)
    #
    #         # 先初始化
    #         if step not in modify_pending_links_by_step:
    #             modify_pending_links_by_step[step] = {}
    #         modify_pending_links_by_step[step].setdefault(modify_src, set()).add(modify_dst)

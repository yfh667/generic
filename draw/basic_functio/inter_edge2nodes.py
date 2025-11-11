import genaric2.tegnode as tegnode

def trans_edge2node(raw_inter_edges_by_step,P,N):
    nodes = {}
    for step, edges in raw_inter_edges_by_step.items():
        for src, dsts in edges.items():
            x1 = src // N
            y1 = src % N
            if x1==1 and y1==0 and step==3:
                print(1)
            if not dsts:
                continue  # 跳过空集合，防止 StopIteration
            dst = next(iter(dsts))
            d_x=dst//N
            d_y=dst%N
            if (x1, y1,  step) not in nodes:
                nodes[(x1, y1,  step)] = tegnode.tegnode(
                    asc_nodes_flag=False,
                    rightneighbor=(d_x, d_y, step),
                    leftneighbor=None,
                    state=-1,
                    importance=0,
                )
            else:
                nodes[ (x1, y1,  step)].rightneighbor = (d_x, d_y, step)

            # 处理左邻居
            if (d_x, d_y, step) not in nodes:
                nodes[ (d_x, d_y, step)] = tegnode.tegnode(
                    asc_nodes_flag=False,
                    rightneighbor=None,
                    leftneighbor=(x1, y1,  step),
                    state=-1,
                    importance=0,
                )
            else:
                nodes[ (d_x, d_y, step)].leftneighbor = (x1, y1,  step)

    return nodes


def trans_edge2node_test(raw_inter_edges_by_step,P,N):
    nodes = {}
    for step, edges in raw_inter_edges_by_step.items():
        for src, dsts in edges.items():
            x1 = src // N
            y1 = src % N
            # if x1==1 and y1==0 and step==3:
            #     print(1)
            if not dsts:
                continue  # 跳过空集合，防止 StopIteration
            dst = next(iter(dsts))
            d_x=dst//N
            d_y=dst%N
            if (x1, y1,  step) not in nodes:
                nodes[(x1, y1,  step)] = tegnode.tegnode_new(
                    asc_nodes_region_id=False,
                    rightneighbor=(d_x, d_y, step),
                    leftneighbor=None,
                    right_state = 1,
                    left_state=-1,
                    node_type=1,
                    timelast = 0,
                )
            else:
                nodes[ (x1, y1,  step)].rightneighbor = (d_x, d_y, step)
                nodes[(x1, y1, step)].right_state =1
                nodes[(x1, y1, step)].node_type = 1
                nodes[(x1, y1, step)].timelast = 0

            # 处理左邻居
            if (d_x, d_y, step) not in nodes:
                nodes[ (d_x, d_y, step)] = tegnode.tegnode_new(
                    asc_nodes_region_id=False,
                    rightneighbor=None,
                    leftneighbor=(x1, y1,  step),
                    right_state=-1,
                    left_state=1,
                    node_type=-1,
                    timelast=-1,
                )
            else:
                nodes[ (d_x, d_y, step)].leftneighbor = (x1, y1,  step)
                nodes[(d_x, d_y, step)].left_state =1

    return nodes



# 这个版本更加快
def trans_nodes2edges(nodes, P, N, check_same_step=False):
    """
    nodes: dict[(x, y, step)] -> tegnode
    return: dict[step][src_id] = set([dst_id, ...])
    仅用 rightneighbor；尽量减少 setdefault/属性与方法查找。
    """
    edges_by_step = {}
    get_step = edges_by_step.get           # 本地绑定，加速查找
    for (x, y, step), node in nodes.items():
        rn = node.rightneighbor
        if rn is None:
            continue
        rx, ry, rstep = rn
        if check_same_step and rstep != step:
            # 如需严格保证同一时间步才连边，打开上面开关
            continue

        step_map = get_step(step)
        if step_map is None:
            step_map = {}
            edges_by_step[step] = step_map

        src = x * N + y
        dst = rx * N + ry

        dsts = step_map.get(src)
        if dsts is None:
            # 直接构造包含首个元素的 set，比 setdefault 再 add 更省一次查找
            step_map[src] = {dst}
        else:
            dsts.add(dst)

    return edges_by_step

# def trans_nodes2edges(nodes, P, N):
#     """
#     nodes: dict[(x, y, step)] -> tegnode
#     返回: dict[step][src_id] = set([dst_id, ...])
#     只处理 rightneighbor
#     """
#     edges_by_step = {}
#     for (x, y, step), node in nodes.items():
#         if node.rightneighbor is not None:
#             rx, ry, rstep = node.rightneighbor
#             if step not in edges_by_step:
#                 edges_by_step[step] = {}
#             src_id = x * N + y
#             dst_id = rx * N + ry
#             edges_by_step[step].setdefault(src_id, set()).add(dst_id)
#     return edges_by_step
#


import draw.basic_functio.motif as motif



def trans_nodes2_pendingedges(nodes, start_ts, end_ts,time_2_build,P, N):
    """
    nodes: dict[(x, y, step)] -> tegnode
    返回: dict[step][src_id] = set([dst_id, ...])
    只处理 rightneighbor
    """
    pendingnodes = {}
    for step in range(start_ts, end_ts-1):

        for i in range(P - 1):
            for j in range(N):
                n1 = nodes.get((i, j, step))
                n2 = nodes.get((i, j, step + 1))
                # 防御式判断
                # 这个表明，某个点在step+1时，其链接改变了，此刻，我们需要直接修改
                if n2 is not None:
                    if n1 is None:
                        right_neighbor = n2.rightneighbor
                        for k in range(time_2_build):
                            bias = step - k
                            if bias < start_ts:
                                break
                            pendingnodes[i, j, bias] = tegnode.tegnode_new(
                                asc_nodes_region_id=-1,
                                rightneighbor=right_neighbor,
                                leftneighbor=None,
                                state=-1,
                                importance=0,
                            )

                    else:


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
    return pending_edges


def trans_nodes2_pendingedges2(nownodes, start_ts, end_ts, time_2_build, P, N):
# 这一个跟之前的不一样在于，我们是针对新的
    pendingnodes = {}
    for step in range(start_ts, end_ts-1):
        for i in range(P - 1):
            for j in range(N):
                n1 = nownodes.get((i, j, step))
                n2 = nownodes.get((i, j, step + 1))
                # 防御式判断
                # 这个表明，某个点在step+1时，其链接改变了，此刻，我们需要直接修改
                if not n1.rightneighbor and n2.rightneighbor:
                    #  说明是建链完成了，因此，我们要反向将建链的链路给加进来
                    right_neighbor = n2.rightneighbor
                    for k in range(time_2_build):
                        bias = step-k
                        if bias<start_ts:
                            break
                        pendingnodes[i, j,bias] = tegnode.tegnode_complete(
                            asc_nodes_region_id=-1,
                            rightneighbor=right_neighbor,
                            leftneighbor=None,
                            left_state=-1,
                            right_state=-1,
                        )

    pending_edges = motif.transform_nodes_2_rawedge(pendingnodes, P, N, start_ts, end_ts)
    return pending_edges
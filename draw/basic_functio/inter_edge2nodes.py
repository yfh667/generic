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

def trans_nodes2edges(nodes, P, N):
    """
    nodes: dict[(x, y, step)] -> tegnode
    返回: dict[step][src_id] = set([dst_id, ...])
    只处理 rightneighbor
    """
    edges_by_step = {}
    for (x, y, step), node in nodes.items():
        if node.rightneighbor is not None:
            rx, ry, rstep = node.rightneighbor
            if step not in edges_by_step:
                edges_by_step[step] = {}
            src_id = x * N + y
            dst_id = rx * N + ry
            edges_by_step[step].setdefault(src_id, set()).add(dst_id)
    return edges_by_step



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

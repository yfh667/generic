import genaric2.tegnode as tegnode

def trans_edge2node(raw_inter_edges_by_step,P,N):
    nodes = {}
    for step, edges in raw_inter_edges_by_step.items():
        for src, dsts in edges.items():
            x1 = src // N
            y1 = src % N
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

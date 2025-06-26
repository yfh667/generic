import random
import genaric2.tegnode as tegnode

from typing import Dict, Tuple


import genaric2.distinct_initial as distinct_initial


def initialize_snap_seq(
    col: int,
    N: int,
    P: int,
    T: int,
    port: Dict[Tuple[int, int, int], int],
    setuptime: int,
    nodes:Dict[Tuple[int, int, int], tegnode.tegnode]
) :
    """
    初始化当前列的右邻居连接。

    参数:
        col: 当前列编号（x轴）
        N: 每列的行数（y轴）
        P: 总列数
        T: 时间层数（z轴）
        port: 所有节点的端口占用情况
        setuptime: 设置链路所需时间延迟
        nodes: 所有节点对象

    返回:
        links: Dict[(row, t)] -> Tuple[next_x, next_y, next_t] 或 1
    """
    links = {(row, t): -1 for row in range(N) for t in range(T)}

    for t in range(T):
        for row in range(N):
            curr_node = (col, row, t)

            # 若状态为 0，表示链路已手动设定，直接记录
            if nodes[curr_node].state == 0:
                neighbor = nodes[curr_node].rightneighbor
                links[(row, t)] = neighbor


                # 标记设链路期间所占用的时间
                for dt in range(1,setuptime+1):

                    if neighbor[2] + dt < T:
                        links[(row, t + dt)] = 1
                continue

            # 如果该节点不是 free 状态，跳过
            if nodes[curr_node].state != -1:
                links[(row, t)] = 1
                continue

            # 如果已经分配过链接（比如被前面逻辑设置），跳过
            if links[(row, t)] != -1:
                continue

            # 搜索候选右邻居
            candidates = []
            next_col = col + 1

            if t + setuptime < T and next_col < P:
                candidates += [
                    (next_col, (row - 1) % N, t + setuptime),
                    (next_col, row % N,       t + setuptime),
                    (next_col, (row + 1) % N, t + setuptime),
                ]

                if next_col + 1 < P:
                    candidates.append((next_col + 1, row, t + setuptime))

            # 筛选可用端口的候选项
          #  available = [c for c in candidates if port.get(c, -1) == -1]
            available = []
            for k in range(len(candidates)):
                if port[candidates[k]] == -1:
                    # first it can't occupy port directly
                    available.append(candidates[k])

            if available:
                chosen = random.choice(available)
                if chosen==(5,8,5):
                    print("")

                links[(row,t)] = chosen
                port[chosen] = 1

                for dt in range(1,setuptime+1):
                    if chosen[2] - dt >= 0:
                        port[(chosen[0], chosen[1], chosen[2] - dt)] = 1
                    if t + dt < T:
                        links[(row, t + dt)] = 1

    return links




def initialize_snap_random(
    col: int,
    N: int,
    P: int,
    T: int,
    port: Dict[Tuple[int, int, int], int],
    setuptime: int,
    nodes:Dict[Tuple[int, int, int], tegnode.tegnode]
) :
    """
    初始化当前列的右邻居连接。

    参数:
        col: 当前列编号（x轴）
        N: 每列的行数（y轴）
        P: 总列数
        T: 时间层数（z轴）
        port: 所有节点的端口占用情况
        setuptime: 设置链路所需时间延迟
        nodes: 所有节点对象

    返回:
        links: Dict[(row, t)] -> Tuple[next_x, next_y, next_t] 或 1
    """
    links = {(row, t): -1 for row in range(N) for t in range(T)}
    all_coords = [(row, t) for t in range(T) for row in range(N)]
    random.shuffle(all_coords)

    for row, t in all_coords:

            curr_node = (col, row, t)

            # 若状态为 0，表示链路已手动设定，直接记录
            if nodes[curr_node].state == 0:
                neighbor = nodes[curr_node].rightneighbor
                links[(row, t)] = neighbor


                # 标记设链路期间所占用的时间
                for dt in range(1,setuptime+1):

                    if neighbor[2] + dt < T:
                        links[(row, t + dt)] = 1
                continue

            # 如果该节点不是 free 状态，跳过
            if nodes[curr_node].state != -1:
                links[(row, t)] = 1
                continue

            # 如果已经分配过链接（比如被前面逻辑设置），跳过
            if links[(row, t)] != -1:
                continue

            # 搜索候选右邻居
            candidates = []
            next_col = col + 1

            if t + setuptime < T and next_col < P:
                candidates += [
                    (next_col, (row - 1) % N, t + setuptime),
                    (next_col, row % N,       t + setuptime),
                    (next_col, (row + 1) % N, t + setuptime),
                ]

                if next_col + 1 < P:
                    candidates.append((next_col + 1, row, t + setuptime))

            # 筛选可用端口的候选项
          #  available = [c for c in candidates if port.get(c, -1) == -1]
            available = []
            for k in range(len(candidates)):
                if port[candidates[k]] == -1:
                    # first it can't occupy port directly
                    available.append(candidates[k])

            if available:
                chosen = random.choice(available)
                if chosen==(5,8,5):
                    print("")

                links[(row,t)] = chosen
                port[chosen] = 1

                for dt in range(1,setuptime+1):
                    if chosen[2] - dt >= 0:
                        port[(chosen[0], chosen[1], chosen[2] - dt)] = 1
                    if t + dt < T:
                        links[(row, t + dt)] = 1

    return links


# def nootherlink()


def initialize_snap_seq_nodes(
        col: int,
        N: int,
        P: int,
        T: int,
        # port: Dict[Tuple[int, int, int], int],
        setuptime: int,
        nodes: Dict[Tuple[int, int, int], tegnode.tegnode]
):
    """
    初始化当前列的右邻居连接。

    参数:
        col: 当前列编号（x轴）
        N: 每列的行数（y轴）
        P: 总列数
        T: 时间层数（z轴）
        port: 所有节点的端口占用情况
        setuptime: 设置链路所需时间延迟
        nodes: 所有节点对象

    返回:
        links: Dict[(row, t)] -> Tuple[next_x, next_y, next_t] 或 1
    """

    for t in range(T):
        for row in range(N):
            curr_node = (col, row, t)
            # 如果该节点不是 free 状态，跳过
            if nodes[curr_node].state != -1:
                continue

            if nodes[curr_node].asc_nodes_flag == 1:
                continue
            # 搜索候选右邻居
            candidates = []
            next_col = col + 1

            if t + setuptime < T and next_col < P:
                candidates += [
                    (next_col, (row - 1) % N, t + setuptime),
                    (next_col, row % N, t + setuptime),
                    (next_col, (row + 1) % N, t + setuptime),
                ]

                if next_col + 1 < P:
                    candidates.append((next_col + 1, row, t + setuptime))

            available = []
            for k in range(len(candidates)):
               # if nodes[candidates[k]].state == -1:
                    if not nodes[candidates[k]].leftneighbor:
                        available.append(candidates[k])

            if available:
                chosen = random.choice(available)
                start_node_id = col * N + row
                end_node_id = chosen[0] * N + chosen[1]
                start_time = t
                distinct_initial.initialize_establish(N, nodes, start_node_id, end_node_id, start_time, setuptime)

    return nodes

def no_left(nodes,candidate_node,setuptime):
    flag = 1
    for i in range(setuptime + 1):
        if nodes[(candidate_node[0], candidate_node[1], candidate_node[2] - i)].leftneighbor:
            flag = 0
            break
    return flag


def can_begin(nodes,T,startnodes,setuptime):
    flag = 1
    for i in range(setuptime + 1):
        if startnodes[2] + i>=T:
            flag = 0
            break
        if nodes[(startnodes[0], startnodes[1], startnodes[2] + i)].state != -1:
            flag = 0
            break
    return flag

def initialize_snap_random_nodes(
    col: int,
    N: int,
    P: int,
    T: int,
    #port: Dict[Tuple[int, int, int], int],
    setuptime: int,
    nodes:Dict[Tuple[int, int, int], tegnode.tegnode]
) :
    """
    初始化当前列的右邻居连接。

    参数:
        col: 当前列编号（x轴）
        N: 每列的行数（y轴）
        P: 总列数
        T: 时间层数（z轴）
        port: 所有节点的端口占用情况
        setuptime: 设置链路所需时间延迟
        nodes: 所有节点对象

    返回:
        links: Dict[(row, t)] -> Tuple[next_x, next_y, next_t] 或 1
    """
   # links = {(row, t): -1 for row in range(N) for t in range(T)}
    all_coords = [(row, t) for t in range(T) for row in range(N)]
    random.shuffle(all_coords)

    for row, t in all_coords:
            curr_node = (col, row, t)
            # 如果该节点不是 free 状态，跳过
            if nodes[curr_node].state != -1:
                continue

            if nodes[curr_node].asc_nodes_flag == 1:
                continue
            if not  can_begin(nodes,T,curr_node,setuptime):
                continue
            # 搜索候选右邻居
            candidates = []
            next_col = col + 1

            if t + setuptime < T and next_col < P:
                candidates += [
                    (next_col, (row - 1) % N, t + setuptime),
                    (next_col, row % N,       t + setuptime),
                    (next_col, (row + 1) % N, t + setuptime),
                ]

                if next_col + 1 < P:
                    candidates.append((next_col + 1, row, t + setuptime))


            available = []
            for k in range(len(candidates)):
                candidate_node = candidates[k]

                if no_left(nodes,candidate_node,setuptime):
                    available.append(candidates[k])

            if available:
                chosen = random.choice(available)
                start_node_id =  col*N+row
                end_node_id = chosen[0]*N+chosen[1]
                start_time = t
                distinct_initial.initialize_establish(N,T,nodes,start_node_id, end_node_id,start_time,setuptime)


    return nodes






def initialize_snap_grid_nodes(



    P: int,

        N: int,
        T: int,
    #port: Dict[Tuple[int, int, int], int],
    setuptime: int,
    nodes:Dict[Tuple[int, int, int], tegnode.tegnode]
) :
    """
    初始化当前列的右邻居连接。

    参数:
        col: 当前列编号（x轴）
        N: 每列的行数（y轴）
        P: 总列数
        T: 时间层数（z轴）
        port: 所有节点的端口占用情况
        setuptime: 设置链路所需时间延迟
        nodes: 所有节点对象

    返回:
        links: Dict[(row, t)] -> Tuple[next_x, next_y, next_t] 或 1
    """
   # links = {(row, t): -1 for row in range(N) for t in range(T)}
    all_coords = [(row, t) for t in range(T) for row in range(N)]
    random.shuffle(all_coords)


    for x in range(P-1):
        for y in range(N):
            start_node_id =x*N+y
            end_node_id = (x+1)*N+y
            distinct_initial.initialize_establish_lifecycle(N, T, nodes, start_node_id, end_node_id, 0,T-1,
                                                            setuptime)




    return nodes




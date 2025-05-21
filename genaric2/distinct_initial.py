import genaric2.tegnode as tegnode


def find_the_start(time_now, node_id, neighbor_id, group_idx, region_satellite_groups,setuptime):
    while time_now >= 0:
        region = region_satellite_groups[time_now][group_idx]
        if node_id in region and neighbor_id in region:
            time_now -= 1
        else:
            break
    #    time_now+=1
    establish_time = time_now


    node1_time = establish_time

    for i in range(setuptime):
        if node1_time < 0:

            break

        region = region_satellite_groups[node1_time][group_idx]
        if node_id not in region:
            break

        node1_time=node1_time-1

    if node1_time < -1:
        node1_time = -1

    node2_time = establish_time
    for i in range(setuptime):
        if node2_time < 0:
            break

        region = region_satellite_groups[node2_time][group_idx]
        if neighbor_id not in region:
            break

        node2_time = node2_time - 1

    if node2_time < -1:
        node2_time = -1

    if node1_time > node2_time:
        return node2_time
    else:
        return node1_time


 #   return node1_time, node2_time


# 分配状态：设定链接、工作状态、设置右邻居指向等
# def assign_state(nodes,node1_time, node2_time, T,start_node_id, end_node_id, setuptime, N):
#     if 0<=node1_time<T and 0<=node2_time<T:
#
#         x_start, y_start = divmod(start_node_id, N)
#         x_end, y_end = divmod(end_node_id, N)
#
#         # 设置链接建立时刻
#         nodes[(x_start, y_start, start_time)].state = 0
#         nodes[(x_start, y_start, start_time)].asc_nodes_flag = 1
#         nodes[(x_start, y_start, start_time)].rightneighbor = (x_end, y_end, start_time + setuptime)

def assign_state(nodes, start_time, end_time, start_node_id, end_node_id, setuptime, N):
    if end_time - start_time > setuptime:
        x_start, y_start = divmod(start_node_id, N)
        x_end, y_end = divmod(end_node_id, N)

        # 设置链接建立时刻
        nodes[(x_start, y_start, start_time)].state = 0
        nodes[(x_start, y_start, start_time)].asc_nodes_flag = 1
        nodes[(x_start, y_start, start_time)].rightneighbor = (x_end, y_end, start_time + setuptime)

        # 设置设置阶段状态
        for t in range(start_time + 1, start_time + setuptime):
            nodes[(x_start, y_start, t)].state = 1
            nodes[(x_start, y_start, t)].asc_nodes_flag = 1
        #  nodes[(x_start, y_start, t)].rightneighbor = (x_end, y_end, start_time + setuptime)

        # 设置工作阶段状态
        for t in range(start_time + setuptime, end_time ):
            nodes[(x_start, y_start, t)].state = 2
            nodes[(x_start, y_start, t)].asc_nodes_flag = 1
            nodes[(x_start, y_start, t)].rightneighbor = (x_end, y_end, t)


def distinct_initial(P,N,T,setuptime,regions_to_color):
    # 假设 P, N, T 是已知的维度上限


    # 初始化三维数组 nodes，键为 (x, y, z)，值为 tegnode 实例
    nodes = {}

    for x in range(P):  # x ∈ [0, P]
        for y in range(N):  # y ∈ [0, N]
            for z in range(T):  # z ∈ [0, T]
                nodes[(x, y, z)] = tegnode.tegnode(
                    asc_nodes_flag=False,  # 或初始为None等你自己定义
                    rightneighbor=None,
                    state=-1  # 默认初始为free
                )


    # 主循环：从后往前处理区域，建立右邻居连接
    for t in range(T - 1, -1, -1):
        region_groups = regions_to_color[t]
        for group_idx, region in enumerate(region_groups):
            for node_id in region:
                x_node, y_node = divmod(node_id, N)
                if nodes[(x_node, y_node, t)].state == -1:
                    # 检查是否有右邻居（非最右列）
                    if x_node < P - 1:
                        neighbor_id = (x_node + 1) * N + y_node
                        if neighbor_id in region:
                            start_time = find_the_start(t - 1, node_id, neighbor_id, group_idx, regions_to_color,setuptime)
                            assign_state(nodes, start_time +1,t, node_id, neighbor_id, setuptime, N)



    return  nodes

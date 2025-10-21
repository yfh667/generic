

def PlaneCominication(graph,nodes_per_plan,iplane,jplane):
    comminicationpoint = []
    for i in range(nodes_per_plan):# we begin iterate the neighbor in graph[iplane]
        for neighbor, weight in graph[iplane*nodes_per_plan+i]:
            neighborid = neighbor//nodes_per_plan
            if neighborid ==jplane:
                comminicationpoint.append([iplane*nodes_per_plan+i,neighbor])
    return comminicationpoint
def merge_paths(i, path1, j, path2):
    """
    合并路径，根据起点 i 和终点 j 动态调整 path1 和 path2 的顺序。

    :param i: 起点
    :param path1: 第一段路径，例如 [1, 2]
    :param j: 终点
    :param path2: 第二段路径，例如 [8, 9]
    :return: 合并后的路径，例如 [2, 1, 9, 8]
    """
    # 调整 path1 的方向
    if path1[-1] == i:  # 如果 path1 的最后一个节点是起点 i，反转 path1
        path1 = path1[::-1]

    # 调整 path2 的方向
    if path2[0] == j:  # 如果 path2 的第一个节点是终点 j，反转 path2
        path2 = path2[::-1]

    # 拼接路径
    merged_path = path1 + path2
    return merged_path



def inPlanPath(i,j,Iid,Jid,nodes_per_plan):
    iindex = min(i,j)
    jindex = max(j,i)
    i = iindex
    j = jindex
    zheng_hops = j - i
    ni_hops = i + nodes_per_plan - j
    path = []
    path.append(i)
    if (zheng_hops <= ni_hops):
        for k in range(j - i):
            path.append(i + k + 1)
    else:
        for k in range(i - 1, Iid * nodes_per_plan - 1, -1):
            path.append(k)
        for k in range((Iid + 1) * nodes_per_plan - 1, j - 1, -1):
            path.append(k)
    return  path
def route2(graph,start,end,nodes_per_plan):

   # nodes_per_plan = number_of_nodes // number_of_plan  # 每个圈的节点数量
    i = min(start,end)
    j  = max(start,end)

    Iid = i //nodes_per_plan
    Jid = j //nodes_per_plan
    if(Iid ==Jid):#  the same id
        path = inPlanPath(i,j,Iid,Jid,nodes_per_plan)


        return path
        # zheng_hops = j-i
        # ni_hops  = i+nodes_per_plan-j
        # path = []
        # path.append(i)
        # if(zheng_hops<=ni_hops):
        #     for k in range(j-i):
        #        path.append(i+k+1)
        # else:
        #     for k in range(i-1, Iid*nodes_per_plan-1, -1):
        #         path.append(k)
        #     for k in range( (Iid+1)*nodes_per_plan-1, j-1,-1):
        #         path.append(k)

    else:

        target_neighbor = j

        # 遍历邻接节点
        found = False
        for neighbor, weight in graph[i]:
            if neighbor == target_neighbor:
                found = True
                break
        if(found):
            path = []
            path.append(i)
            path.append(j)

            return path
        else:# 说明接下来是要通过轨间通信
            comminicationpoint = PlaneCominication(graph,nodes_per_plan,Iid,Jid)
            paths = []
            for pints in comminicationpoint:
                path1 = inPlanPath(i, pints[0], Iid, Jid, nodes_per_plan)
                path2 = inPlanPath(j, pints[1], Jid, Iid, nodes_per_plan)
                path = merge_paths(i, path1, j, path2)
                paths.append(path)
            minlength = 1000
            realpath =[]
            for path in paths:
                if(len(path) < minlength):
                    minlength = len(path)
                    realpath = path

           # print(realpath)

        # do something


            return realpath




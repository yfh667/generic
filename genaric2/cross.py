import random

import genaric2.distinct_initial as distinct_initial
	# 交叉操作
# 这里，我们要注意，可能


def crossover_wenti(p1_left, p2_right, point, N, T,setuptime):
    """
    处理交叉后可能出现的重复连接问题
    规则：如果发现重复连接，使p2_right中的连接设为-1

    参数:
        p1_left: 父代1的左部分染色体
        p2_right: 父代2的右部分染色体
        point: 交叉点层数
        N: 拓扑宽度
        T: 总层数

    返回:
        处理后的染色体
    """
    # 创建标记字典





    modified_p2_right = {k: v for k, v in p2_right.items()}

    afect_region = set()

    p1_xianxin =[]
    for k, v in p1_left.items():
        if k[0] == point :
            if v.rightneighbor:#
                    if v.rightneighbor[0]==point+2:# 终点在point+2层
                        afect_region.add(v.rightneighbor)

                        p1_xianxin.append(k)
                        if v.rightneighbor == (2,4,21):
                            print(1)

# 接下来，affection region 内的点只能由p1进行控制也就是p2_right的第point+1层要取消所有对point+2的动作
    #包括，建链动作，建链准备动作，工作动作，不可建链动作。也就是把上述所有行为都转为自由状态
    afect_region = list(afect_region)

    for i in range(len(afect_region)):
        leftnode = p2_right[afect_region[i]].leftneighbor
        if leftnode:
            if leftnode[0]==point+1:
                #首先，原来p2right 内部配对的的区域，就是指被链接一方，需要抹除所有与链接以防的行为
                #下面表明，当它的邻居处于开始建链时，那么就说明，被链接一方也要抹除自己的所有行为

                if  modified_p2_right[leftnode].state ==0:
                    t = leftnode[2]
                    x=afect_region[i][0]
                    y=afect_region[i][1]
                    for timesump in range(setuptime+1):
                        modified_p2_right[x,y,t+timesump].leftneighbor=None
                #其次，p2right作为链接一方，也要抹除所有与链接的行为，这里是不用循环的，原因在于大循环会自动找到所有连接方行为的。
                modified_p2_right[leftnode].rightneighbor = None
                modified_p2_right[leftnode].state = -1

    # 3. 拼接 modified_p2_right 和 p1_left
    merged_dict = {}
    merged_dict.update(p1_left)  # 添加 p1_left 的所有节点
    merged_dict.update(modified_p2_right)  # 添加处理后的 p2_right 节点

    for i in range(len(p1_xianxin)):
        if merged_dict[p1_xianxin[i]].state != 0:
            break
        if p1_xianxin[i]==(0,9,0):
            print(1)
        chosen =merged_dict[p1_xianxin[i]].rightneighbor

        start_node_id = p1_xianxin[i][0] * N + p1_xianxin[i][1]
        end_node_id = chosen[0] * N + chosen[1]
        start_time = p1_xianxin[i][2]
        distinct_initial.initialize_establish(N,T, merged_dict, start_node_id, end_node_id, start_time, setuptime)


    return merged_dict


def crossover_wenti2(p1_left, p2_right, point, N, T, setuptime):
    """
    处理交叉后可能出现的重复连接问题
    规则：如果发现重复连接，使p2_right中的连接设为-1

    参数:
        p1_left: 父代1的左部分染色体
        p2_right: 父代2的右部分染色体
        point: 交叉点层数
        N: 拓扑宽度
        T: 总层数

    返回:
        处理后的染色体
    """
    # 创建标记字典

    modified_p2_right = copy.deepcopy(p2_right)

    afect_region = set()

    p1_xianxin = []
    for k, v in p1_left.items():
        if k[0] == point:
            if v.rightneighbor:  #
                if v.rightneighbor[0] == point + 2:  # 终点在point+2层
                    area = v.rightneighbor
                    if p2_right[area].leftneighbor:
                        if p2_right[area].leftneighbor[0]==point+1:
                            afect_region.add(p2_right[area].leftneighbor)

                    p1_xianxin.append(k)


    # 接下来，affection region 内的点只能由p1进行控制也就是p2_right的第point+1层要取消所有对point+2的动作
    # 包括，建链动作，建链准备动作，工作动作，不可建链动作。也就是把上述所有行为都转为自由状态

    # 22. 拼接 modified_p2_right 和 p1_left,首先显性基因完成对隐形基因的影响
    merged_dict = {}

    merged_dict.update(p1_left)  # 添加 p1_left 的所有节点

    merged_dict.update(modified_p2_right)  # 添加处理后的 p2_right 节点

    # 我们首先需要把右边的point+1的所有节点的leftneighbor都取消掉，原因在于这一面的所有配置，都是由显性基因决定的，首先清楚干净
    for i in range(N):
        for j in range(T):
            merged_dict[point+1,i,j].leftneighbor=None



# 其次我们完成，point层的显性基因影响
    for i in range(len(p1_xianxin)):
        if merged_dict[p1_xianxin[i]].state != 0:
            continue
        if p1_xianxin[i] == (0, 9, 0):
            print(1)
        chosen = merged_dict[p1_xianxin[i]].rightneighbor

        start_node_id = p1_xianxin[i][0] * N + p1_xianxin[i][1]
        end_node_id = chosen[0] * N + chosen[1]
        start_time = p1_xianxin[i][2]
        distinct_initial.initialize_establish(N, T, merged_dict, start_node_id, end_node_id, start_time, setuptime)

## 其次，我们还要完成point-1层的显性基因影响。假如存在的话
    if point-1>=0:
        for i in range(N):

            for j in range(T):


                if merged_dict[point-1,i,j].state == 0:
                    right_node =  merged_dict[point-1,i,j].rightneighbor
                    if right_node:
                        if right_node[0]==point+1:
                            chosen = right_node

                            start_node_id = (point-1) * N + i
                            end_node_id = chosen[0] * N + chosen[1]
                            start_time =j
                            distinct_initial.initialize_establish(N, T, merged_dict, start_node_id, end_node_id, start_time,
                                                                  setuptime)



## 其次，隐形基因根据被影响的区域，调整自身的基因


    for i in range(N):

        for j in range(T):


            rightneighbor = merged_dict[(point+1,i,j)].rightneighbor
            if not rightneighbor:
                continue
            if rightneighbor[0]!=point+2:
                continue

            dominate_leftneighbor = merged_dict[rightneighbor].leftneighbor
            # if not dominate_leftneighbor:
            #     break
            x_dominate_leftneighbor=dominate_leftneighbor[0]
            y_dominate_leftneighbor=dominate_leftneighbor[1]


            if (x_dominate_leftneighbor,y_dominate_leftneighbor)!=(point+1,i):
                merged_dict[(point+1,i,j)].rightneighbor=None
                merged_dict[(point+1,i,j)].state=-1
            elif  (x_dominate_leftneighbor,y_dominate_leftneighbor)!=(point+1,i) and p2_right[(point+1,i,j)].state == 0 and p2_right[(point+1,i,j)].state == 1:
                merged_dict[(point + 1, i, j)].rightneighbor = None
                merged_dict[(point + 1, i, j)].state = -1



    return merged_dict



import copy
def crossover(parent1, parent2,P,N,T,setuptime):

   # point = random.randint(0, P - 2)
    point=3
    parent1_copy = copy.deepcopy(parent1)
    parent2_copy = copy.deepcopy(parent2)

    p1_left = {k: v for k, v in parent1_copy.items() if k[0] <= point}
    p1_right = {k: v for k, v in parent1_copy.items() if k[0] > point}

    p2_left = {k: v for k, v in parent2_copy.items() if k[0] <= point}
    p2_right = {k: v for k, v in parent2_copy.items() if k[0] > point}

    if point == P-2:

        # 3. 拼接 modified_p2_right 和 p1_left
        child1 = {}
        child1.update(p1_left)  # 添加 p1_left 的所有节点
        child1.update(p2_right)  # 添加处理后的 p2_right 节点

        child2 = {}
        child2.update(p2_left)  # 添加 p1_left 的所有节点
        child2.update(p1_right)  # 添加处理后的 p2_right 节点

    else:
        # 分割父代个体
        child1 = crossover_wenti2(p1_left, p2_right, point, N, T,setuptime)
        child2 = crossover_wenti2(p2_left, p1_right, point, N, T,setuptime)
    # 这里，我们开始进行处理冲突点


    return child1, child2
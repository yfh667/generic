import genaric2.tegnode as tegnode
import genaric2.action_table as action_table
from typing import Dict, Tuple


import sys

import genaric2.writetoxml as writetoxml

import draw.snapshotf_romxml as snapshotf_romxml
import graph.time_2d2 as time_2d
import genaric2.action_table as action_table
import graph.drawall as drawall
import genaric2.mutation.mutate1 as mutate1

# below the SeqTopologyChecker_leftneighbor and SeqTopologyChecker_nownodes are basic topology check
# it is a simple checker from the node's leftneighbor view
def SeqTopologyChecker_leftneighbor(nodes: Dict[Tuple[int, int, int], tegnode.tegnode], P, N, T,setuptime):
    flag=1
    for i in range(P):
        for t in range(T):
            for j in range(N):
                node = nodes[(i, j, t)]

                leftneighbor_id=node.leftneighbor


                if leftneighbor_id is None:
                    continue  # Skip this iteration if there is no left neighbor

                leftneighbor_node= nodes[ leftneighbor_id]


                if leftneighbor_node is not None:
                    # here we need use the state of the leftneighbor  to determin the edge
                    x,y,t_left=leftneighbor_id
                    if leftneighbor_node.state == 0:
                        # here we need check  the stage
                        # that we know ,the now node is the node that be connected ,and the leftneighbor's rightneighbor is the
                        #the true leftneighbor's rightneighbor
                        x2,y2,t2=leftneighbor_node.rightneighbor
                        mubiao_neighbor=(i,j,t+setuptime)
                        if (x2,y2,t2)!=mubiao_neighbor:
                            print(f"leftneighbor's rightneighbor is not the node,{(i, j, t)},")
                            flag=0
                    elif leftneighbor_node.state == 1:
                        x2, y2, t2 = leftneighbor_node.rightneighbor
                        if(x2,y2)!=(i,j):
                            print(f"leftneighbor's rightneighbor is not the node,{(i, j, t)},state is {leftneighbor_node.state}")
                            flag = 0
                        if not (t2-t<setuptime):
                            print(f"time is false,{(i, j, t)}")
                            flag = 0
                    elif leftneighbor_node.state == 2:
                        # . we need check the leftneighbor 's rightneighbor it is the node
                        if leftneighbor_node.rightneighbor != (i, j, t):
                            print(f"leftneighbor's rightneighbor is not the node,{(i, j, t)}")
                            flag = 0
    return  flag


#we need check all the nodes are in a Finate state machine
def SeqTopologyChecker_nownodes(nodes: Dict[Tuple[int, int, int], tegnode.tegnode], P, N, T, setuptime):
    flag = 1  # 初始假设所有节点状态转移都正常

    # 定义状态转移规则字典
    # 格式: {当前状态: (允许的前驱状态, 允许的后继状态)}
    transition_rules = {
        -1: ([-1, 2], [-1, 0]),  # 状态-1的规则
        0: ([-1, 2], [1]),  # 状态0的规则
        1: ([0, 1], [1, 2]),  # 状态1的规则
        2: ([1, 2], [2, -1,0])  # 状态2的规则
    }

    for i in range(P):
        for t in range(T):
            for j in range(N):
                node_id = (i, j, t)
                node = nodes[node_id]
                current_state = node.state  # 获取当前节点状态

                # 检查前驱节点 (t-1)
                if t > 0:
                    prev_id = (i, j, t - 1)
                    prev_state = nodes[prev_id].state
                    allowed_prev = transition_rules[current_state][0]

                    if prev_state not in allowed_prev:
                        print(f"状态转移错误! 节点({i},{j},{t})状态{current_state}的前驱节点状态{prev_state}无效")
                        print(f"允许的前驱状态: {allowed_prev}")
                        flag = 0

                # 检查后继节点 (t+1)
                if t < T - 1:
                    next_id = (i, j, t + 1)
                    next_state = nodes[next_id].state
                    allowed_next = transition_rules[current_state][1]

                    if next_state not in allowed_next:
                        print(f"状态转移错误! 节点({i},{j},{t})状态{current_state}的后继节点状态{next_state}无效")
                        print(f"允许的后继状态: {allowed_next}")
                        flag = 0

                # 特殊检查: 状态0的后继必须是1 (当不是最后一个时间步时)
                if current_state == 0 and t < T - 1:
                    next_id = (i, j, t + 1)
                    next_state = nodes[next_id].state
                    if next_state != 1:
                        print(f"状态转移错误! 节点({i},{j},{t})状态0的后继必须是1, 实际为{next_state}")
                        flag = 0

                # 特殊检查: 状态0不能出现在最后一个时间步
                if current_state == 0 and t == T - 1:
                    print(f"状态转移错误! 节点({i},{j},{t})状态0不能出现在最后一个时间步")
                    flag = 0

                #t=0
                # it must in 2 or 0 or -1,can'be 1.
                if t==0:
                    if current_state==1:
                        print(f"状态转移错误! 节点({i},{j},{t})状态不能为1")
                        flag=0

    return flag


def find_connection_differences_per_timestep(conn1, conn2, T):
    """按时间片找出两个连接列表之间的差异"""
    # 存储每个时间片的差异
    all_missing = []  # 每个时间片在conn2中缺失的连接
    all_extra = []  # 每个时间片在conn2中多余的连接

    for t in range(T):
        # 获取当前时间片的连接列表
        list1_t = conn1[t]
        list2_t = conn2[t]

        # 将连接转换为元组（tuple）使其可哈希
        tuple_conn1_t = [tuple(link) for link in list1_t]
        tuple_conn2_t = [tuple(link) for link in list2_t]

        # 创建集合进行比较
        set1_t = set(tuple_conn1_t)
        set2_t = set(tuple_conn2_t)

        # 找出差异
        missing_in_conn2_t = set1_t - set2_t
        extra_in_conn2_t = set2_t - set1_t

        # 记录当前时间片的差异
        all_missing.append((t, list(missing_in_conn2_t)))
        all_extra.append((t, list(extra_in_conn2_t)))

    return all_missing, all_extra


def TologialSequenceValidator(Sequence, P, N, T,setuptime):
    """
    This function takes a sequence of numbers as input and returns True if the sequence is a valid topological sequence, and False otherwise.
    """


    # 2, we could use the each node's left neighnor to get all the edge
    flag1 = SeqTopologyChecker_nownodes(Sequence, P, N, T,setuptime)

    flag2=SeqTopologyChecker_leftneighbor(Sequence, P, N, T,setuptime)
    if not  flag1&flag2:
        print("the basic topology is false")

        return -1,-1,-1



    # 1. we could use the each node's right neighbor to get the all edge
    connection_list1 = action_table.action_map2_shanpshots(Sequence, P, N, T)
    connection_list2=leftneighbor(Sequence, P, N, T,setuptime)
# 在你的函数中使用


    # 首先检查时间片数量是否一致
    if len(connection_list1) != len(connection_list2):
        print(f"时间片数量不一致! list1={len(connection_list1)}, list2={len(connection_list2)}")


        return -1,-1,-1

    # 按时间片比较连接
    all_missing, all_extra = find_connection_differences_per_timestep(connection_list1, connection_list2, T)

    has_differences = False

    # 打印每个时间片的差异
    for t in range(T):
        missing_t = all_missing[t][1]
        extra_t = all_extra[t][1]

        if missing_t or extra_t:
            has_differences = True
            print(f"\n时间片 {t} 的差异:")

            if missing_t:
                print(f"  在connection_list2中缺失的连接({len(missing_t)}个):")
                for conn in missing_t:
                    print(f"    {conn[0]} -> {conn[1]}")

            if extra_t:
                print(f"  在connection_list2中多余的连接({len(extra_t)}个):")
                for conn in extra_t:
                    print(f"    {conn[0]} -> {conn[1]}")

    # 如果没有差异但之前的比较显示不等，可能是顺序问题
    if has_differences:
        print("\n左右邻居生成的拓扑不一致。")
    else:
        print("\n拓扑序列正确")

    return not has_differences,connection_list1,connection_list2


# here we need use every node's leftneighbor to get the adjacncy list
def leftneighbor(nodes: Dict[Tuple[int, int, int], tegnode.tegnode], P, N, T,setuptime):
    adj_list_array = [[] for _ in range(T)]
    for i in range(P):
        for t in range(T):
            for j in range(N):

                node = nodes[(i, j, t)]

                leftneighbor_id=node.leftneighbor


                if leftneighbor_id is None:
                    continue  # Skip this iteration if there is no left neighbor

                leftneighbor_node= nodes[ leftneighbor_id]
                x2,y2,t2=leftneighbor_id
                leftneighbor_node_topology_id = x2*N+y2
                node_topology_id=i*N+j
                if leftneighbor_node.state==2:
                    adj_list_array[t ].append((leftneighbor_node_topology_id, node_topology_id))

    return adj_list_array




if __name__ == "__main__":
    # 示例数据 (需要替换为您的实际数据)
    N = 10
    P = 10
    T = 23
    setuptime = 2
    start_ts = 1500
    end_ts = 1523
    dummy_file_name = "E:\\code\\data\\station_visible_satellites_100_test.xml"

    # 提取区域信息
    regions_to_color = {}
    region_satellite_groups = snapshotf_romxml.extract_region_satellites_from_file(dummy_file_name, start_ts, end_ts)
    target_time_step = len(region_satellite_groups)
    T = target_time_step
    for i in range(len(region_satellite_groups)):
        region_satellite_group = [[int(point) for point in region] for region in region_satellite_groups[i]]
        u = region_satellite_group[0]
        v = region_satellite_group[3]
        o = [u, v]
        regions_to_color[i] = o
    individual1 = writetoxml.xml_to_nodes("E:\\code\\data\\1\\individual2.xml")



#usage
    flag1 = TologialSequenceValidator(individual1, P, N, T, setuptime)
#


    if not flag1:
        print("chuwentile")
    else:
        print("correct")

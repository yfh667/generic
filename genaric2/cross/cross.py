import random

import genaric2.distinct_initial as distinct_initial
	# 交叉操作
# 这里，我们要注意，可能

import genaric2.TopoSeqValidator as TopoSeqValidator






import genaric2.writetoxml as writetoxml
import draw.snapshotf_romxml as snapshotf_romxml
import graph.time_2d2 as time_2d
import genaric2.action_table as action_table
import graph.drawall as drawall
import genaric2.mutation.mutate3 as mutate
import genaric2.TopoSeqValidator as TopoSeqValidator
from genaric2.TopoSeqValidator import leftneighbor
import genaric2.mutation.basic_mutate_func as  basic_fuc

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


# def crossover_wenti2(p1_left, p2_right, point, N, T, setuptime):
#     """
#     处理交叉后可能出现的重复连接问题
#     规则：如果发现重复连接，使p2_right中的连接设为-1
#
#     参数:
#         p1_left: 父代1的左部分染色体
#         p2_right: 父代2的右部分染色体
#         point: 交叉点层数
#         N: 拓扑宽度
#         T: 总层数
#
#     返回:
#         处理后的染色体
#     """
#     # 创建标记字典
#
#     modified_p2_right = copy.deepcopy(p2_right)
#
#     afect_region = set()
#
#     p1_xianxin = []
#     for k, v in p1_left.items():
#         if k[0] == point:
#             if v.rightneighbor:  #
#                 if v.rightneighbor[0] == point + 2:  # 终点在point+2层
#                     area = v.rightneighbor
#                     if p2_right[area].leftneighbor:
#                         if p2_right[area].leftneighbor[0]==point+1:
#                             afect_region.add(p2_right[area].leftneighbor)
#
#                     p1_xianxin.append(k)
#
#
#     # 接下来，affection region 内的点只能由p1进行控制也就是p2_right的第point+1层要取消所有对point+2的动作
#     # 包括，建链动作，建链准备动作，工作动作，不可建链动作。也就是把上述所有行为都转为自由状态
#
#     # 22. 拼接 modified_p2_right 和 p1_left,首先显性基因完成对隐形基因的影响
#     merged_dict = {}
#
#     merged_dict.update(p1_left)  # 添加 p1_left 的所有节点
#
#     merged_dict.update(modified_p2_right)  # 添加处理后的 p2_right 节点
#
#     # 我们首先需要把右边的point+1的所有节点的leftneighbor都取消掉，原因在于这一面的所有配置，都是由显性基因决定的，首先清楚干净
#     for i in range(N):
#         for j in range(T):
#             merged_dict[point+1,i,j].leftneighbor=None
#
#
#
# # 其次我们完成，point层的显性基因影响
#     for i in range(len(p1_xianxin)):
#         if merged_dict[p1_xianxin[i]].state != 0:
#             continue
#         if p1_xianxin[i] == (0, 9, 0):
#             print(1)
#         chosen = merged_dict[p1_xianxin[i]].rightneighbor
#
#         start_node_id = p1_xianxin[i][0] * N + p1_xianxin[i][1]
#         end_node_id = chosen[0] * N + chosen[1]
#         start_time = p1_xianxin[i][2]
#         distinct_initial.initialize_establish(N, T, merged_dict, start_node_id, end_node_id, start_time, setuptime)
#
# ## 其次，我们还要完成point-1层的显性基因影响。假如存在的话
#     if point-1>=0:
#         for i in range(N):
#
#             for j in range(T):
#
#
#                 if merged_dict[point-1,i,j].state == 0:
#                     right_node =  merged_dict[point-1,i,j].rightneighbor
#                     if right_node:
#                         if right_node[0]==point+1:
#                             chosen = right_node
#
#                             start_node_id = (point-1) * N + i
#                             end_node_id = chosen[0] * N + chosen[1]
#                             start_time =j
#                             distinct_initial.initialize_establish(N, T, merged_dict, start_node_id, end_node_id, start_time,
#                                                                   setuptime)
#
#
#
# ## 其次，隐形基因根据被影响的区域，调整自身的基因
#
#
#     for i in range(N):
#
#         for j in range(T):
#
#
#             rightneighbor = merged_dict[(point+1,i,j)].rightneighbor
#             if not rightneighbor:
#                 continue
#             if rightneighbor[0]!=point+2:
#                 continue
#
#             dominate_leftneighbor = merged_dict[rightneighbor].leftneighbor
#             # if not dominate_leftneighbor:
#             #     break
#             x_dominate_leftneighbor=dominate_leftneighbor[0]
#             y_dominate_leftneighbor=dominate_leftneighbor[1]
#
#
#             if (x_dominate_leftneighbor,y_dominate_leftneighbor)!=(point+1,i):
#                 merged_dict[(point+1,i,j)].rightneighbor=None
#                 merged_dict[(point+1,i,j)].state=-1
#             elif  (x_dominate_leftneighbor,y_dominate_leftneighbor)!=(point+1,i) and p2_right[(point+1,i,j)].state == 0 and p2_right[(point+1,i,j)].state == 1:
#                 merged_dict[(point + 1, i, j)].rightneighbor = None
#                 merged_dict[(point + 1, i, j)].state = -1
#
#
#
#     return merged_dict

def merge_node_dicts_safe(p1_left, modified_p2_right):
    """
    高效合并两个无冲突的节点字典
    返回一个新字典，包含两个输入字典的所有内容
    """
    return {**p1_left, **modified_p2_right}

# 使用示例

def write_state_to_rightneighbor(p1_left, point, i, j, t, x2, y2, rightneighbor, modified_p2_right,setuptime):
    if p1_left[(point - 1 + i, j, t)].state == 0:

        for k in range(t, t + setuptime + 1):
            modified_p2_right[(x2, y2, k)].leftneighbor = (point - 1 + i, j, k)
    elif p1_left[(point - 1 + i, j, t)].state == 2:

        modified_p2_right[rightneighbor].leftneighbor = (point - 1 + i, j, t)


def crossover_wenti2(p1_left, p2_right, point,P, N, T, setuptime):
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



# 1. we need delete all the leftneighbor for the point+1 for p2_right

    for t in range(T):
        for j in range(N):
            modified_p2_right[point+1,j,t].leftneighbor=None

    #individual1 = merge_node_dicts_safe(p1_left, modified_p2_right)
    # connection2_test = action_table.action_map2_shanpshots(individual1, P, N, T)
    # vis = time_2d.DynamicGraphVisualizer(connection2_test, regions_to_color, N, P)
    # vis.show()
    #
    # # 3D 拓扑图可视化
    # main_plotter, original_points_objs, all_coords = drawall.plot_multi_layer_topology(P, N, target_time_step)
    # main_plotter = drawall.apply_region_colors(main_plotter, P, N, target_time_step, regions_to_color, all_coords)
    #
    # main_plotter = drawall.add_dashed_connections(main_plotter, connection2_test)
    # main_plotter.show(viewup="z", title="Interactive 3D Topology")

    #  then we need write the state from p1_left to the modifed_p2_right


    #flag1, connection1_test, connection2_test = TopoSeqValidator.TologialSequenceValidator(individual1, P, N, T,    setuptime)


# we need use the p1_left to write to p2_right






    # test = merge_node_dicts_safe(p1_left, modified_p2_right)
    #flag1, connection1_test, connection2_test = TopoSeqValidator.TologialSequenceValidator(test, P, N, T,    setuptime)

    if point<P-2:
    # we need clear all the nodes in the p2_right,which leftneighbor is in point,because p2_right point col is dropped
    #2. we need change the point+2's state for the afect_region
        afect_region = set()
        for t in range(T):
                for j in range(N):
                    rightneighbor = p1_left[(point, j, t)].rightneighbor
                    if rightneighbor:
                        x2, y2, t2 = rightneighbor
                        if x2 == point + 2:
                            if p1_left[(point, j, t)].state == 0:
                                for k in range(t, t + setuptime + 1):

                                  #  modified_p2_right[x2, y2, k].leftneighbor = (point, j, k)
                                    afect_region.add((x2, y2, k))

                            elif p1_left[(point, j, t)].state == 2:
                                afect_region.add(rightneighbor)
                           #     modified_p2_right[rightneighbor].leftneighbor = (point, j, t)

                # 假设 afect_region 是包含三元组 (x, y, t) 的集合

        glag=1
        for coord in afect_region:
            x, y, t = coord  # 解包坐标

            # if coord==(6,4,6):
            #     print(1)

            leftneighbor = p2_right[coord].leftneighbor



            # 在这里执行你的操作，例如：
            if leftneighbor:
                x2,y2,t2=leftneighbor
                if x2==point+1:

                    basic_fuc.clear_state(leftneighbor, modified_p2_right, P, N, T, setuptime)

                    #
                    # if p2_right[leftneighbor].state == 0:#that means we need delete all the state for this link
                    #     for k in range(t2, t2 + setuptime + 1):
                    #
                    #
                    #         modified_p2_right[( x2,y2,k)].state = -1
                    #         modified_p2_right[(x2, y2, k)].rightneighbor = None
                    # elif p2_right[leftneighbor].state == 1:
                    #     # up
                    #
                    #
                    #     modified_p2_right[leftneighbor].state = -1
                    #     modified_p2_right[leftneighbor].rightneighbor = None
                    #     uptime=leftneighbor[2]
                    #     while uptime-1>=0:
                    #         uptime=uptime-1
                    #         if p2_right[(x2,y2,uptime)].state == 0 or p2_right[(x2,y2,uptime)].state == 1:
                    #
                    #
                    #
                    #             modified_p2_right[(x2,y2,uptime)].state = -1
                    #             modified_p2_right[(x2,y2,uptime)].rightneighbor = None
                    #         else:
                    #             break
                    #
                    #     downtime = leftneighbor[2]
                    #     while downtime + 1 < T:
                    #         downtime = downtime + 1
                    #         if p2_right[(x2, y2, downtime)].state == 2 or p2_right[(x2, y2, downtime)].state == 1:
                    #
                    #
                    #
                    #
                    #             modified_p2_right[(x2, y2, downtime)].state = -1
                    #             modified_p2_right[(x2, y2, downtime)].rightneighbor = None
                    #         else:
                    #             break
                    #
                    # elif p2_right[leftneighbor].state == 2:
                    #
                    #     modified_p2_right[leftneighbor].state = -1
                    #     modified_p2_right[leftneighbor].rightneighbor = None
                    #     uptime = leftneighbor[2]
                    #
                    #
                    #     # we need find the zhouqi
                    #     flag=1
                    #     while uptime - 1 >= 0 and flag:
                    #         uptime = uptime - 1
                    #
                    #         if p2_right[(x2, y2, uptime)].state == 0:
                    #             flag=flag-1
                    #
                    #
                    #
                    #
                    #
                    #         modified_p2_right[(x2, y2, uptime)].state = -1
                    #         modified_p2_right[(x2, y2, uptime)].rightneighbor = None

            # if modified_p2_right[5,4,7].rightneighbor==None:
            #             print(1)
        for t in range(T):
                for j in range(N):
                    leftneighbor=p2_right[(point+2, j, t)].leftneighbor
                    if leftneighbor:
                        x2, y2, t2 = leftneighbor
                        if x2 == point :

                            modified_p2_right[(point+2, j, t)].leftneighbor = None


        for t in range(T):
                for j in range(N):
                    rightneighbor = p1_left[(point, j, t)].rightneighbor
                    if rightneighbor:
                        x2, y2, t2 = rightneighbor
                        if x2 == point + 2:
                            if p1_left[(point, j, t)].state == 0:
                                for k in range(t, t + setuptime + 1):

                                    modified_p2_right[x2, y2, k].leftneighbor = (point, j, k)


                            elif p1_left[(point, j, t)].state == 2:

                                modified_p2_right[rightneighbor].leftneighbor = (point, j, t)
    # test = merge_node_dicts_safe(p1_left, modified_p2_right)
    # flag1, connection1_test, connection2_test = TopoSeqValidator.TologialSequenceValidator(test, P, N, T,    setuptime)
    #    writetoxml.nodes_to_xml(test, "E:\\code\\data\\1\\test.xml")

    #3. we need change the affect region
    for i in range(2):
        for t in range(T):
            for j in range(N):
                if point-1+i>=0:
                    rightneighbor = p1_left[(point-1+i, j, t)].rightneighbor


                    if rightneighbor:

                        x2,y2,t2=rightneighbor
                        if x2==point+1:
                            write_state_to_rightneighbor(p1_left, point, i, j, t, x2, y2, rightneighbor, modified_p2_right,
                                                         setuptime)


    child = merge_node_dicts_safe(p1_left, modified_p2_right)
    return child



import copy
def crossover(parent1, parent2,P,N,T,setuptime,test=0):

    if not test:
        point = random.randint(0, P - 2)
    else:
        point=5
   # point = random.randint(0, P - 2)



    parent1_copy = copy.deepcopy(parent1)
    parent2_copy = copy.deepcopy(parent2)

    p1_left = {k: v for k, v in parent1_copy.items() if k[0] <= point}
    p1_right = {k: v for k, v in parent1_copy.items() if k[0] > point}

    p2_left = {k: v for k, v in parent2_copy.items() if k[0] <= point}
    p2_right = {k: v for k, v in parent2_copy.items() if k[0] > point}

    # if point == P-2:
    #
    #     # 3. 拼接 modified_p2_right 和 p1_left
    #     child1 = {}
    #     child1.update(p1_left)  # 添加 p1_left 的所有节点
    #     child1.update(p2_right)  # 添加处理后的 p2_right 节点
    #
    #     child2 = {}
    #     child2.update(p2_left)  # 添加 p1_left 的所有节点
    #     child2.update(p1_right)  # 添加处理后的 p2_right 节点
    #
    # else:
        # 分割父代个体
    child1 = crossover_wenti2(p1_left, p2_right, point,P, N, T,setuptime)
    child2 = crossover_wenti2(p2_left, p1_right, point, P,N, T,setuptime)
    # 这里，我们开始进行处理冲突点


    return child1, child2,point

if __name__ == '__main__':

    ##1.我们先获得原始的卫星分布数据，主要是热点区域的拓扑序列数据
    N = 10
    P = 10
    start_ts = 1500
    end_ts = 1523

    dummy_file_name = "E:\\code\\data\\station_visible_satellites_100_test.xml"

    regions_to_color = {}
    # Iterate over the time steps
    region_satellite_groups = snapshotf_romxml.extract_region_satellites_from_file(dummy_file_name, start_ts, end_ts)
    target_time_step = len(region_satellite_groups)
    T = target_time_step
    for i in range(len(region_satellite_groups)):
        region_satellite_group = [[int(point) for point in region] for region in region_satellite_groups[i]]
        u = region_satellite_group[0]
        v = region_satellite_group[3]
        o = [u, v]
        regions_to_color[i] = o  # Corrected append to dictionary assignment

    ##2. 完成对热点区域预建链安排

    T = target_time_step
    setuptime = 2
    base = distinct_initial.distinct_initial(P, N, T, setuptime, regions_to_color)

    individual1 = writetoxml.xml_to_nodes("E:\\code\\data\\2\\parent1.xml")
    individual2 = writetoxml.xml_to_nodes("E:\\code\\data\\2\\parent2.xml")






    #
##usage
    child1, child2,_ = crossover(individual1, individual2, P, N, T, setuptime, 1)
    #attebntion ,if it is not test , just use
    #child1, child2 = crossover(individual1, individual2, P, N, T, setuptime)
###











    flag1, connection1_test, connection2_test = TopoSeqValidator.TologialSequenceValidator(child1, P, N, T, setuptime)
    #
    writetoxml.nodes_to_xml(child1, "E:\\code\\data\\2\\child1.xml")

    writetoxml.nodes_to_xml(child2, "E:\\code\\data\\2\\child2.xml")

    # writetoxml.nodes_to_xml(child1, "E:\\code\\data\\1\\child1.xml")

    # 2D 动态图可视化
    connection_list = action_table.action_map2_shanpshots(individual1, P, N, T)
    vis = time_2d.DynamicGraphVisualizer(connection_list, regions_to_color, N, P)
    vis.show()

    # 3D 拓扑图可视化
    main_plotter, original_points_objs, all_coords = drawall.plot_multi_layer_topology(P, N, target_time_step)
    main_plotter = drawall.apply_region_colors(main_plotter, P, N, target_time_step, regions_to_color, all_coords)

    main_plotter = drawall.add_dashed_connections(main_plotter, connection_list)
    main_plotter.show(viewup="z", title="Interactive 3D Topology")

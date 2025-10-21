
import basicSa.getthepath.checkconflic as  checkconflic
import basicSa.getthepath.timegraph as  Timegraph

import collections
import heapq
from math import inf
from collections import defaultdict
from collections import deque
import basicSa.getthepath.tiemsegment as tiemsegment



def build_path_graph(paths, time_segments,chenfa,conflict):
    """
    构建路径图(graph)并计算abundance和cumulative abundance

    参数:
        paths: 列表形式的路径信息
        示例:
        [
            {  # 路径0
                'path': [1, 2, 3, 4],
                'time_intervals': [(0, 55), (70, 100)]
            },
            {  # 路径1
                'path': [1, 2, 5, 4],
                'time_intervals': [(0, 20), (44, 100)]
            }
        ]

        time_segments: 时间间隔列表，如 [(0,20), (20,40), ...]
    """
    # 首先获取每个时间间隔内各路径的状态
    segment_status = get_path_status(paths, time_segments)

    # 初始化图结构
    graph = {}
    timegraph = Timegraph.TimeGraph()

    # 将paths转为字典格式方便索引（如果需要）
    paths_dict = {idx: path for idx, path in enumerate(paths)}
    temp_last_node = [(-1, -1)] * len(paths)
    own_last_node = [(-1, -1)] * len(paths)

    matrix = [[-1 for _ in range(len(paths))] for _ in range(len(paths))]


    # 为每个路径在每个时间点创建节点
    for seg_idx, segment in enumerate(time_segments):
        seg_start, seg_end = segment
        duration = seg_end - seg_start
        for path_idx in range(len(paths)):
            # 确定当前路径在当前间隔是否活跃
            is_active = segment_status[seg_idx]['status'][path_idx]

            if is_active:

                ## we need add two parts,the before and the end ,we need check the
                #head is exits?
                head = (seg_idx , path_idx)
                if head in  timegraph.node_attributes:
                    pass
                else:
                    abundance=0
                    cumulative=0

                    own_last_node[path_idx] = head
                    timegraph.add_node_attribute(head, abundance=abundance, cumulative=cumulative)

                # here  we arragnge all the nodes

                end = (seg_idx+1 , path_idx)
                # 计算abundance

                abundance = duration
                cumulative = 0
                # 计算cumulative abundance

                if abundance > 0:  # 只在路径活跃时处理

                    prev_node_id =own_last_node[path_idx]
                    # if prev_node_id in graph:
                    cumulative =  timegraph.node_attributes[prev_node_id]['cumulative'] + abundance

                    # 添加同路径的边
                #    timegraph.add_edge(prev_node_id, end, abundance)

                    # 添加冲突路径的惩罚边
                #    temp_last_node[path_idx] = end
                #     matrix[path_idx] = [-1] * len(paths)  # 该行全部 -1
                    own_last_node[path_idx] = end

                timegraph.add_node_attribute(end, abundance=abundance, cumulative=cumulative)
        #here  we arragnge all the nodes


    own_last_node = [(-1, -1)] * len(paths)

    for i in range(len(time_segments)+1):
        for path_idx in range(len(paths)):
            nodeid = (i, path_idx)
            if nodeid in timegraph.node_attributes:
                if timegraph.node_attributes[nodeid]['cumulative'] != 0:
                    temp_last_node[path_idx] = nodeid
                    matrix[path_idx] = [-1] * len(paths)  # 该行全部 -1

        for path_idx in range(len(paths)):
            nodeid = (i,path_idx)



            if nodeid in  timegraph.node_attributes:


                if  own_last_node[path_idx] !=(-1,-1):
                    abundance = timegraph.node_attributes[nodeid]['abundance']
                    timegraph.add_edge(own_last_node[path_idx], nodeid, abundance)  # 惩罚边

                own_last_node[path_idx] = nodeid

                for other_path_idx in range(len(paths)):
                    if other_path_idx != path_idx:
                        if temp_last_node[other_path_idx] != (-1, -1):
                            if matrix[other_path_idx][path_idx] == -1:
                                other_prev_node = temp_last_node[other_path_idx]
                                if conflict[other_path_idx][path_idx]:
                                    timegraph.add_edge(other_prev_node, nodeid, chenfa)  # 惩罚边
                                else:
                                    timegraph.add_edge(other_prev_node, nodeid, 0)  # 惩罚边

                                matrix[other_path_idx][path_idx]=1





    return timegraph


def get_path_status(paths, time_segments):
    """获取每个时间间隔内各路径的状态"""
    segment_status = []
    for seg_start, seg_end in time_segments:
        status = {}
        for path_idx, path_data in enumerate(paths):  # path_idx 是数字索引，代表路径序号
            active = False
            for int_start, int_end in path_data['time_intervals']:
                # 检查时间片段是否在本段区间内
                if int_start <= seg_start and seg_end <= int_end:
                    active = True
                    break
            status[path_idx] = active
        segment_status.append({'segment': (seg_start, seg_end), 'status': status})
    return segment_status

def build_Time_graph(paths,time_segments,chenfa,conflict_matrix):
   # time_segments = tiemsegment.calculate_time_segments(paths)


    graph = build_path_graph(paths, time_segments,chenfa,conflict_matrix)
    return graph

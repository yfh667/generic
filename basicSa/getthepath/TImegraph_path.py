

import basicSa.getthepath.checkconflic as  checkconflic
import basicSa.getthepath.timegraph as  Timegraph

import collections
import heapq
from math import inf
from collections import defaultdict
from collections import deque
import basicSa.getthepath.tiemsegment as tiemsegment
import basicSa.getthepath.buildTimegraph as buildTimegraph
def get_time_segments( time_segments, dijkstra_path, distance):
    """
    计算各路径的实际使用时间区间

    参数:
        paths: 路径字典 {path_id: path_info}
        time_segments: 原始时间段列表 [(start1, end1), (start2, end2), ...]
        dijkstra_path: 最短路径节点列表 [(seg_idx1, path_id1), (seg_idx2, path_id2), ...]
        distance: 距离字典 {node: dist}

    返回:
        dict: {path_id: [time_interval1, time_interval2, ...]}
    """
    path_intervals = defaultdict(list)

    for i in range(len(dijkstra_path) - 1):
        current_node = dijkstra_path[i]
        next_node = dijkstra_path[i + 1]

        # 获取当前路径ID和时间段索引
        path_id = next_node[1]
        seg_idx = current_node[0]

        # 计算实际使用时长
        seg_start, seg_end = time_segments[seg_idx]
        seg_duration = seg_end - seg_start
        usage_duration = distance[next_node] - distance[current_node]

        # 计算实际使用时间区间
        usage_start = seg_start + (seg_duration - usage_duration)
        path_intervals[path_id].append((usage_start, seg_end))

    # 合并连续的时间区间
    merged_intervals = {}
    for path_id in path_intervals:
        intervals = sorted(path_intervals[path_id])
        merged = []

        for interval in intervals:
            if not merged:
                merged.append(list(interval))
            else:
                last = merged[-1]
                if interval[0] <= last[1]:  # 重叠或连续
                    last[1] = max(last[1], interval[1])
                else:
                    merged.append(list(interval))

        merged_intervals[path_id] = [tuple(interval) for interval in merged]

    return merged_intervals
def find_shortest_path(self, start_node,conflict_matrix,time_segments):
    """
    使用Dijkstra算法查找最短路径
    参数:
        start_node: 起始节点，如(0,0)
        end_time: 目标时间段索引（如7表示时间段(70,100)）
    返回:
        (total_weight, path) 元组
    """
    # 初始化
    distances = defaultdict(lambda: 0)
    distances[start_node] = 0
    previous_nodes = {}
   # heap = [( 0,start_node)]
    queue=[]
   # queue = deque([(0, start_node)])  # 使用双端队列替代堆
    #conflict_matrix = checkconflic.build_conflict_matrix(paths)
    heapq.heappush(queue, (start_node[0], start_node))  # (priority, node)

    while queue:
      #  current_dist, current_node =  queue.popleft()

        current_priority, current_node = heapq.heappop(queue)  # 每次取出 least current_node[0]
        current_dist = distances[current_node]
        # # 到达目标时间段即可终止
        # if current_node[0] == end_time:
        #     break

        if current_dist < distances[current_node]:
            continue

        # 遍历所有邻居
        for neighbor in self.get_neighbors(current_node):
            weight = self.get_edge_weight(current_node, neighbor)
            if neighbor[1] != current_node[1]:
                flag = conflict_matrix[neighbor[1]][current_node[1]]
                if flag:
                    start_time_1 = (current_node[0]+1) //2
                    start_time_2=(current_node[0]+1) %2
                    start_time = time_segments[start_time_1][start_time_2]

                    end_time_1 = (neighbor[0]+1) // 2
                    end_time_2 = (neighbor[0]+1) % 2
                    end_time = time_segments[end_time_1][end_time_2]
                    interval = end_time-start_time
                    new_dist = current_dist + min(weight+interval,0)


                else:

                    new_dist = current_dist + weight

            else:
                new_dist = current_dist + weight

            if new_dist > distances[neighbor]:
                distances[neighbor] = new_dist
                previous_nodes[neighbor] = current_node
              #  heapq.heappush(heap, (new_dist, neighbor))
                new_priority = neighbor[0]  # 你要排序的依据
                heapq.heappush(queue, (new_priority, neighbor))  # 按优先级插入
           #     queue.append((new_dist, neighbor))  # 添加到队列末尾

    # 找到目标时间段中距离最小的节点
    max_dist = -float('inf')
    best_node = None

    for node, dist in distances.items():
        if dist > max_dist:
            max_dist = dist
            best_node = node



    total_weight = distances[best_node]

    # 重建路径
    path = []
    current = best_node
    while current is not None:
        path.append(current)
        current = previous_nodes.get(current)
    path.reverse()

    u = get_time_segments(time_segments, path, distances)
    return (u,total_weight, path)


def find_shortest_paths(graph,conflict_matrix,time_segments,paths):
    number_of_paths= len(paths)
    startnodes = []

    for i in range(number_of_paths):
        if (0,i) in graph.node_attributes:
            startnodes.append((0,i))

    bestpaths = []
    max = 0
    for i in range(len(startnodes)):
        bestpath = find_shortest_path(graph, startnodes[i],conflict_matrix,time_segments)
        if bestpath[1]>max:
            max = bestpath[1]
            path = bestpath[2]
            anpai =  bestpath[0]
    richenbiao  = []
    for key in anpai:  # 迭代键（如 0, 1, 2）
        values = anpai[key]  # 取出列表如 [(40, 60)] 或[(80, 20), (30, 50)]
        for (a, b) in values:  # 确保每个value能拆包为(a,b)


        #    print(f"key={key}, a={a}, b={b}")
            print(f"path={paths[key]['path']}, a={a}, b={b}")

    return  (anpai,max, path)




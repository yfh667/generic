
import basicSa.getthepath.checkconflic as  checkconflic
import basicSa.getthepath.timegraph as  Timegraph

import collections
import heapq
from math import inf
from collections import defaultdict
from collections import deque
import basicSa.getthepath.tiemsegment as tiemsegment
import basicSa.getthepath.buildTimegraph as buildTimegraph
import basicSa.getthepath.TImegraph_path as TImegraph_path


# paths = [
#
#       {  # 原path1
#     'path': [1, 2, 3, 4],
#     'time_intervals': [(0, 40), (50, 100)]
# },
#  {  # 原path2
#     'path': [1, 2, 5, 4],
#     'time_intervals': [(0, 20), (44, 100)]
# },
#  {  # 原path3
#     'path': [1, 7, 6, 4],
#     'time_intervals': [(0, 55), (70, 100)]
# }
# ]

paths = [

      {  # 原path1
    'path': [1, 2, 3, 4],
    'time_intervals': [(40, 60) ]
},
 {  # 原path2
    'path': [1, 2, 5, 4],
    'time_intervals': [(0, 20) ,(63,65)]
},
 {  # 原path3
    'path': [1, 5, 6, 4],
    'time_intervals': [( 44,50),(60,63) ]
},



]



time_segments = tiemsegment.calculate_time_segments(paths)

## first we need divide all the paths into serveral big intervals. that means, some paths will not appear in this interval:
## for example:
## paths[0] will disappear after 40s,so there is no need for us to think about it,and we find a biggest interval for it ,such as (0,40) not (0,20),
##and then we will split the time into two parts, 1 is (0,20),(20,40), and 2 is (55,70),(70,80),(80,90)






chenfa=-5
conflict_matrix = checkconflic.build_conflict_matrix(paths)
graph =buildTimegraph.build_Time_graph(paths,time_segments,chenfa,conflict_matrix)
print("图构建完成，使用数字索引路径:")
print("节点ID格式: (时间段索引, 路径索引)")
interval ,total_weight, dijkstra_path = TImegraph_path.find_shortest_paths(graph,conflict_matrix,time_segments,paths)

print(f"interval: {interval}")
print(f"total_weight: {total_weight}")
print(f"dijkstra_path: {dijkstra_path}")


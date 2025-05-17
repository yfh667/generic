import ga.graphalgorithm.fcnfp_multi as fcnfp_multi
import genaric.chrom2adjact as c2a
import genaric.plotgraph as plotgraph
import satnode.relative_position as relpos
import ga.graphalgorithm.adjact2weight as a2w

import ga.graphalgorithm.mcmf.ssp_multi as ssp_multi
adjacency_list =      {0: {1, 14, 6}, 1: {0, 9, 2}, 2: {8, 1, 3}, 3: {10, 2, 4}, 4: {18, 3, 5}, 5: {11, 4, 6}, 6: {0, 5, 13}, 7: {8, 13, 21}, 8: {9, 2, 7, 15}, 9: {8, 1, 10, 17}, 10: {16, 9, 3, 11}, 11: {10, 19, 12, 5}, 12: {11, 20, 13}, 13: {27, 12, 6, 7}, 14: {0, 20, 22, 15}, 15: {8, 16, 29, 14}, 16: {17, 10, 30, 15}, 17: {24, 9, 18, 16}, 18: {25, 19, 4, 17}, 19: {33, 18, 11, 20}, 20: {34, 19, 12, 14}, 21: {27, 35, 22, 7}, 22: {28, 21, 14, 23}, 23: {24, 22, 31}, 24: {17, 25, 38, 23}, 25: {24, 18, 26, 39}, 26: {32, 25, 27}, 27: {41, 21, 26, 13}, 28: {42, 34, 29, 22}, 29: {36, 28, 30, 15}, 30: {16, 29, 37, 31}, 31: {32, 45, 30, 23}, 32: {33, 26, 46, 31}, 33: {40, 32, 19, 34}, 34: {48, 33, 20, 28}, 35: {49, 36, 21, 41}, 36: {37, 35, 43, 29}, 37: {38, 51, 36, 30}, 38: {24, 52, 37, 39}, 39: {40, 25, 38, 47}, 40: {33, 41, 54, 39}, 41: {40, 35, 27, 55}, 42: {48, 50, 43, 28}, 43: {57, 42, 36, 44}, 44: {58, 43, 45}, 45: {59, 44, 46, 31}, 46: {32, 45, 53, 47}, 47: {48, 61, 46, 39}, 48: {34, 42, 62, 47}, 49: {56, 50, 35, 55}, 50: {49, 42, 51}, 51: {50, 52, 37}, 52: {51, 60, 53, 38}, 53: {54, 52, 46}, 54: {40, 53, 55}, 55: {41, 54, 49}, 56: {49, 62, 57}, 57: {56, 58, 43}, 58: {57, 59, 44}, 59: {58, 60, 45}, 60: {59, 52, 61}, 61: {60, 62, 47}, 62: {48, 56, 61}}
N=7
P=9

full_adjacency_list = adjacency_list

inter_link_bandwidth = 50
intra_link_bandwidth = 100

cost =1

edge =a2w.adjacent2edge(full_adjacency_list,N,inter_link_bandwidth,intra_link_bandwidth,cost)



distinct = [[17,18,24,25],[36,37,43]]

SOURCES = {17: 150, 18: 150, 24: 150, 25: 150}
SINKS = distinct[1]

plotgraph.plot_graph_with_auto_curve_distinct(full_adjacency_list, N, P,distinct)

# 使用新函数求解
multi_result = ssp_multi.solve_multi_source_sink_with_super_nodes(
    edges_data=edge,
    sources=SOURCES,
    sinks=SINKS
)
cost = multi_result['total_cost']


print(cost)


if multi_result == 0:
    print("No solution found")
else:
    # 输出结果（与原有格式兼容）
    cost = multi_result['total_cost']

    print("\nMulti-source multi-sink solution via super nodes:")
    print(f"Status: {multi_result['status']}")
   # if multi_result['status'] == "Optimal":
    print(multi_result['status'] )
    print(f"Total Fixed Cost: {multi_result['total_cost']}")
    if multi_result['flow_details']:
        print("\nFlow Details (原始网络中的边):")
        for detail in multi_result['flow_details']:
            print(f"  Edge ({detail['from']} -> {detail['to']}): "
                  f"Fixed Cost={detail['cost']}, Flow={detail['flow']}, Capacity={detail['capacity']}")
import ga.graphalgorithm.fcnfp_multi as fcnfp_multi
import genaric.chrom2adjact as c2a
import genaric.plotgraph as plotgraph
import satnode.relative_position as relpos
import ga.graphalgorithm.adjact2weight as a2w

adjacency_list =     {0: {1, 6, 7}, 1: {0, 2, 15}, 2: {16, 1, 3}, 3: {9, 2, 4}, 4: {3, 12, 5}, 5: {19, 4, 6}, 6: {0, 5, 13}, 7: {0, 8, 20, 13}, 8: {9, 14, 7}, 9: {8, 10, 3, 23}, 10: {17, 11, 9}, 11: {18, 10, 12}, 12: {26, 11, 4, 13}, 13: {27, 12, 6, 7}, 14: {8, 28, 20, 15}, 15: {16, 1, 22, 14}, 16: {17, 2, 30, 15}, 17: {24, 16, 10, 18}, 18: {19, 25, 11, 17}, 19: {33, 18, 20, 5}, 20: {34, 19, 14, 7}, 21: {27, 29, 22}, 22: {23, 36, 21, 15}, 23: {24, 9, 37, 22}, 24: {17, 23, 25, 31}, 25: {32, 24, 18, 26}, 26: {40, 25, 27, 12}, 27: {41, 21, 26, 13}, 28: {34, 35, 29, 14}, 29: {43, 28, 21, 30}, 30: {16, 44, 29, 31}, 31: {24, 32, 38, 30}, 32: {25, 31, 33, 39}, 33: {32, 34, 19, 47}, 34: {48, 33, 20, 28}, 35: {41, 42, 28, 36}, 36: {50, 35, 37, 22}, 37: {36, 45, 38, 23}, 38: {39, 52, 37, 31}, 39: {32, 40, 53, 38}, 40: {41, 26, 46, 39}, 41: {40, 35, 27, 55}, 42: {56, 43, 48, 35}, 43: {49, 42, 44, 29}, 44: {43, 51, 45, 30}, 45: {59, 44, 37, 46}, 46: {40, 60, 45, 47}, 47: {46, 33, 48, 54}, 48: {34, 42, 62, 47}, 49: {57, 50, 43, 55}, 50: {49, 58, 51, 36}, 51: {50, 44, 52}, 52: {51, 53, 38}, 53: {52, 61, 54, 39}, 54: {55, 53, 47}, 55: {41, 54, 49}, 56: {57, 42, 62}, 57: {56, 49, 58}, 58: {57, 50, 59}, 59: {58, 60, 45}, 60: {59, 61, 46}, 61: {60, 53, 62}, 62: {48, 56, 61}}
N=7
P=9

full_adjacency_list = adjacency_list

inter_link_bandwidth = 50
intra_link_bandwidth = 100

cost =1

edge =a2w.adjacent2edge(full_adjacency_list,N,inter_link_bandwidth,intra_link_bandwidth,cost)

plotgraph.plot_graph_with_auto_curve(full_adjacency_list, N, P)


SOURCES = {17: 70, 18: 70,24:70,25:70}
SINKS = [31, 32,38,39]

# 使用新函数求解
multi_result = fcnfp_multi.solve_multi_source_sink_with_super_nodes(
    edges_data=edge,
    sources=SOURCES,
    sinks=SINKS
)
cost = multi_result['total_fixed_cost']
if multi_result == 0:
    print("No solution found")
else:
    # 输出结果（与原有格式兼容）
    cost = multi_result['total_fixed_cost']

    print("\nMulti-source multi-sink solution via super nodes:")
    print(f"Status: {multi_result['status']}")
    if multi_result['status'] == "Optimal":
        print(f"Total Fixed Cost: {multi_result['total_fixed_cost']}")
        if multi_result['flow_details']:
            print("\nFlow Details (原始网络中的边):")
            for detail in multi_result['flow_details']:
                print(f"  Edge ({detail['from']} -> {detail['to']}): "
                      f"Fixed Cost={detail['fixed_cost']}, Flow={detail['flow']}, Capacity={detail['capacity']}")
import ga.graphalgorithm.fcnfp_multi as fcnfp_multi
import genaric.chrom2adjact as c2a
import genaric.plotgraph as plotgraph
import satnode.relative_position as relpos
import ga.graphalgorithm.adjact2weight as a2w

import ga.graphalgorithm.mcmf.ssp_multi as ssp_multi
adjacency_list = {2: {1, 3, 12},
             0: {1, 9, 10},
             3: {2, 4, 13},
             6: {5, 7, 16},
             12: {11, 13, 32},
             13: {12, 14, 23},
             15: {14, 16, 35},
             19: {10, 18, 20},
             21: {20, 22, 41},
             23: {22, 24, 43},
             34: {33, 35, 54},
             38: {37, 39, 47},
             49: {40, 48, 69},
             50: {51, 59, 70},
             54: {53, 55, 74},
             56: {55, 57, 66},
             57: {56, 58, 77},
             62: {61, 63, 82},
             63: {62, 64, 83},
             66: {65, 67, 86},
             71: {70, 72, 91},
             72: {71, 73, 92},
             73: {72, 74, 93},
             74: {73, 75, 94},
             75: {74, 76, 95},
             76: {75, 77, 96},
             77: {76, 78, 97},
             88: {87, 89, 98},
             98: {88, 97, 99},
             1: {0, 2},
             9: {0, 8},
             4: {3, 5},
             5: {4, 6},
             7: {6, 8},
             8: {7, 9},
             10: {11, 19},
             11: {10, 12},
             14: {13, 15},
             16: {15, 17},
             17: {16, 18},
             18: {17, 19},
             20: {21, 29},
             29: {20, 28},
             22: {21, 23},
             24: {23, 25},
             25: {24, 26},
             26: {25, 27},
             27: {26, 28},
             28: {27, 29},
             30: {31, 39},
             31: {30, 32},
             39: {30, 38},
             32: {31, 33},
             33: {32, 34},
             35: {34, 36},
             36: {35, 37},
             37: {36, 38},
             40: {41, 49},
             41: {40, 42},
             42: {41, 43},
             43: {42, 44},
             44: {43, 45},
             45: {44, 46},
             46: {45, 47},
             47: {46, 48},
             48: {47, 49},
             51: {50, 52},
             59: {50, 58},
             52: {51, 53},
             53: {52, 54},
             55: {54, 56},
             58: {57, 59},
             60: {61, 69},
             61: {60, 62},
             69: {60, 68},
             64: {63, 65},
             65: {64, 66},
             67: {66, 68},
             68: {67, 69},
             70: {71, 79},
             79: {70, 78},
             78: {77, 79},
             80: {81, 89},
             81: {80, 82},
             89: {80, 88},
             82: {81, 83},
             83: {82, 84},
             84: {83, 85},
             85: {84, 86},
             86: {85, 87},
             87: {86, 88},
             90: {91, 99},
             91: {90, 92},
             99: {90, 98},
             92: {91, 93},
             93: {92, 94},
             94: {93, 95},
             95: {94, 96},
             96: {95, 97},
             97: {96, 98}}
N=10
P=10

full_adjacency_list = adjacency_list

inter_link_bandwidth = 50
intra_link_bandwidth = 100

cost =1

edge =a2w.adjacent2edge(full_adjacency_list,N,inter_link_bandwidth,intra_link_bandwidth,cost)



distinct = [[2, 12, 88, 98], [58]]


SOURCES =  {2: 150, 12: 150, 88: 150, 98: 150}
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



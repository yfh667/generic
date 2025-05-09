import ga.graphalgorithm.fcnfp_multi as fcnfp_multi
import genaric.chrom2adjact as c2a
import genaric.plotgraph as plotgraph
import satnode.relative_position as relpos
import ga.graphalgorithm.adjact2weight as a2w

adjacency_list = {
    0: {7},
    1: {8},
    2: {9},
    3: {10},
    4: {11},
    5: {12},
    6: {13},
    7: {14},
    8: {15},
    9: {16},
    10: {17},
    11: {18},
    12: {19},
    13: {20},
    14: {21},
    15: {29},
    16: {30},
    17: {24},
    18: {25},
    19: {26},
    20: {27},
    21: {28},
    22: {-1},
    23: {-1},
    24: {31},
    25: {32},
    26: {33},
    27: {34},
    28: {35},
    29: {36},
    30: {37},
    31: {38},
    32: {39},
    33: {40},
    34: {41},
    35: {42},
    36: {43},
    37: {44},
    38: {45},
    39: {46},
    40: {47},
    41: {48},
    42: {49},
    43: {50},
    44: {51},
    45: {52},
    46: {53},
    47: {54},
    48: {55},
    49: {56},
    50: {57},
    51: {58},
    52: {59},
    53: {60},
    54: {61},
    55: {62},
    56: {-1},
    57: {-1},
    58: {-1},
    59: {-1},
    60: {-1},
    61: {-1},
    62: {-1},
}
N=7
P=9

full_adjacency_list = c2a.full_adjacency_list(adjacency_list,N,P)

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
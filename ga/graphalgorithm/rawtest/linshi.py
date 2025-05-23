import ga.graphalgorithm.fcnfp_multi as fcnfp_multi
import genaric.chrom2adjact as c2a
import genaric.plotgraph as plotgraph
import satnode.relative_position as relpos
import ga.graphalgorithm.adjact2weight as a2w
import os
import draw.snapshotf_romxml as snapshotf_romxml

import random
import math
import matplotlib.pyplot as plt
import random
import numpy as np
from networkx.classes import neighbors
import genaric.chrom2adjact as c2a
import ga.graphalgorithm.fcnfp_multi as fcnfp_multi
import genaric.chrom2adjact as c2a
import genaric.plotgraph as plotgraph
import satnode.relative_position as relpos
import ga.graphalgorithm.adjact2weight as a2w
import ga.initiallink as initiallink
# 示例用法
import genaric.adjact2chrom as a2c
import genaric.dijstra as dij
import genaric.chrom2adjact as c2a
import genaric.plotgraph as plotgraph

target_time_step = 0
dummy_file_name = "E:\code\data\station_visible_satellites.xml"
# Extract and print the satellite lists for each region from the file
print(f"\nExtracting data for time step {target_time_step} from '{dummy_file_name}'...")
region_satellite_groups = snapshotf_romxml.extract_region_satellites_from_file(dummy_file_name, target_time_step)

print(f"Satellite groups for time step {target_time_step}:")
for i, satellite_list in enumerate(region_satellite_groups):
    print(f"Region {i}: {satellite_list}")
distinct =  []


integer_satellite_groups_0 = [int(satellite_id) for satellite_id in region_satellite_groups[0]]

integer_satellite_groups_1 = [int(satellite_id) for satellite_id in region_satellite_groups[1]]


distinct.append(integer_satellite_groups_0)
distinct.append(integer_satellite_groups_1)
#
# adjacency_list = {
#     0: {7},
#     1: {8},
#     2: {9},
#     3: {10},
#     4: {11},
#     5: {12},
#     6: {13},
#     7: {14},
#     8: {15},
#     9: {16},
#     10: {17},
#     11: {18},
#     12: {19},
#     13: {20},
#     14: {21},
#     15: {29},
#     16: {30},
#     17: {24},
#     18: {25},
#     19: {26},
#     20: {27},
#     21: {28},
#     22: {-1},
#     23: {-1},
#     24: {31},
#     25: {32},
#     26: {33},
#     27: {34},
#     28: {35},
#     29: {36},
#     30: {37},
#     31: {38},
#     32: {39},
#     33: {40},
#     34: {41},
#     35: {42},
#     36: {43},
#     37: {44},
#     38: {45},
#     39: {46},
#     40: {47},
#     41: {48},
#     42: {49},
#     43: {50},
#     44: {51},
#     45: {52},
#     46: {53},
#     47: {54},
#     48: {55},
#     49: {56},
#     50: {57},
#     51: {58},
#     52: {59},
#     53: {60},
#     54: {61},
#     55: {62},
#     56: {-1},
#     57: {-1},
#     58: {-1},
#     59: {-1},
#     60: {-1},
#     61: {-1},
#     62: {-1},
# }


chrom =[37, 36, 38, 39, 40, 41, 78, 44, 43, 81, 45, 46, 48, 49, 86, 51, 52, 89, 90, 56, 57, 93, 58, 59, 61, 60, 62, 63, 64, 66, 67, 103, 104, 69, 70, 71, 107, 72, 75, 76, 112, 77, 114, 80, 79, 117, 83, 84, 120, 121, 85, 123, 87, 88, 91, 127, 128, 94, 95, 131, 132, 98, 134, 135, 100, 137, 138, 139, 140, 106, 105, 143, 108, 109, 146, 111, 148, 113, 150, 151, 115, 153, 154, 155, 156, 122, 158, 159, 125, 126, 162, 163, 164, 165, 129, 130, 133, 169, 170, 171, 136, 173, 174, 175, 141, 142, 178, 179, 145, 181, 182, 147, 184, 149, 186, 187, 188, 152, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 166, 202, 167, 168, 205, 206, 207, 208, 209, 210, 176, 177, 213, 214, 144, 180, -1, 183, 219, 185, 221, 222, 223, 189, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 201, 239, 203, 241, 242, 243, 244, 245, 211, 212, 248, 249, 215, 251, 252, 217, 218, 220, 256, 257, 258, 224, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 238, 274, 240, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 250, 288, 253, 254, 291, 255, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 273, 309, 275, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 287, 323, -1, 289, 290, 292, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 308, 344, 345, 346, 310, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 322, 359, 324, 325, 362, 326, 327, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 343, 379, 380, 381, 382, 347, 384, 385, 386, 387, 388, 389, 390, 391, 392, 358, 394, 395, 360, 361, 398, 364, 363, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 383, 419, 420, 421, 422, 423, 424, 425, 426, 427, 393, 429, 430, 431, 432, 397, 434, 400, 399, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 396, 468, 433, 470, 435, 436, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 469, 505, 471, 472, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, -1, -1, 539, 540, 504, 506, 543, 507, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 537, 538, 575, 541, 577, 578, 544, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 573, 574, 611, 576, 613, 614, 615, 579, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, -1, -1, 643, -1, 609, 646, 647, 612, -1, -1, 616, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 641, 642, -1, 644, 645, -1, -1]

N=36
P=18

base_adjacency_list = c2a.base_chrom2adjacent(chrom, N,P)

adjacency_list = c2a.full_adjacency_list(base_adjacency_list,N,P)


full_adjacency_list = c2a.full_adjacency_list(adjacency_list,N,P)

inter_link_bandwidth = 50
intra_link_bandwidth = 100

cost =1

edge =a2w.adjacent2edge(full_adjacency_list,N,inter_link_bandwidth,intra_link_bandwidth,cost)

plotgraph.plot_graph_with_auto_curve_distinct(full_adjacency_list, N, P,distinct)

demand=70
# SOURCES = {17: 70, 18: 70,24:70,25:70}
# SINKS = [31, 32,38,39]
SOURCES = {node_id: demand for node_id in distinct[0]}

SINKS = distinct[1]
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
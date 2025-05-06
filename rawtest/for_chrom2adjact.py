

import graph_ga.chrom2adjact as c2a

import graph_ga.plotgraph as plotgraph

N = 4
P=4

chrom =[5, 9, 7, 6, 8, -1, 10, 11, 12, 14, 13, 15]
base_adjacency_list = c2a.base_chrom2adjacent(chrom, N,P)

adjacency_list = c2a.full_adjacency_list(base_adjacency_list,N,P)

plotgraph.plot_graph_with_auto_curve(adjacency_list, N, P)
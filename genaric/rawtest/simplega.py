

import graph_ga.ga.initiallink as initiallink
import graph_ga.chrom2adjact as c2a

import graph_ga.plotgraph as plotgraph

def initialize_individual(N,P):
    individual = [-1]*(P-1)*N
    left_link =[0]*(P)*N
    for i in range(P-1):
        nowlink = initiallink.initialize_links(N)
        for j in range(len(nowlink)):
         #   individual[j] = nowlink[j]
            nowlink[j] = (i + 1) * N + nowlink[j]




        individual[i*N:(i+1)*N]=nowlink

    return individual









N = 4
P=4
chrom = initialize_individual(N,P)

base_adjacency_list = c2a.base_chrom2adjacent(chrom, N,P)

adjacency_list = c2a.full_adjacency_list(base_adjacency_list,N,P)

plotgraph.plot_graph_with_auto_curve(adjacency_list, N, P)
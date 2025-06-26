

maxhop=30
import genaric2.adj2adjacylist as adj2adjaclist

import ga.graphalgorithm.adjact2weight as a2w

import utilis.adjacency as adjacency

import genaric2.action_table as action_table


import score.BFS.BFS as BFS



def decode_chromosome(P, N, T,chromosome):
    connection_list = action_table.action_map2_shanpshots(chromosome, P, N, T)
    adjacency_list = adj2adjaclist.adj2adjaclist(connection_list, N, P, T)


    return adjacency_list


def calculate_fitness_onedividual(individual,regions_to_color,inter_link_bandwidth,intra_link_bandwidth,cost,P, N, T ):
    adjacency_list = decode_chromosome(P, N, T, individual)

    indictor = 0
    seq = []
    for i in range(2, T ):

        onecost = cauculate_one_snap_fitness(adjacency_list[i], N, inter_link_bandwidth, intra_link_bandwidth, cost,
                                             regions_to_color[i])
        if onecost == -1:
            print("No solution found")
        else:
            indictor = indictor + onecost
            seq.append(onecost)

    return indictor, seq


def calculate_fitness_test(individual,regions_to_color,inter_link_bandwidth,intra_link_bandwidth,cost,P, N, T ):
    adjacency_list = decode_chromosome(P, N, T, individual)

    indictor = 0
    seq = []
    for i in range(2, T ):
        if i==20:
            print(1)
        onecost = cauculate_one_snap_fitness(adjacency_list[i], N, inter_link_bandwidth, intra_link_bandwidth, cost,
                                             regions_to_color[i])
        if onecost == -1:
            print("No solution found")
        else:
            indictor = indictor + onecost
            seq.append(onecost)

    return indictor, seq

def cauculate_one_snap_fitness(adjacency_list, N, inter_link_bandwidth, intra_link_bandwidth, cost,distinct):
    edge = a2w.adjacent2edge(adjacency_list, N, inter_link_bandwidth, intra_link_bandwidth, cost)

  #  distinct = regions_to_color[i]
   # path = BFS.shortest_path(adjacency_list, start, end)
    complet_adjacency_list  = adjacency.complete_undirected_graph(adjacency_list)

    start=distinct[0]
    end=distinct[1]
    all_hops = 0
    hops_num =0
    flag=1
   # hops_seq=[]
    for source in start:
        for destination in end:
            path = BFS.shortest_path(complet_adjacency_list, source, destination)
            if path:
                hops_len = len(path)-1
                all_hops += hops_len
             #   hops_seq.append(hops_len)
                hops_num+=1

            else:
                # if there is no path, we could just think this hop is 30 hops.
                all_hops += maxhop
              #  hops_seq.append(maxhop)
                hops_num+=1
            #    flag=-1
    if flag==1:
     onecost=all_hops/hops_num
    else:
        onecost=-1

    return onecost


def calculate_fitness_min(adjacency_list,regions_to_color,intra_link_bandwidth, inter_link_bandwidth,cost,P,N,T):
    indictor = 0
    seq = []
    for i in range(2, T ):

        onecost = cauculate_one_snap_fitness(adjacency_list[i], N, inter_link_bandwidth, intra_link_bandwidth, cost,
                                             regions_to_color[i])
        if onecost == -1:
            print("No solution found")
        else:
            indictor = indictor + onecost
            seq.append(onecost)

    return indictor,seq




def fitness_function(population,regions_to_color, intra_link_bandwidth, inter_link_bandwidth, cost,P,N,T):
    indictors = []
    decoded_values=[]
    for ind in population:
        adjacency_list = decode_chromosome(P,N,T,ind)
        decoded_values.append(adjacency_list)
    indictors_seq=[]
   # decoded_values = [decode_chromosome(P,N,T,ind) for ind in population]
    for adjacency_list in decoded_values:




        indictor, seq= calculate_fitness_min(adjacency_list, regions_to_color, intra_link_bandwidth, inter_link_bandwidth, cost, P, N,         T)


        indictors.append(indictor)
        indictors_seq.append(seq)



    return indictors,indictors_seq

import genaric2.adj2adjacylist as adj2adjaclist
import genaric2.writetoxml as writetoxml
# import genaric.plotgraph as plotgraph
import ga.graphalgorithm.adjact2weight as a2w
import draw.snapshotf_romxml as snapshotf_romxml
import utilis.adjacency as adjacency
import genaric2.distinct_initial as distinct_initial
import genaric2.action_table as action_table

import ga.graphalgorithm.mcmf.ssp_multi as ssp_multi
import score.BFS.BFS as BFS
inter_link_bandwidth = 50
intra_link_bandwidth = 100
cost =1
demand= 30


maxhop=30





def decode_chromosome(P, N, T,chromosome):
    connection_list = action_table.action_map2_shanpshots(chromosome, P, N, T)
    adjacency_list = adj2adjaclist.adj2adjaclist(connection_list, N, P, T)


    return adjacency_list


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



def fitness_function(P,N,T,population,regions_to_color):
    indictors = []
    decoded_values=[]
    for ind in population:
        adjacency_list = decode_chromosome(P,N,T,ind)
        decoded_values.append(adjacency_list)
    indictors_seq=[]
   # decoded_values = [decode_chromosome(P,N,T,ind) for ind in population]
    for adjacency_list in decoded_values:
        indictor=0
        seq = []
        for i in range(3,T-2):


           onecost= cauculate_one_snap_fitness(adjacency_list[i], N, inter_link_bandwidth, intra_link_bandwidth, cost, regions_to_color[i])
           if onecost == -1:
               print("No solution found")
           else:
               indictor=indictor+onecost
               seq.append(onecost)



        indictors.append(indictor)
        indictors_seq.append(seq)



    return indictors,indictors_seq

if __name__ == '__main__':
    ##1.我们先获得原始的卫星分布数据，主要是热点区域的拓扑序列数据
    N = 10
    P=10
    start_ts = 1500
    end_ts = 1523

    dummy_file_name = "E:\\code\\data\\station_visible_satellites_100_test.xml"



    regions_to_color = {}
    # Iterate over the time steps
    region_satellite_groups= snapshotf_romxml.extract_region_satellites_from_file(dummy_file_name, start_ts, end_ts)
    target_time_step = len(region_satellite_groups)
    T = target_time_step
    for i in range(len(region_satellite_groups)):
            region_satellite_group = [[int(point) for point in region] for region in region_satellite_groups[i]]
            u =region_satellite_group[0]
            v = region_satellite_group[3]
            o = [u,v]
            regions_to_color[i] = o  # Corrected append to dictionary assignment

    ##2. 完成对热点区域预建链安排

    T = target_time_step
    setuptime=2
    base =distinct_initial.distinct_initial(P,N,T,setuptime,regions_to_color)

    individual1 = writetoxml.xml_to_nodes("E:\\code\\data\\1\\individual2.xml")

    fitness,indictors_seq = fitness_function(P, N, T, [individual1],regions_to_color)
    print("1")
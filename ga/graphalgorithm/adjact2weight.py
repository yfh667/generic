


import satnode.relative_position as relpos

def adjacent2edge(full_adjacency_list,N,inter_link_bandwidth,intra_link_bandwidth,cost):

    # inter_link_bandwidth = 50
    # intra_link_bandwidth = 100
    #
    # cost =1
    edge = []
    for i in full_adjacency_list:
        neighbors = full_adjacency_list[i]


        for j in neighbors:
            if relpos.intra_or_inter(N,i,j):## intra
                edge.append((i,j,cost,intra_link_bandwidth))
            edge.append((i,j,cost,inter_link_bandwidth))


    return edge
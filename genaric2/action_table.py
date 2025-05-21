def action_map2_shanpshots(nodes, P, N, T):
    adj_list_array = [[] for _ in range(T)]
    for i in range(P):
        for t in range(T):
            for j in range(N):
                node = nodes[(i, j, t)]
                if node.state == 0:
                    temp = 1
                    nownode = i*N + j
                    neighbor_id = node.rightneighbor
                    neighbor_node =  neighbor_id[0]*N+neighbor_id[1]
                    while(1):

                        if t+temp>=T:
                            break
                        if nodes[(i, j, t+temp)].state == 1:
                            pass
                        elif nodes[(i, j, t+temp)].state == 2:
                            neighbor_id = nodes[(i, j, t+temp)].rightneighbor
                            adj_list_array[t+temp].append((nownode,neighbor_node))
                        else:
                            break
                        temp+=1
    return adj_list_array



def action_map2connecttion_list(nodes, P, N, T):
    connections_list = []
    for i in range(P):
        for t in range(T):
            for j in range(N):
                node = nodes[(i, j, t)]
                if node.state == 0:
                    neighbor_id = node.rightneighbor
                    links = [(i, j, t),neighbor_id]
                    connections_list.append(links)

    return connections_list
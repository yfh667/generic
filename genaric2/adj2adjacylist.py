from networkx.readwrite.json_graph import adjacency
from collections import defaultdict


def full_adjacency_list(adjacency_list, P, N):
    # First convert all neighbor collections to sets if they aren't already
    for node in adjacency_list:
        if not isinstance(adjacency_list[node], set):
            adjacency_list[node] = set(adjacency_list[node])

    # Create a copy to avoid modifying during iteration
    original_nodes = list(adjacency_list.keys())


    for i in range(P * N):


        if i  in original_nodes:

            for neighbor in list(adjacency_list[node]):
                if neighbor not in adjacency_list:
                    adjacency_list[neighbor] = set()
                adjacency_list[neighbor].add(node)
        x = i // N  # current row
        y = i % N  # current column



        # up neighbor (same row, next column)
        up_neighbor = x * N + (y + 1) % N
        adjacency_list[i].add(up_neighbor)

        adjacency_list[up_neighbor].add(i)

        # down neighbor (same row, previous column)
        down_neighbor = x * N + (y - 1 + N) % N
        adjacency_list[i].add(down_neighbor)

        adjacency_list[down_neighbor].add(i)

    # for node in original_nodes:
    #     # Add reverse connections for existing edges
    #     for neighbor in list(adjacency_list[node]):
    #         if neighbor not in adjacency_list:
    #             adjacency_list[neighbor] = set()
    #         adjacency_list[neighbor].add(node)
    #
    #     # Calculate left and right neighbors (same row)
    #
    #     x = node // N  # current row
    #     y = node % N  # current column
    #
    #
    #     if x==3 and y==2:
    #         print(1)
    #
    #     # up neighbor (same row, next column)
    #     up_neighbor = x * N + (y + 1) % N
    #     adjacency_list[node].add(up_neighbor)
    #
    #     adjacency_list[up_neighbor].add(node)
    #
    #     # down neighbor (same row, previous column)
    #     down_neighbor = x * N + (y - 1 + N) % N
    #     adjacency_list[node].add(down_neighbor)
    #
    #     adjacency_list[down_neighbor].add(node)

    return adjacency_list


def adj2adjaclist(adj,P,N,T):
    full_adjacncy=[]
    for i in range(T):
        adjacncy_linshi = defaultdict(set)
        for pair in  adj[i]:
            adjacncy_linshi[pair[0]].add(pair[1])
        adjacency_list = full_adjacency_list(adjacncy_linshi, P, N)
        full_adjacncy.append(adjacency_list)
    return full_adjacncy
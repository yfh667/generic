
def maintenance_mutate(coordinate,chromosome,P, N, T):
    x = coordinate[0]
    y = coordinate[1]
    t = coordinate[2]
    now_right_neighbor = chromosome[coordinate].rightneighbor


    start,end= find_next_setup_time(coordinate, chromosome, P, N, T)

    for i in range(start,end+1):
        chromosome[(x, y, i)].state = 2
        chromosome[(x, y, i)].rightneighbor = (now_right_neighbor[0], now_right_neighbor[1], i)
        chromosome[(now_right_neighbor[0], now_right_neighbor[1], i)].leftneighbor = (x, y, i)







def find_next_setup_time(coordinate,chromosome ,P, N, T):
    # 检查当前节点是否已经有手动设置的链路
    # Flag for finding the next establishment
    x = coordinate[0]
    y = coordinate[1]
    t = coordinate[2]
    start_time = 0

    # Loop through time steps to find the next establishment
    while t + 1 < T :
        t += 1
        if chromosome[(x, y, t)].state == 0:  # Node is setting up the link
            break
        elif chromosome[(x, y, t)].state == 2:
            continue
        elif chromosome[(x, y, t)].state == -1:
            start_time = t
            continue

    # The time when the node starts setting up the link

    # The time when the node stops setting up the link
    down_time = t - 1

    return start_time, down_time

import genaric2.mutation.basic_mutate_func as basic_mutate_func

def maintenance_mutate(coordinate,chromosome,P, N, T,setuptime):
    x = coordinate[0]
    y = coordinate[1]
    t = coordinate[2]
    now_right_neighbor = chromosome[coordinate].rightneighbor


    start,end= find_next_setup_time(coordinate, chromosome, P, N, T)

    for i in range(start,end+1):

        if chromosome[(x, y, i)].state==0:
            # this is to delet the raw rightneighbor's left neighbor
            right_neighbor = chromosome[x, y, i].rightneighbor
            x2,y2,t2=right_neighbor
            for j in range(i,i+setuptime):
               chromosome[x2, y2, j].leftneighbor=None


        elif chromosome[(x, y, i)].state==2:
            right_neighbor = chromosome[x, y, i].rightneighbor
            basic_mutate_func.clear_leftneighbor(right_neighbor, chromosome)




        # then we need modify the objetct node
        objetct_node = (now_right_neighbor[0], now_right_neighbor[1], i)

        objetct_node_leftneighbor = chromosome[objetct_node].leftneighbor

        if objetct_node_leftneighbor:
            basic_mutate_func.clear_state(objetct_node_leftneighbor, chromosome, P, N, T, setuptime)


        chromosome[(x, y, i)].state = 2

        chromosome[(x, y, i)].rightneighbor = (now_right_neighbor[0], now_right_neighbor[1], i)

        chromosome[(now_right_neighbor[0], now_right_neighbor[1], i)].leftneighbor = (x, y, i)







def find_next_setup_time(coordinate,chromosome ,P, N, T):
    # 检查当前节点是否已经有手动设置的链路
    # Flag for finding the next establishment
    x = coordinate[0]
    y = coordinate[1]
    t = coordinate[2]
    start_time = t


    flag=1
    flag2=1
    # Loop through time steps to find the next establishment
    while t + 1 < T and flag :
        t += 1
        if chromosome[(x, y, t)].state == 0:  # Node is setting up the link
            if flag2:
                start_time = t
                flag2 = 0
            flag=flag-1


        elif chromosome[(x, y, t)].state == 2:
            continue
        elif chromosome[(x, y, t)].state != 1 and chromosome[(x, y, t)].state != 2 :
            if flag2:
                start_time = t
                flag2 = 0


    # The time when the node starts setting up the link

    # The time when the node stops setting up the link
    down_time = t - 1

    return start_time, down_time

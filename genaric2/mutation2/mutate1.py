import random
import genaric2.mutation.basic_mutate_func as basic_mutate_func
import genaric2.distinct_initial as distinct_initial



def establishment_mutate(coordinate,chromosome,P, N, T,setuptime):
    x = coordinate[0]
    y = coordinate[1]
    t = coordinate[2]

    # 搜索候选右邻居
    candidates = []
    col=x
    row=y
    next_col = col + 1

    if t + setuptime < T and next_col < P:

        for i in range(3):
            candi_node= (next_col, (row - 1+i) % N, t + setuptime)


            leftneighbor=chromosome[candi_node].leftneighbor


            if  leftneighbor:
                if chromosome[leftneighbor].asc_nodes_flag==1:
                    continue
                else:
                    candidates+=[candi_node]
            else:
                # here we need check whether the candi_node last few time  has been in setup or use

                candidates += [candi_node]



        # candidates += [
        #     (next_col, (row - 1) % N, t + setuptime),
        #     (next_col, row % N, t + setuptime),
        #     (next_col, (row + 1) % N, t + setuptime),
        # ]

        if next_col + 1 < P:
            candi_node = (next_col + 1, row, t + setuptime)
            leftneighbor=chromosome[candi_node].leftneighbor
            if  leftneighbor:
                if chromosome[leftneighbor].asc_nodes_flag==1:
                  pass
                else:
                    candidates+=[candi_node]

            #
            # candidates.append((next_col + 1, row, t + setuptime))

    # now_right_neighbor = chromosome[coordinate].rightneighbor

    chosen_righbor = random.choice(candidates)

# we first need delet the raw chosen_righbor data
    leftneighbor = chromosome[chosen_righbor].leftneighbor

    if leftneighbor:

        basic_mutate_func.clear_state2(leftneighbor, chromosome, P, N, T, setuptime)



    if chromosome[coordinate].asc_nodes_flag==1:
        start, end = find_next_setup_time(coordinate, chromosome, P, N, T)
    else:

        start,end= find_next_setup_time_for_hot_nodes(coordinate, chromosome, P, N, T)


# the we need clear all the link state for the now  node

    if start==-1 and end==-1:
        return


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

        start_node_id = x*N+y
        end_node_id = chosen_righbor[0]*N+chosen_righbor[1]

     #   distinct_initial.initialize_establish(N, T, chromosome, start_node_id, end_node_id, t, setuptime)

        #





def find_next_setup_time_for_hot_nodes(coordinate,chromosome ,P, N, T):
    # 检查当前节点是否已经有手动设置的链路
    # Flag for finding the next establishment
    x = coordinate[0]
    y = coordinate[1]
    t = coordinate[2]
    start_time = t


    flag=2
    flag2=1
    # Loop through time steps to find the next establishment
    start_time, down_time=-1,-1

# if it is the hotnodes we  can't over write the link

    right_neighbor=chromosome[coordinate].rightneighbor

    if chromosome[right_neighbor].asc_nodes_flag==1:
        pass


    else:

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

        if t==T-1:
            down_time=t
        else:
         down_time = t - 1

        if start_time==coordinate[2]:
            #imply that there is no next settp time
            start_time=down_time


    return start_time, down_time


def find_next_setup_time(coordinate,chromosome ,P, N, T):
    # 检查当前节点是否已经有手动设置的链路
    # Flag for finding the next establishment
    x = coordinate[0]
    y = coordinate[1]
    t = coordinate[2]
    start_time = t


    flag=2
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

    if t==T-1:
        down_time=t
    else:
     down_time = t - 1

    if start_time==coordinate[2]:
        #imply that there is no next settp time
        start_time=down_time


    return start_time, down_time

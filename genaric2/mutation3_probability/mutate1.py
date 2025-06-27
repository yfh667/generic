import random
import genaric2.mutation.basic_mutate_func as basic_mutate_func
import genaric2.distinct_initial as distinct_initial



def establishment_mutate(coordinate,chromosome,distinct,P, N, T,setuptime,test=0):
    x = coordinate[0]
    y = coordinate[1]
    t = coordinate[2]

    # 搜索候选右邻居
    candidates = []
    col=x
    row=y
    next_col = col + 1


    if len(distinct)==0:
        #it is simple
        if t + setuptime < T and next_col < P:
            for i in range(3):
                candi_node= (next_col, (row - 1+i) % N, t + setuptime)
                candidates += [candi_node]
        if next_col + 1 < P:
            candi_node = (next_col + 1, row, t + setuptime)
            candidates += [candi_node]

    else:
        # here we need check the distinct
        if t + setuptime < T and next_col < P:
            for i in range(3):
                candi_node = (next_col, (row - 1 + i) % N, t + setuptime)

                if candi_node not in distinct:
                    candidates += [candi_node]


        if next_col + 1 < P:
            candi_node = (next_col + 1, row, t + setuptime)
            if candi_node not in distinct:
                candidates += [candi_node]


    for candi_node in candidates:
        if chromosome[candi_node].asc_nodes_flag==1:
            if chromosome[candi_node].leftneighbor:
                leftneighbor = chromosome[candi_node].leftneighbor
                if chromosome[leftneighbor].asc_nodes_flag == 1:
                    candidates.remove(candi_node)



    max_importance= 0
    max_index = 0

    for i in range(len(candidates)):
        if chromosome[candidates[i]].importance > max_importance:
            max_importance = chromosome[candidates[i]].importance
            max_index = i



    if test:
        chosen_righbor=(7,8,6)
    else:

       # chosen_righbor = random.choice(candidates)
        chosen_righbor = candidates[max_index]

    # here we test
    start, end = find_next_setup_time(coordinate, chromosome, P, N, T)
#first we need delet the  (x y t) raw link


# we first need delet the raw chosen_righbor data


    afect_region=[]
    for i in range(start,end+1):
        afect_region.append((chosen_righbor[0],chosen_righbor[1],i))
        # we need delete the raw (x,y,i) rightneighbor 's leftneighbor information

        if chromosome[(x,y,i)].state==0:

            rightbor = chromosome[(x, y, i)].rightneighbor

            x2=rightbor[0]
            y2=rightbor[1]

            for j in range(i,i+setuptime):
                chromosome[(x2,y2,j)].leftneighbor = None
        else:
            rightbor=chromosome[(x,y,i)].rightneighbor
            if rightbor:
                chromosome[rightbor].leftneighbor=None




    for coord in afect_region:

        leftneighbor = chromosome[coord].leftneighbor

        if leftneighbor:


            basic_mutate_func.clear_state2(leftneighbor, chromosome, P, N, T, setuptime)








    #
    # for i in range(start,end+1):
    start_node_id = x*N+y
    end_node_id = chosen_righbor[0]*N+chosen_righbor[1]
    distinct_initial.initialize_establish_lifecycle(N, T, chromosome, start_node_id, end_node_id, t, end,setuptime)




    # print(1)
        #





def find_next_setup_time(coordinate,chromosome ,P, N, T):
    # 检查当前节点是否已经有手动设置的链路
    # Flag for finding the next establishment
    x = coordinate[0]
    y = coordinate[1]
    t = coordinate[2]
    start_time = t


    flag=1

    # Loop through time steps to find the next establishment
    while t + 1 < T and flag :
        t += 1
        if chromosome[(x, y, t)].state == 0:  # Node is setting up the link

            flag=flag-1


        elif chromosome[(x, y, t)].state == 2:
            continue



    # The time when the node starts setting up the link

    # The time when the node stops setting up the link

    if t==T-1:
        down_time=t
    else:
     down_time = t - 1




    return start_time, down_time

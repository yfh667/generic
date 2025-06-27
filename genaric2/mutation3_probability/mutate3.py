import genaric2.mutation.basic_mutate_func as basic_mutate_func
import random

import random
import genaric2.mutation.basic_mutate_func as basic_mutate_func
import genaric2.distinct_initial as distinct_initial


def 	disconenct_mutate(coordinate,chromosome,distinct,P, N, T,setuptime,test=0):


    x = coordinate[0]
    y = coordinate[1]
    t = coordinate[2]

    # 搜索候选右邻居
    candidates = []
    col = x
    row = y
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
        chosen_righbor=(9, 8, 2)
    else:

       # chosen_righbor = random.choice(candidates)
        if len(candidates)==0:

            return None


        chosen_righbor = candidates[max_index]




    start,end= find_next_setup_time(coordinate, chromosome, P, N, T)

    # we first need delet the raw chosen_righbor data

    if end-start < setuptime:
        return None

    afect_region = []
    for i in range(start, end + 1):
        afect_region.append((chosen_righbor[0], chosen_righbor[1], i))
        # we need delete the raw (x,y,i) rightneighbor 's leftneighbor information

        if chromosome[(x, y, i)].state == 0:

            rightbor = chromosome[(x, y, i)].rightneighbor

            x2 = rightbor[0]
            y2 = rightbor[1]

            for j in range(i, i + setuptime):
                chromosome[(x2, y2, j)].leftneighbor = None
        else:
            rightbor = chromosome[(x, y, i)].rightneighbor
            if rightbor:
                chromosome[rightbor].leftneighbor = None

    for coord in afect_region:

        leftneighbor = chromosome[coord].leftneighbor

        if leftneighbor:
            basic_mutate_func.clear_state2(leftneighbor, chromosome, P, N, T, setuptime)

    #
    # for i in range(start,end+1):
    start_node_id = x * N + y
    end_node_id = chosen_righbor[0] * N + chosen_righbor[1]
    distinct_initial.initialize_establish_lifecycle(N, T, chromosome, start_node_id, end_node_id, t, end, setuptime)

    return  chosen_righbor


def find_next_setup_time(coordinate,chromosome ,P, N, T):
    # 检查当前节点是否已经有手动设置的链路
    # Flag for finding the next establishment
    x = coordinate[0]
    y = coordinate[1]
    t = coordinate[2]
    start_time = t


    # Loop through time steps to find the next establishment
    while t + 1 < T  :
        t += 1
        if chromosome[(x, y, t)].state == 0:  # Node is setting up the link
            break


    if t+1==T:
        down_time=t
    else:
        down_time = t - 1

    return start_time, down_time





def disconnect2disconect( coordinate, chromosome, P, N, T,setuptime):




    x, y, t = coordinate

    # If the node at the given coordinate is setting up the link (state == 0)

    # Flag for finding the next establishment
    flag = 2
    # Loop through time steps to find the next establishment
    while t + 1 < T and flag:
        t += 1
        if chromosome[(x, y, t)].state == 0:  # Node is setting up the link
            flag -= 1  # Decrease flag to exit loop

    # The time when the node stops setting up the link
    down_time = t - 1

    for i in range(coordinate[2],down_time+1):
        chromosome[(x, y, i)].state = -1
        right_neighbor = chromosome[x, y, i].rightneighbor
        basic_mutate_func.clear_leftneighbor(right_neighbor, chromosome)

        chromosome[x, y, i].rightneighbor = None




#
# def disconnect2establishment(    start,end,coordinate, chromosome, P, N, T,setuptime):
#     x = coordinate[0]
#     y = coordinate[1]
#     t = coordinate[2]
#     #here in fact ,start=t+1
#     duration = end - start+1
#     if duration+1 <= setuptime:
#         return
#     else:
#         col=x
#         row=y
#         candidates = []
#         next_col = col + 1
#
#         if t + setuptime < T and next_col < P:
#             candidates += [
#                 (next_col, (row - 1) % N, t + setuptime),
#                 (next_col, row % N, t + setuptime),
#                 (next_col, (row + 1) % N, t + setuptime),
#             ]
#
#             if next_col + 1 < P:
#                 candidates.append((next_col + 1, row, t + setuptime))
#         chosen = random.choice(candidates)
#
#
#
#
#         chromosome[(x, y, start-1)].state = 0
#         chromosome[(x, y, start-1)].rightneighbor =  chosen
#
#
#         ##first we need check the chosen,we need reserved the time for  the setup ,we need check the chosen up
#
#         #1  1
#         #1  1:so we may distory this state
#         #1  1:here we need to reserved
#         #1  1:
#         #1  chose
#
#         for i in range(setuptime):
#             # we check
#             cordinate = (chosen[0], chosen[1], t+i)
#             left_neighbor = chromosome[cordinate].leftneighbor
#             if left_neighbor == None:
#                 pass
#             else:
#                 if chromosome[left_neighbor].state == 2:
#                     chromosome[left_neighbor].state=-1
#                     chromosome[left_neighbor].rightneighbor=None
#
#                     chromosome[cordinate].leftneighbor=None
#
#
#                 elif chromosome[left_neighbor].state == 1 or chromosome[left_neighbor].state == 0:
#                     basic_mutate_func.clear_state(left_neighbor, chromosome, P, N, T, setuptime)
#                     chromosome[cordinate].leftneighbor = None
#
#         # then need handling conflicts
#         #
#         if  chromosome[ chosen].leftneighbor!=None:
#             leftneighbor = chromosome[ chosen].leftneighbor
#             basic_mutate_func.clear_state(leftneighbor, chromosome, P, N, T)
#
#         chromosome[ chosen].leftneighbor = (x, y, t)
#
#
#
#
#         for i in range(setuptime-1):
#             if start + i < T :
#                 chromosome[(x, y, start+i)].state = 1
#                 chromosome[(x, y, start + i)].rightneighbor = chosen
#
#
#         for i in range(t+setuptime, end + 1):
#             chromosome[(x, y, i)].state = 2
#             chromosome[(x, y, i)].rightneighbor = (chosen[0],chosen[1],i)
#             chromosome[(chosen[0],chosen[1],i)].leftneighbor = (x, y, i)
#
#
#
#
#
# def disconnect2mantanince(    coordinate, chromosome, P, N, T,setuptime):
#     x = coordinate[0]
#     y = coordinate[1]
#     t = coordinate[2]
#     #here in fact,start=t+1
#     if t-1>=0:
#         if chromosome[x, y, t-1].state == 2:
#             # imply that last time  is setup link,
#             _, end = basic_mutate_func.find_next_setup_time(coordinate, chromosome, P, N, T)
#             rightneighbor =  chromosome[x, y, t-1].rightneighbor
#             x2,y2,t2=rightneighbor
#             for i in range(t, end + 1):
#                 chromosome[(x, y, i)].state = 2
#                 chromosome[(x, y, i)].rightneighbor =(x2,y2,i)
#                 chromosome[(x2,y2,i)].leftneighbor = (x, y, i)

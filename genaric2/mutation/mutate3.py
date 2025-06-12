import genaric2.mutation.basic_mutate_func as basic_mutate_func
import random

def 	disconenct_mutate(coordinate,chromosome,P, N, T,setuptime):


    t=coordinate[2]
    start,end= basic_mutate_func.find_next_setup_time(coordinate, chromosome, P, N, T)
    start=t


    # 随机选择一个函数执行（这里只是示例，实际调用您需要的三个函数之一）
    choice = random.choice(["func1", "func2", "func3"])

    if choice == "func1":
        disconnect2establishment(start, end, coordinate, chromosome, P, N, T, setuptime)

        # disconnect2establishment(...)  # 取消注释并填入实际参数
    elif choice == "func2":
        disconnect2mantanince(    coordinate, chromosome, P, N, T,setuptime)

        # disconnect2maintenance(...)   # 取消注释并填入实际参数
    else:

        disconnect2disconect(coordinate, chromosome, P, N, T, setuptime)

        # disconnect2disconnect(...)    # 取消注释并填入实际参数









import random
import genaric2.mutation.basic_mutate_func as basic_mutate_func


def disconnect2establishment(    start,end,coordinate, chromosome, P, N, T,setuptime):
    x = coordinate[0]
    y = coordinate[1]
    t = coordinate[2]
    #here in fact ,start=t+1
    duration = end - start+1
    if duration+1 <= setuptime:
        return
    else:
        col=x
        row=y
        candidates = []
        next_col = col + 1

        if t + setuptime < T and next_col < P:
            candidates += [
                (next_col, (row - 1) % N, t + setuptime),
                (next_col, row % N, t + setuptime),
                (next_col, (row + 1) % N, t + setuptime),
            ]

            if next_col + 1 < P:
                candidates.append((next_col + 1, row, t + setuptime))
        chosen = random.choice(candidates)




        chromosome[(x, y, start-1)].state = 0
        chromosome[(x, y, start-1)].rightneighbor =  chosen


        ##first we need check the chosen,we need reserved the time for  the setup ,we need check the chosen up

        #1  1
        #1  1:so we may distory this state
        #1  1:here we need to reserved
        #1  1:
        #1  chose

        for i in range(setuptime):
            # we check
            cordinate = (chosen[0], chosen[1], t+i)
            left_neighbor = chromosome[cordinate].leftneighbor
            if left_neighbor == None:
                pass
            else:
                if chromosome[left_neighbor].state == 2:
                    chromosome[left_neighbor].state=-1
                    chromosome[left_neighbor].rightneighbor=None

                    chromosome[cordinate].leftneighbor=None


                elif chromosome[left_neighbor].state == 1 or chromosome[left_neighbor].state == 0:
                    basic_mutate_func.clear_state(left_neighbor, chromosome, P, N, T, setuptime)
                    chromosome[cordinate].leftneighbor = None

        # then need handling conflicts
        #
        if  chromosome[ chosen].leftneighbor!=None:
            leftneighbor = chromosome[ chosen].leftneighbor
            basic_mutate_func.clear_state(leftneighbor, chromosome, P, N, T)

        chromosome[ chosen].leftneighbor = (x, y, t)




        for i in range(setuptime-1):
            if start + i < T :
                chromosome[(x, y, start+i)].state = 1
                chromosome[(x, y, start + i)].rightneighbor = chosen


        for i in range(t+setuptime, end + 1):
            chromosome[(x, y, i)].state = 2
            chromosome[(x, y, i)].rightneighbor = (chosen[0],chosen[1],i)
            chromosome[(chosen[0],chosen[1],i)].leftneighbor = (x, y, i)





def disconnect2mantanince(    coordinate, chromosome, P, N, T,setuptime):
    x = coordinate[0]
    y = coordinate[1]
    t = coordinate[2]
    #here in fact,start=t+1
    if t-1>=0:
        if chromosome[x, y, t-1].state == 2:
            # imply that last time  is setup link,
            _, end = basic_mutate_func.find_next_setup_time(coordinate, chromosome, P, N, T)
            rightneighbor =  chromosome[x, y, t-1].rightneighbor
            x2,y2,t2=rightneighbor
            for i in range(t, end + 1):
                chromosome[(x, y, i)].state = 2
                chromosome[(x, y, i)].rightneighbor =(x2,y2,i)
                chromosome[(x2,y2,i)].leftneighbor = (x, y, i)


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
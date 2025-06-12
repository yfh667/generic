
import genaric2.mutation.basic_mutate_func as basic_mutate_func
def establishment_mutate(coordinate,chromosome,P, N, T,setuptime):
    #we will continued this estavlishment and cover the next establishment until the second setabliashment
    x = coordinate[0]
    y=coordinate[1]
    t=coordinate[2]
    right_neighbor=chromosome[coordinate].rightneighbor

    #first we will find the next establishment
    uptime, down_time=find_time_period_for_establishment(coordinate,chromosome,  P, N, T)
    nexttime_period =find_one_time_period_for_establishment(coordinate,chromosome , P, N, T)
    for i in range(nexttime_period,down_time+1):
        chromosome[(x, y, i)].state = 2
        # we need firstly change the raw rightbor
        thisright_neighbor=chromosome[(x, y, i)].rightneighbor
        if thisright_neighbor!=None:
          chromosome[thisright_neighbor].leftneighbor=None

        chromosome[(x, y, i)].rightneighbor=(right_neighbor[0],right_neighbor[1],i)
        # here we need solve the confilct for (right_neighbor[0],right_neighbor[1],i)

        left_neighbor=chromosome[(right_neighbor[0],right_neighbor[1],i)].leftneighbor

        basic_mutate_func.clear_state(left_neighbor,chromosome ,P, N, T,setuptime)

        chromosome[(right_neighbor[0],right_neighbor[1],i)].leftneighbor=(x, y, i)




def find_one_time_period_for_establishment(coordinate,chromosome ,P, N, T):
    x, y, t = coordinate

    while t + 1 < T :
        t += 1
        if chromosome[(x, y, t)].state == -1:  # Node is setting up the link
            break
    return t



def find_time_period_for_establishment(coordinate,chromosome, P, N, T):
    # Extract x, y, and t values from coordinate
    x, y, t = coordinate

    # If the node at the given coordinate is setting up the link (state == 0)
    if chromosome[coordinate].state == 0:
        # Flag for finding the next establishment
        flag = 2
        # Loop through time steps to find the next establishment
        while t + 1 < T and flag:
            t += 1
            if chromosome[(x, y, t)].state == 0:  # Node is setting up the link
                flag -= 1  # Decrease flag to exit loop

        # The time when the node stops setting up the link
        down_time = t - 1

        return coordinate[2], down_time

    # If the node at the given coordinate is working (state == 1)
    elif chromosome[coordinate].state == 1:
        # Start with the current time step
        uptime = t

        # Loop backward to find the last time the node was working
        while uptime - 1 >=0:
            uptime -= 1
            if chromosome[(x, y, uptime)].state == 0:  # Node is setting up the link
                break

        # Find the next time step when the node starts setting up the link
        t = coordinate[2]
        flag = 2

        # Loop through time steps to find the next establishment
        while t + 1 < T and flag:
            t += 1
            if chromosome[(x, y, t)].state == 0:  # Node is setting up the link

                flag -= 1  # Decrease flag to exit loop


        # The time when the node stops setting up the link
        down_time = t - 1

        return uptime, down_time


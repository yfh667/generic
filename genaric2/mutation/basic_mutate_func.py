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


def clear_leftneighbor(rightneighbor,chromosome):
    if rightneighbor:
        chromosome[rightneighbor].leftneighbor = None
        # we need clear all the state until the state is 0


def clear_state(coordinate,chromosome ,P, N, T,setuptime):

    if not coordinate:
        return
    # if not coordinate[0]:
    #     print(1)
    # if not isinstance(coordinate, (tuple, list)) or len(coordinate) < 3:
    #     print(f"错误：coordinate参数类型不正确，应为(x,y,z)格式，实际得到: {coordinate}")
    #     return


    x = coordinate[0]
    y = coordinate[1]
    t = coordinate[2]
    if chromosome[coordinate].state == 0:
        # we need clear all the state until the state is 0
        chromosome[(x, y, t)].state=-1

        rightneighbor=chromosome[(x, y, t)].rightneighbor
        clear_leftneighbor(rightneighbor, chromosome)

        if rightneighbor:
            x2,y2,t2=rightneighbor
            for i in range(setuptime):
                chromosome[(x2, y2, t2-i-1)].leftneighbor=None

        chromosome[(x, y, t)].rightneighbor = None
        while t + 1 < T:
            t += 1
            if chromosome[(x, y, t)].state == -1 or chromosome[(x, y, t)].state == 0:  # Node is setting up the link
                break
            else:
                chromosome[(x, y, t)].state = -1

                rightneighbor = chromosome[(x, y, t)].rightneighbor
                clear_leftneighbor(rightneighbor, chromosome)

                chromosome[(x, y, t)].rightneighbor = None


    elif chromosome[coordinate].state == 1 or chromosome[coordinate].state == 2:
        # Start with the current time step
        uptime = t

        # Loop backward to find the last time the node was working
        while uptime - 1 >= 0:
            uptime -= 1
            if chromosome[(x, y, uptime)].state == 0:  # Node is setting up the link
                break

        t=uptime

        chromosome[(x, y, t)].state = -1

        rightneighbor = chromosome[(x, y, t)].rightneighbor
        clear_leftneighbor(rightneighbor, chromosome)

        if rightneighbor:
            x2, y2, t2 = rightneighbor
            for i in range(setuptime):
                chromosome[(x2, y2, t2 - i - 1)].leftneighbor = None

        chromosome[(x, y, t)].rightneighbor = None
        while t + 1 < T:
            t += 1
            if chromosome[(x, y, t)].state == -1 or  chromosome[(x, y, t)].state==0 :  # Node is setting up the link
                break
            else:
                chromosome[(x, y, t)].state = -1

                rightneighbor = chromosome[(x, y, t)].rightneighbor
                clear_leftneighbor(rightneighbor, chromosome)

                chromosome[(x, y, t)].rightneighbor = None


# this variant  solve this conflict , when  A is in a establsihment state with B, we will not delet all the lifycycle for the A.
# because , sometimes ,just A in this stage need solve the confilct ,previous time ,there is no need to solve the conflict




def clear_state2(coordinate,chromosome ,P, N, T,setuptime):

    if not coordinate:
        return
    # if not coordinate[0]:
    #     print(1)
    # if not isinstance(coordinate, (tuple, list)) or len(coordinate) < 3:
    #     print(f"错误：coordinate参数类型不正确，应为(x,y,z)格式，实际得到: {coordinate}")
    #     return


    x = coordinate[0]
    y = coordinate[1]
    t = coordinate[2]
    if chromosome[coordinate].state == 0:
        # we need clear all the state until the state is 0
        chromosome[(x, y, t)].state=-1

        rightneighbor=chromosome[(x, y, t)].rightneighbor
        clear_leftneighbor(rightneighbor, chromosome)

        if rightneighbor:
            x2,y2,t2=rightneighbor
            for i in range(setuptime):
                chromosome[(x2, y2, t2-i-1)].leftneighbor=None

        chromosome[(x, y, t)].rightneighbor = None
        while t + 1 < T:
            t += 1
            if chromosome[(x, y, t)].state == -1 or chromosome[(x, y, t)].state == 0:  # Node is setting up the link
                break
            else:
                chromosome[(x, y, t)].state = -1

                rightneighbor = chromosome[(x, y, t)].rightneighbor
                clear_leftneighbor(rightneighbor, chromosome)

                chromosome[(x, y, t)].rightneighbor = None


    elif chromosome[coordinate].state == 1  :
        # Start with the current time step
        uptime = t

        # Loop backward to find the last time the node was working
        while uptime - 1 >= 0:
            uptime -= 1
            if chromosome[(x, y, uptime)].state == 0:  # Node is setting up the link
                break

        t=uptime

        chromosome[(x, y, t)].state = -1

        rightneighbor = chromosome[(x, y, t)].rightneighbor
        clear_leftneighbor(rightneighbor, chromosome)

        if rightneighbor:
            x2, y2, t2 = rightneighbor
            for i in range(setuptime):
                chromosome[(x2, y2, t2 - i - 1)].leftneighbor = None

        chromosome[(x, y, t)].rightneighbor = None
        while t + 1 < T:
            t += 1
            if chromosome[(x, y, t)].state == -1 or  chromosome[(x, y, t)].state==0 :  # Node is setting up the link
                break
            else:
                chromosome[(x, y, t)].state = -1

                rightneighbor = chromosome[(x, y, t)].rightneighbor
                clear_leftneighbor(rightneighbor, chromosome)

                chromosome[(x, y, t)].rightneighbor = None
    elif chromosome[coordinate].state == 2:
        # Start with the current time step
        uptime = t


        # first we check ,whether it is just  the establsihmet 1st mantenace.

        if chromosome[x,y,t-1].state==1:
            # it is the first establsihment 1st matence,we need delete all the lifecycle

            # Loop backward to find the last time the node was working
            while uptime - 1 >= 0:
                uptime -= 1
                if chromosome[(x, y, uptime)].state == 0:  # Node is setting up the link
                    break

            t = uptime

            chromosome[(x, y, t)].state = -1

            rightneighbor = chromosome[(x, y, t)].rightneighbor
            clear_leftneighbor(rightneighbor, chromosome)

            if rightneighbor:
                x2, y2, t2 = rightneighbor
                for i in range(setuptime):
                    chromosome[(x2, y2, t2 - i - 1)].leftneighbor = None

            chromosome[(x, y, t)].rightneighbor = None
            while t + 1 <T:
                t += 1
                if chromosome[(x, y, t)].state == -1 or chromosome[(x, y, t)].state == 0:  # Node is setting up the link
                    break
                else:
                    chromosome[(x, y, t)].state = -1

                    rightneighbor = chromosome[(x, y, t)].rightneighbor
                    clear_leftneighbor(rightneighbor, chromosome)

                    chromosome[(x, y, t)].rightneighbor = None
        elif chromosome[x,y,t-1].state==2:
            chromosome[(x, y, t)].state = -1

            rightneighbor = chromosome[(x, y, t)].rightneighbor
            clear_leftneighbor(rightneighbor, chromosome)



            chromosome[(x, y, t)].rightneighbor = None

            while t + 1 < T:
                t += 1
                if chromosome[(x, y, t)].state == 2 :  # Node is setting up the link
                    chromosome[(x, y, t)].state = -1
                    rightneighbor = chromosome[(x, y, t)].rightneighbor
                    clear_leftneighbor(rightneighbor, chromosome)
                    chromosome[(x, y, t)].rightneighbor = None

                else:
                  break



def clear_state_just_leftneighbor(coordinate,chromosome ,P, N, T,setuptime):

    if not coordinate:
        return
    # if not coordinate[0]:
    #     print(1)
    # if not isinstance(coordinate, (tuple, list)) or len(coordinate) < 3:
    #     print(f"错误：coordinate参数类型不正确，应为(x,y,z)格式，实际得到: {coordinate}")
    #     return


    x = coordinate[0]
    y = coordinate[1]
    t = coordinate[2]
    if chromosome[coordinate].state == 0:
        # we need clear all the state until the state is 0
        chromosome[(x, y, t)].state=-1

        rightneighbor=chromosome[(x, y, t)].rightneighbor
        clear_leftneighbor(rightneighbor, chromosome)

        if rightneighbor:
            x2,y2,t2=rightneighbor
            for i in range(setuptime):
                chromosome[(x2, y2, t2-i-1)].leftneighbor=None

        chromosome[(x, y, t)].rightneighbor = None
        while t + 1 < T:
            t += 1
            if chromosome[(x, y, t)].state == -1 or chromosome[(x, y, t)].state == 0:  # Node is setting up the link
                break
            else:
                chromosome[(x, y, t)].state = -1

                rightneighbor = chromosome[(x, y, t)].rightneighbor
                clear_leftneighbor(rightneighbor, chromosome)

                chromosome[(x, y, t)].rightneighbor = None


    elif chromosome[coordinate].state == 1  :
        # Start with the current time step
        uptime = t

        # Loop backward to find the last time the node was working
        while uptime - 1 >= 0:
            uptime -= 1
            if chromosome[(x, y, uptime)].state == 0:  # Node is setting up the link
                break

        t=uptime

        chromosome[(x, y, t)].state = -1

        rightneighbor = chromosome[(x, y, t)].rightneighbor
        clear_leftneighbor(rightneighbor, chromosome)

        if rightneighbor:
            x2, y2, t2 = rightneighbor
            for i in range(setuptime):
                chromosome[(x2, y2, t2 - i - 1)].leftneighbor = None

        chromosome[(x, y, t)].rightneighbor = None
        while t + 1 < T:
            t += 1
            if chromosome[(x, y, t)].state == -1 or  chromosome[(x, y, t)].state==0 :  # Node is setting up the link
                break
            else:
                chromosome[(x, y, t)].state = -1

                rightneighbor = chromosome[(x, y, t)].rightneighbor
                clear_leftneighbor(rightneighbor, chromosome)

                chromosome[(x, y, t)].rightneighbor = None
    elif chromosome[coordinate].state == 2:
        # Start with the current time step
        uptime = t


        # first we check ,whether it is just  the establsihmet 1st mantenace.

        if chromosome[x,y,t-1].state==1:
            # it is the first establsihment 1st matence,we need delete all the lifecycle

            # Loop backward to find the last time the node was working
            while uptime - 1 >= 0:
                uptime -= 1
                if chromosome[(x, y, uptime)].state == 0:  # Node is setting up the link
                    break

            t = uptime

            chromosome[(x, y, t)].state = -1

            rightneighbor = chromosome[(x, y, t)].rightneighbor
            clear_leftneighbor(rightneighbor, chromosome)

            if rightneighbor:
                x2, y2, t2 = rightneighbor
                for i in range(setuptime):
                    chromosome[(x2, y2, t2 - i - 1)].leftneighbor = None

            chromosome[(x, y, t)].rightneighbor = None
            while t + 1 <T:
                t += 1
                if chromosome[(x, y, t)].state == -1 or chromosome[(x, y, t)].state == 0:  # Node is setting up the link
                    break
                else:
                    chromosome[(x, y, t)].state = -1

                    rightneighbor = chromosome[(x, y, t)].rightneighbor
                    clear_leftneighbor(rightneighbor, chromosome)

                    chromosome[(x, y, t)].rightneighbor = None
        elif chromosome[x,y,t-1].state==2:
            chromosome[(x, y, t)].state = -1

            rightneighbor = chromosome[(x, y, t)].rightneighbor
            clear_leftneighbor(rightneighbor, chromosome)



            chromosome[(x, y, t)].rightneighbor = None

            while t + 1 < T:
                t += 1
                if chromosome[(x, y, t)].state == 2 :  # Node is setting up the link
                    chromosome[(x, y, t)].state = -1
                    rightneighbor = chromosome[(x, y, t)].rightneighbor
                    clear_leftneighbor(rightneighbor, chromosome)
                    chromosome[(x, y, t)].rightneighbor = None

                else:
                  break



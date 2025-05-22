

def find_end(time_now, node_id, neighbor_id,setuptime):






def structure2nodes(P,N,T,setuptime,nodes,structure):
    for i in range(P):
        for j in range(N):
            for k in range(T):
                if structure[(i,j,k)] !=-1 and structure[(i,j,k)] !=1:
                    neighbor = structure[(i,j,k)]

                    time_end = find_end(k+1, (i,j,k), neighbor,setuptime)
                  #  print(neighbor)
                    nodes[(i,j,k)].state = 0

                    nodes[(i,j,k)].rightneighbor = neighbor

                    nowtime = k+1
                    while 1:
                        if nowtime >= T:
                            break
                        if  nodes[(i,j,nowtime)]==-1 :
                            if nowtime<k+setuptime:
                                nodes[(i,j,nowtime)].state = 1
                            else:
                                nodes[(i,j,nowtime)].state = 2
                            nodes[(i, j, nowtime)].rightneighbor = structure[(i, j, k)]
                            nowtime = nowtime+1
                        else:
                            break
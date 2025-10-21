import basicSa.linkalgorith.alltogether1 as alltogether

def linkengine(time,P, N,positions,gndnodes,oldpath,):
    # in the python code ,it is needed to reset the linked_arrary
    for i in range(len(gndnodes)):
        gndnodes[i].linked_array[0] = -1
# below it is the link algorith and the route algorithm
    adj_list,min_paths=alltogether.alltogether_new(time,P, N, positions, gndnodes,[[0,1]],  oldpath)

    nodes = []
    for i in range(len(gndnodes)):
        nodes.append(gndnodes[i])
    for i in range(len(positions)):
        nodes.append(positions[i][0])

    return nodes,adj_list,min_paths

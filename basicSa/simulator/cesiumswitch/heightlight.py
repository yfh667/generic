from copy import deepcopy

import basicSa.simulator.sendtocesium as sendtocesium2


def highlight_selected_path(  taget, RAWnodes):
    """更新Cesium显示指定路径"""
    # snapshot = self.database.snapshots[self.currenttime]
    Nodes = deepcopy(RAWnodes)

    srcid = taget[0]
    destid = taget[1]
    if Nodes[srcid].linked_array[0] == -1:
        # 重置所有链路
        for node in Nodes:
            node.linked_array = [-1] * 5


    else:
        path = []
        path.append(srcid)
        nexthop = Nodes[srcid].linked_array[0]
        flag = 1

        path.append(nexthop)

        while (flag):

            for i in range(5, 15):
                if Nodes[nexthop].get_value(i) == [-1, -1]:
                    flag = 0
                    break
                elif Nodes[nexthop].get_value(i)[0] == destid:

                    port = Nodes[nexthop].get_value(i)[1]
                    #  destid =  Nodes[nexthop].get_value(i)[0]
                    if port == 0:
                        path.append(destid)
                        flag = 0
                        break
                    else:
                        nexthop = Nodes[nexthop].linked_array[port]
                        path.append(nexthop)
                    break

        # 重置所有链路
        for node in Nodes:
            node.linked_array = [-1] * 5

        #

        src_id = path[0]
        dest_id = path[-1]
        Nodes[src_id].linked_array[0] = path[1]
        Nodes[dest_id].linked_array[0] = path[-2]

        for i in range(1, len(path) - 2):
            Nodes[path[i]].linked_array[1] = path[i + 1]
    return Nodes

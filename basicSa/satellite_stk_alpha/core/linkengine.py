# core/link_engine.py
from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
from collections import defaultdict
import basicSa.satellite_stk_alpha.config.settings as settings
# from config.settings import SatelliteConfig
import basicSa.calculate_angular.calculate_angular as calculate_angular
from basicSa.satellite_stk_alpha.contrib.route import SatelliteRouter
from basicSa.satellite_stk_alpha.core.angularvector import SatelliteVector
import basicSa.utilis.Node as Node
from basicSa.los import  Sat2Gnd


def addtoroute(node:Node.Node,destid,nexthop):

    for i in range(5,15):
        if node.get_value(i) ==[-1,-1]:
            node .set_value(i, [destid, nexthop])
            return

    print("ERROR addtoroute")

def get_the_min_orbit(path1,distance1,path2,distance2,gndnodes,N):
    min_path = []
    min_distance = []
    stationnum = len(gndnodes)
    orbit_id1 = set()
    for i in range(1,len(path1)-1):
        satid = path1[i]-stationnum
        orbitid = satid //N
        orbit_id1.add(orbitid)
    orbit_id2 = set()
    for i in range(1,len(path2)-1):
        satid = path2[i]-stationnum
        orbitid = satid //N
        orbit_id2.add(orbitid)

    if len(orbit_id1)>len(orbit_id2):
        min_path = path2
        min_distance = distance2
    else:
        min_path = path1
        min_distance = distance1

    return min_path ,min_distance
def alltogether_new(time,P, N, positions, gndnodes,targets):
    # 使用向量化计算
   # target:[[A,B],[C,D]]


    gndlenth = len(gndnodes)

    ## sat-sat visible
    # 计算角速度
    sv = SatelliteVector(P, N)
    sv.load_from_3dview(positions)

    # 批量计算
    angular_vel = sv.calculate_angular_velocity(time)
    los_mask = angular_vel < settings.SatelliteConfig.ANGULAR_VELOCITY

    #test_angular = sv.calculate_angular_velocity_7_all(time)


    # 生成邻接表
    adj_list = defaultdict(list)
    for i in range(len(los_mask)):
        if los_mask[i] and sv._get_right_neighbor(i) is not None:
            orbit = i // N
            plane = i % N
            adj_list[(orbit, plane)].append((orbit + 1, plane))
    #why we need the adj_list? cause we need plot the 2d view of the linktable

    # if time==1619:
    #     print("1")

    ## 设定卫星的激光链路表
    # 遍历所有卫星位置，建立相邻卫星之间的链路连接
    for i in range(len(positions)):
        node1 = positions[i][0]  # 获取当前卫星节点
        plainid = i // N  # 计算当前卫星所在的平面编号
        oribit_id = i % N  # 计算当前卫星在平面中的轨道编号
        up_oribit_id = (oribit_id - 1) % N  # 获取上一轨道的编号（循环连接）
        down_oribit_id = (oribit_id + 1) % N  # 获取下一轨道的编号（循环连接）
        upnodeid = plainid * N + up_oribit_id  # 计算上一轨道卫星的全局编号
        downnodeid = plainid * N + down_oribit_id  # 计算下一轨道卫星的全局编号
        node1.linked_array[2]=upnodeid + gndlenth
        node1.linked_array[4] = downnodeid + gndlenth
        node1.set_value(2, [upnodeid + gndlenth, 4])  # 设置当前卫星的上链路连接
        node1.set_value(4, [downnodeid + gndlenth, 2])  # 设置当前卫星的下链路连接

    # 将邻接列表（adj_list）写入链路表
    # 遍历邻接列表，为每个卫星节点设置与邻居的链路连接
    for grid_key, nexthop in adj_list.items():
        nodeid = grid_key[0] * N + grid_key[1]  # 计算当前卫星的全局编号
        node = positions[nodeid][0]  # 获取当前卫星节点
        for pos in nexthop:
            dextnodeid = pos[0] * N + pos[1]  # 计算邻居卫星的全局编号
            node.linked_array[1] = dextnodeid + gndlenth  # 更新当前卫星的链路数组
            node.set_value(1, [dextnodeid + gndlenth, 3])  # 设置当前卫星到邻居的链路连接
            node.linked_array[1] =dextnodeid + gndlenth
            node2 = positions[dextnodeid][0]  # 获取邻居卫星节点
            node2.linked_array[3] = nodeid + gndlenth  # 更新邻居卫星的链路数组
            node2.set_value(3, [node2.linked_array[3], 1])  # 设置邻居卫星到当前卫星的链路连接



    ## sat-gnd visible
    ##-------------------------------------------------------------__#####
    # next we need calculate the sat and the ground
    positionslength = len(positions)
    # 遍历每个地面站（行索引 i）
   # gndlink = np.zeros(len(gndnodes), dtype='int32')
    # 假设 gndnodes 是地面节点的列表/数组
    gndlink = np.empty(len(gndnodes), dtype=object)  # 用 empty 替代 zeros

    # 为每个元素初始化为空列表
    for i in range(len(gndlink)):
        gndlink[i] = []  # 每个地面节点关联一个空列表（用于存储可见卫星）

    for i, node1 in enumerate(gndnodes):
        # node1:gnd
        # node2:sat
        # 遍历每个卫星位置（列索引 j）→ 直接取字典的值（即元组）
        for j in range(positionslength):
            node2 = positions[j][0]  # 从元组中提取 Node 对象
            visibility = Sat2Gnd.Ground_Sat(node1, node2)
            if visibility:
                gndlink[i].append(j)

                # if not gndlink[i]:
                #     gndlink[i] = j
                #     # gnd-sat
                #     node1.linked_array[0] =j+len(gndnodes)
                #     # sat-gnd
                #     node2.linked_array[0] = i



    ## sat-gnd visible
    ##-------------------------------------------------------------#####
    for target in targets:
        start_satellite = -1
        end_satellite = -1

        gndnode0_id = target[0]
        gndnode0 = gndnodes[gndnode0_id]

        gndnode1_id = target[1]
        gndnode1 = gndnodes[gndnode1_id]


        # below it is the
        gdnlink_copy0_len = len(gndlink[target[0]])
        gdnlink_copy1_len = len(gndlink[target[1]])

        if gdnlink_copy0_len and gdnlink_copy1_len:
                pass
        else:
            gndnode0.linked_array[0] = -1
            gndnode1.linked_array[0] = -1
            break

        start_satellites = gndlink[target[0]]
        end_satellites = gndlink[target[1]]
        router = SatelliteRouter(P, N)
        router.build_topology(adj_list)

        min_path = None
        min_distance = float('inf')

        # 遍历每个起始卫星
        for start_satellite in start_satellites:
            # 计算从当前卫星到所有节点的最短路径
            distances, paths = router.find_path(start_satellite)

            # 遍历所有可达的终点节点,start_satellite to all nodes in start_satellites
            # 遍历所有可达的终点节点（paths的键是目标节点）
            for end_node in paths.keys():  # 明确遍历字典的键（目标节点）
                # 检查当前目标节点是否在目标卫星列表中
                if end_node in end_satellites:
                    current_distance = distances[end_node]
                    current_path = paths[end_node]
                    # 更新最短路径
                    if current_distance < min_distance:

                        min_distance = current_distance
                        min_path = current_path
                    else:
                        min_path,min_distance =get_the_min_orbit(current_path, current_distance, min_path, min_distance, gndnodes, N)



            # here we add the route to  the node
            # first we need set the gnd-sat
         #
        # gndnode0 link table
        start_satellite = min_path[0]
        gndnode0.set_value(0,( start_satellite+gndlenth,0))
        gndnode0.linked_array[0] = start_satellite+gndlenth

        #gndnode0 raw route table,
     #   gndnode0 .set_value(5, (1, start_satellite+gndlenth))

        addtoroute(gndnode0, gndnode1_id, start_satellite+gndlenth)



        # the we set the sat -sat
        for i in range(len(min_path)):
            if i!=len(min_path)-1:
                node = positions[min_path[i]][0]
                addtoroute(node, gndnode1_id, min_path[i+1]+gndlenth)
         #       node.set_value(5,(gndnode1_id,path[i+1]+gndlenth))
            else:
                node = positions[min_path[i]][0]
                # attention the last basicSa need set the link table and the route table
                #link table
                node.set_value(0,[gndnode1_id,0])
                # route table, casue it is the route to the gnd ,so there is  no need to set the link table for gnd link
                addtoroute(node, gndnode1_id, 0)


    # for target in targets:
    #     start_satellite = -1
    #     end_satellite = -1
    #
    #     gndnode0_id = target[0]
    #     gndnode0 = gndnodes[gndnode0_id]
    #
    #     gndnode1_id = target[1]
    #     gndnode1 = gndnodes[gndnode1_id]
    #
    #
    #     # below it is the
    #     gdnlink_copy0  = gndlink[target[0]]
    #     gdnlink_copy1 = gndlink[target[1]]
    #
    #     if gdnlink_copy0 and gdnlink_copy1:
    #         start_satellite = gdnlink_copy0
    #         end_satellite = gdnlink_copy1
    #     else:
    #         gndnode0.linked_array[0] = -1
    #         gndnode1.linked_array[0] = -1
    #    # print(f"basicSa: {start_satellite}, end: {end_satellite}")
    #
    #     ##here we need calculate the route
    #     if start_satellite != -1 and end_satellite != -1:
    #
    #         router = SatelliteRouter(P, N)
    #         router.build_topology(adj_list)
    #         path = router.find_path(start_satellite, end_satellite)
    #        # route_adj = router.generate_route_adj(path)
    #
    #
    #
    #         # here we add the route to  the node
    #         # first we need set the gnd-sat
    #
    #         # gndnode0 link table
    #         gndnode0.set_value(0,( start_satellite+gndlenth,0))
    #
    #         #gndnode0 raw route table,
    #      #   gndnode0 .set_value(5, (1, start_satellite+gndlenth))
    #
    #         addtoroute(gndnode0, gndnode1_id, start_satellite+gndlenth)
    #
    #
    #
    #     # the we set the sat -sat
    #         for i in range(len(path)):
    #             if i!=len(path)-1:
    #                 node = positions[path[i]][0]
    #                 addtoroute(node, gndnode1_id, path[i+1]+gndlenth)
    #          #       node.set_value(5,(gndnode1_id,path[i+1]+gndlenth))
    #             else:
    #                 node = positions[path[i]][0]
    #                 # attention the last basicSa need set the link table and the route table
    #                 #link table
    #                 node.set_value(0,[gndnode1_id,0])
    #                 # route table, casue it is the route to the gnd ,so there is  no need to set the link table for gnd link
    #                 addtoroute(node, gndnode1_id, 0)
    #
    #
    #


##here we need transform the  raw route table to the ns3-route table
    for i in range(len(gndnodes) ):
        node_y = gndnodes[i]
        for i in range(5, 15):
            if node_y.get_value(i) != [-1, -1] :
                nodeid_next = node_y.get_value(i)[1]
                for j in range(5):
                    if node_y.get_value(j)[0] == nodeid_next:
                        node_y.set_value(i, [node_y.get_value(i)[0], j])
            else:
                break

    for i in range(len(positions)):
        node_y = positions[i][0]
        for i in range(5, 15):
            if node_y.get_value(i) != [-1, -1]:
                nodeid_next = node_y.get_value(i)[1]
                if nodeid_next != 0:
                    for j in range(5):
                        if node_y.get_value(j)[0] == nodeid_next:
                            node_y.set_value(i, [node_y.get_value(i)[0], j])
            else:
                break



    return  adj_list



class LinkEngine(QObject):
    topology_updated = pyqtSignal(dict)  # 新增拓扑更新信号
    modify_nodes= pyqtSignal(  list)  # 保持字典参数类型

    def __init__(self, data_hub):
        super().__init__()
        self.data_hub = data_hub
        self.current_params = (0, 0)  # (P, N)
       # self.vector_cls = SatelliteVector  # 可替换计算实现
        #
        self.runflag = 0
        # 连接数据更新信号


    def set_parameters(self, P, N):
        """更新轨道参数"""
        self.current_params = (P, N)





    def on_positions_updated(self,time, positions,gndnodes):
        """位置更新时的处理"""
        P, N = self.current_params
        if P == 0 or N == 0:
            return


# in the python code ,it is needed to reset the linked_arrary
        for i in range(len(gndnodes)):
            gndnodes[i].linked_array[0] = -1



# below it is the link algorith and the route algorithm
        adj_list=alltogether_new(time,P, N, positions, gndnodes,[[0,1],[2,3],[4,5]])

        nodes = []
        for i in range(len(gndnodes)):
            nodes.append(gndnodes[i])
        for i in range(len(positions)):
            nodes.append(positions[i][0])
        self.modify_nodes.emit( nodes)  # 发射信号




        if not self.runflag :

            self.topology_updated.emit({
                "adjacency":adj_list,
                "satellites": positions,
                "gnd":gndnodes,
                "params": self.current_params,
                "timestamp": self.data_hub.current_time,
            })






# 老方法，一般是不用的了,下面都是不用的了
#     def on_positions_updated2(self, positions):
#
#         P, N = self.current_params
#         nodes = [(i, j) for i in range(0, P) for j in range(0, N + 1)]
#         adj_list = {node: [] for node in nodes}
#         for id in range(1,(P-1)*N+1):
#             self.calculatelink( id, adj_list, positions)
#      #   self.update_topology(N, P, adj_list)
#         self.topology_updated.emit({
#             "adjacency": adj_list,
#             "params": self.current_params,
#             "timestamp": self.data_hub.current_time
#         })
#
#
#
#
#     def calculatelink(self,id,adj_list,satellite_positions_3d):
#         P,N = self.current_params
#         orbitid = (id-1) //N
#         planeid_start =orbitid * N+1
#         planeid_id = id-planeid_start
#         # if(id==30):
#         #     print("1")
#         # # # right
#         if orbitid != P-1:
#             #RIGHTE
#             rightnode = (orbitid+1)* N+planeid_id+1
#
#             # sat1 = satellite_positions_3d[id  ]
#             # x1,y1,z1 =  sat1.x,sat1.y,sat1.z
#             # sat2 = satellite_positions_3d[rightnode]
#             #
#             # x2,y2,z2 = sat2.x,sat2.y,sat2.z
#             #
#             # #here is the los code
#             #
#             # #this use the simplest los based on the light of sight
#             # los = sat2sat.Sat_Sat2(x1, y1, z1, x2, y2, z2)
#
#             sat1_1 = satellite_positions_3d[id][0]
#             x1_1,y1_1,z1_1 = sat1_1.x,sat1_1.y,sat1_1.z
#             sat1_2 = satellite_positions_3d[id][1]
#             x1_2,y1_2,z1_2 = sat1_2.x,sat1_2.y,sat1_2.z
#
#
#             sat2_1 = satellite_positions_3d[rightnode][0]
#             x2_1,y2_1,z2_1 = sat2_1.x,sat2_1.y,sat2_1.z
#             sat2_2 = satellite_positions_3d[rightnode][1]
#             x2_2,y2_2,z2_2 = sat2_2.x,sat2_2.y,sat2_2.z
#
#
#             timestep =1
#             params = ( x1_1,y1_1,z1_1 ,   x1_2,y1_2,z1_2,
#                       x2_1,y2_1,z2_1,   x2_2,y2_2,z2_2 ,
#                       sat1_1.trackangle, timestep,  sat1_1.RAAN)
#
#             angular_velocity = calculate_angular.Get_angular_velocity(params)
#
#             if abs(angular_velocity[2])< settings.SatelliteConfig.ANGULAR_VELOCITY:
#                 los=1
#             else:
#                 los=0
#
#             if los:
#                 adj_list[orbitid, planeid_id].append((orbitid+1, planeid_id))

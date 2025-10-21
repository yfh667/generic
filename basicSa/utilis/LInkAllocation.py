import basicSa.utilis.Node as Nodepy

# 主链路分配函数
# 参数:
# - stationnum: 地面站编号，用于标识与卫星的连接关系
# - Numerperplane: 每一轨道上的卫星数量
# - path: 包含经过卫星的路径，路径中的卫星将建立链路连接
# - nodes: 包含所有卫星的 Node 列表，每个元素都是一个 Node 实例
# 功能:
# 1. 分配地面站到路径中的首尾卫星的链接。
# 2. 对路径中的卫星，依序建立链路连接。
def LinkAllocation(stationnum, Numerperplane, path: list, nodes: list[Nodepy.Node]):
    # 确定路径的起点和终点，即地面站 station1 和 station2 的 ID
    station1id = path[0]
    station2id = path[-1]

    # 完成地面站和路径中首尾卫星的链路分配
    SatGndLink(path[1], station1id, nodes[path[1]], nodes[station1id])
    SatGndLink(path[-2], station2id, nodes[path[-2]], nodes[station2id])

    # 遍历路径中的卫星，逐对分配激光链路
    for i in range(1, len(path) - 2):
        SatLInkAllocation(stationnum, Numerperplane, path[i], path[i + 1], nodes[path[i]], nodes[path[i + 1]])


# 卫星间激光链路分配函数
# 功能:
# 分配两个卫星之间的激光链路，判断是否同轨道并调用相应的子分配函数。
# 参数:
# - basicSa: 初始卫星编号，用于计算实际卫星ID
# - N: 每个轨道上的卫星数量
# - sat1id: 第一个卫星的 ID（全局序号）
# - sat2id: 第二个卫星的 ID
# - satellite1, satellite2: 分别表示第一个和第二个卫星的 Node 实例
def SatLInkAllocation(start, N, sat1id, sat2id, satellite1: Nodepy.Node, satellite2: Nodepy.Node):
    # 计算实际的卫星 ID，相对于起始卫星 ID
    realsat1id = sat1id - start + 1
    realsat2id = sat2id - start + 1

    # 计算两个卫星的轨道编号
    plane_id_sat1 = realsat1id // N
    plane_id_sat2 = realsat2id // N
    print("Starting link allocation between satellites...")

    # 判断是否在同一轨道
    if plane_id_sat1 == plane_id_sat2:
        print("Satellites in the same orbit")
        allocate_same_orbit_link(sat1id, sat2id, satellite1, satellite2)
    else:
        print("Satellites in different orbits")
        allocate_different_orbit_link(sat1id, sat2id, satellite1, satellite2)


# 同轨道卫星激光链路分配
# 功能:
# 分配同轨道卫星间的激光链路
# 参数:
# - sat1id, sat2id: 第一个和第二个卫星的 ID
# - satellite1, satellite2: 第一个和第二个卫星的 Node 实例
def allocate_same_orbit_link(sat1id, sat2id, satellite1: Nodepy.Node, satellite2: Nodepy.Node):
    # 检查并分配链路至对应的激光终端
    if satellite1.linked_array[1] == -1:
        # 设定激光链路到 terminal 1 和 terminal 3
        satellite1.linked_array[1] = sat2id
        satellite1.set_value(1, [sat2id, 3])
        satellite2.linked_array[3] = sat1id
        satellite2.set_value(3, [sat1id, 1])
    else:
        # 若 terminal 1 已被占用，则使用 terminal 3 和 terminal 1
        satellite1.linked_array[3] = sat2id
        satellite1.set_value(3, [sat2id, 1])
        satellite2.linked_array[1] = sat1id
        satellite2.set_value(1, [sat1id, 3])


# 不同轨道卫星激光链路分配
# 功能:
# 分配不同轨道卫星间的激光链路
# 参数:
# - sat1id, sat2id: 第一个和第二个卫星的 ID
# - satellite1, satellite2: 第一个和第二个卫星的 Node 实例
def allocate_different_orbit_link(sat1id, sat2id, satellite1: Nodepy.Node, satellite2: Nodepy.Node):
    # 检查并分配链路至对应的激光终端
    if satellite1.linked_array[2] == -1:
        # 设定激光链路到 terminal 2 和 terminal 4
        satellite1.linked_array[2] = sat2id
        satellite1.set_value(2, [sat2id, 4])
        satellite2.linked_array[4] = sat1id
        satellite2.set_value(4, [sat1id, 2])
    else:
        # 若 terminal 2 已被占用，
        satellite1.linked_array[4] = sat2id
        satellite1.set_value(4, [sat2id, 2])
        satellite2.linked_array[2] = sat1id
        satellite2.set_value(2, [sat1id, 4])


# 地面链路分配
# 功能:
# 分配地面站与卫星之间的链路
# 参数:
# - satid: 卫星的 ID
# - gndid: 地面站的 ID
# - sat, gnd: 分别表示卫星和地面站的 Node 实例
def SatGndLink(satid, gndid, sat: Nodepy.Node, gnd: Nodepy.Node):
    # 设定卫星和地面站的链接
    sat.linked_array[0] = gndid
    sat.set_value(0, [gndid, 0])
    gnd.linked_array[0] = satid
    gnd.set_value(0, [satid, 0])

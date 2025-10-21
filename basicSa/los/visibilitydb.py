import sqlite3


from basicSa.los import  Sat2Gnd
from basicSa.utilis import db
from  basicSa.los import  satllite2satellite as sat2satellite

def custom_print(*args, **kwargs):
    print("visibility.py:", *args, **kwargs)

def timevisibility(time, stationnum, Nodes, db_name):
    visibility_results = visibilityfornode(stationnum, Nodes)

    # 连接到数据库
    conn = sqlite3.connect(db_name)

    for i, row in enumerate(visibility_results):
        print(f"i is {i}")
        for j, value in enumerate(row):
            print(f"j is {j}")

            if value == 1:
                distance = Sat2Gnd.GetDistance(Nodes[i], Nodes[j])
                db.insert_visibility_record(conn, time, i, j, distance)

    # 关闭数据库连接
    conn.close()

def timevisibilitynodes(conn,time, stationnum, Nodes, db_name):
    visibility_results = visibilityfornode(stationnum, Nodes)

    # 连接到数据库
    #conn = sqlite3.connect(db_name)

    for i, row in enumerate(visibility_results):
        print(f"i is {i}")
        for j, value in enumerate(row):
            print(f"j is {j}")

            if value == 1:
                distance = Sat2Gnd.GetDistance(Nodes[i], Nodes[j])
                db.insert_visibility_record(conn, time, i, j, distance)

    # 关闭数据库连接
   # conn.close()


# nodes always for the [time] nodes position
def visibilityfornode(stationnum, Nodes):
    # 初始化visibility_results为一个 len(Nodes) × len(Nodes) 的矩阵
    visibility_results = [[0 for _ in range(len(Nodes))] for _ in range(len(Nodes))]
    # for node in Nodes:
    #  custom_print(node)

    for i in range(len(Nodes)):
        node1 = Nodes[i]
      #  print(f"Node {i} - x: {node1.x}, y: {node1.y}, z: {node1.z}, angle: {node1.angle}")

        if i < stationnum:
            for j in range(stationnum, len(Nodes)):
                node2 = Nodes[j]


                # 调用 Sat2Gnd 的函数计算能见度
                visibility = Sat2Gnd.Ground_Sat(node1, node2)
            #    print(f"node[{i}][{j}]: visibility: {visibility}")
                visibility_results[i][j] = visibility
        else:
            for j in range(stationnum, len(Nodes)):
                if i != j:
                    node2 = Nodes[j]
                    # 调用 sat2satellite 的函数计算能见度
                    visibility = sat2satellite.Sat_Sat(node1, node2,i,j)

                    visibility_results[i][j] = visibility

    return visibility_results


def visibilityforallnodes(db_name,staionnodes,satellites):
    conn = sqlite3.connect(db_name)
    timelength = len(satellites[0])
    stationnum = len(staionnodes)
    for i in range(timelength):
        satenodes = []
        for nodes in satellites:
            satenodes.append(nodes[i])
        Nodes = staionnodes+satenodes

        timevisibilitynodes(conn, i, stationnum, Nodes, db_name)
     #   timevisibility(i, stationnum, Nodes, db_name)
    conn.close()



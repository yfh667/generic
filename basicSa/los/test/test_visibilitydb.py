from basicSa.los import satllite2satellite
import basicSa.los.Sat2Gnd as Sat2Gnd

import basicSa.utilis.Node as Node
import math
from basicSa.los import visibilitydb
##test

import numpy as np
import pandas as pd

from basicSa.utilis import db

import math
from basicSa.los  import visibilitydb

import  basicSa.utilis.readdata as  readdata

# test for the los between the nodes


def testforvisibility():
    station_positions = [
        (-3330.416038 * 1000, 5432.320718 * 1000, 279.864892 * 1000),
        (-1026.070264 * 1000, 5803.649126 * 1000, 2430.157814 * 1000),
        (-1159.388221 * 1000, 6270.180168 * 1000, 145.425828 * 1000),
        (1059.258741 * 1000, 6286.974615 * 1000, 179.818758 * 1000),
        (-1121.670064 * 1000, 5894.144937 * 1000, -2156.433001 * 1000)
    ]

    stations = [Node.Node(x, y, z, angle=math.radians(10)) for x, y, z in station_positions]

    # 初始化卫星节点
    satellite_positions = [
        (-1343.875841 * 1000, 6837.318990 * 1000, 0.000000 * 1000),
        (-3601.331032 * 1000, 5965.345593 * 1000, -0.000000 * 1000),
        (-1262.830211 * 1000, 6424.978201 * 1000, 2383.243215 * 1000),
        (1075.670610 * 1000, 6884.610809 * 1000, 0.000000 * 1000),
        (-1262.830211 * 1000, 6424.978201 * 1000, -2383.243215 * 1000)
    ]

    satellites = [Node.Node(x, y, z, angle=math.radians(30)) for x, y, z in satellite_positions]

    # 合并地面站和卫星节点到一个列表
    Nodes = stations + satellites
    stationnum = 5
    visibility_results = visibilitydb.visibilityfornode(stationnum, Nodes)

    print(visibility_results)
    # 初始化10x10矩阵
    # 初始化10x10矩阵
    num_nodes = 10
    matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # 填充矩阵
    for i in visibility_results:
        for j in visibility_results[i]:
            matrix[i][j] = visibility_results[i][j]

    # 转换为 DataFrame 以便更好地打印
    df = pd.DataFrame(matrix, columns=[f"Node {i}" for i in range(num_nodes)],
                      index=[f"Node {i}" for i in range(num_nodes)])

    # 设置 pandas 显示选项以显示所有列
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    # 打印结果
    print(df)


# we write the visibility result to the db,
# if you have some question,you need first try the above function
def testforwritetodb():
    station_positions = [
        (-3330.416038 * 1000, 5432.320718 * 1000, 279.864892 * 1000),
        (-1026.070264 * 1000, 5803.649126 * 1000, 2430.157814 * 1000),
        (-1159.388221 * 1000, 6270.180168 * 1000, 145.425828 * 1000),
        (1059.258741 * 1000, 6286.974615 * 1000, 179.818758 * 1000),
        (-1121.670064 * 1000, 5894.144937 * 1000, -2156.433001 * 1000)
    ]

    stations = [Node.Node(x, y, z, angle=math.radians(10)) for x, y, z in station_positions]

    # 初始化卫星节点
    satellite_positions = [
        (-1343.875841 * 1000, 6837.318990 * 1000, 0.000000 * 1000),
        (-3601.331032 * 1000, 5965.345593 * 1000, -0.000000 * 1000),
        (-1262.830211 * 1000, 6424.978201 * 1000, 2383.243215 * 1000),
        (1075.670610 * 1000, 6884.610809 * 1000, 0.000000 * 1000),
        (-1262.830211 * 1000, 6424.978201 * 1000, -2383.243215 * 1000)
    ]

    satellites = [Node.Node(x, y, z, angle=math.radians(30)) for x, y, z in satellite_positions]

    # 合并地面站和卫星节点到一个列表
    Nodes = stations + satellites
    pwd = "/home/yfh/Desktop/NS3/manswn/satpython/data/"
    db_name = pwd + 'simulation_data.db'
    db.initialize_database(db_name)
    # 示例调用
    time = 1
    stationnum = 5

    visibilitydb.timevisibility(time, stationnum, Nodes, db_name)

def testforfilewritetodb():

    stationpath = "/home/yfh/Desktop/Data/stations1"
    stationangle = 20
    stations = readdata.readstation(stationpath, stationangle)  # 获取地面站数据

    # 读取卫星初始位置
    satpath = "/home/yfh/Desktop/Data/sats1"
    satangle = 60
    satellites = readdata.readsats(satpath, satangle)  # 获取卫星数据

    # 合并地面站和卫星节点到一个列表

    pwd = "/home/yfh/Desktop/NS3/manswn/satpython/data/"
    db_name = pwd + 'simulation_data.db'
    db.initialize_database(db_name)



    visibilitydb.visibilityforallnodes(db_name,stations, satellites )




def testforvisibility():
    station_positions = [
        (-3330.416038 * 1000, 5432.320718 * 1000, 279.864892 * 1000),
        (-1026.070264 * 1000, 5803.649126 * 1000, 2430.157814 * 1000),
        (-1159.388221 * 1000, 6270.180168 * 1000, 145.425828 * 1000),
        (1059.258741 * 1000, 6286.974615 * 1000, 179.818758 * 1000),
        (-1121.670064 * 1000, 5894.144937 * 1000, -2156.433001 * 1000)
    ]

    stations = [Node.Node(x, y, z, angle=math.radians(10)) for x, y, z in station_positions]

    # 初始化卫星节点
    satellite_positions = [
        (-1343.875841 * 1000, 6837.318990 * 1000, 0.000000 * 1000),
        (-3601.331032 * 1000, 5965.345593 * 1000, -0.000000 * 1000),
        (-1262.830211 * 1000, 6424.978201 * 1000, 2383.243215 * 1000),
        (1075.670610 * 1000, 6884.610809 * 1000, 0.000000 * 1000),
        (-1262.830211 * 1000, 6424.978201 * 1000, -2383.243215 * 1000)
    ]

    satellites = [Node.Node(x, y, z, angle=math.radians(30)) for x, y, z in satellite_positions]

    # 合并地面站和卫星节点到一个列表
    Nodes = stations + satellites
    resullts = visibilitydb.visibilityfornode(len(stations), Nodes)
    return resullts

if __name__ == '__main__':
    # if you want to know the function of the visibilitydb ,you need first try the below function
   # testforvisibility()

    # below function it will create a db that contains the results of all the nodes ,sat-sat sat-station.
 #   testforwritetodb()
#    testforwritetodb()
    #testforfilewritetodb()
    re = testforvisibility()
    print(re)
    # resullts = testforvisibility()
    # resullts_array = np.array(resullts)
    # print(resullts_array)

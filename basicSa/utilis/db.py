# 初始化数据库和表（如果尚未创建）

def initialize_database(db_name):
    # 检查数据库文件是否存在
    if not os.path.exists(db_name):
        # 创建连接并初始化表
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS SimulationData (
            time INTEGER,
            basicSa INTEGER,
            dest INTEGER,
            distance DOUBLE
        )
        ''')
        conn.commit()
        conn.close()
        print(f"Database '{db_name}' created and table initialized.")
    else:
        print(f"Database '{db_name}' already exists. No need to create it.")

##test
##initialize_database( 'simulation_data.db')


# def insert_visibility_record(conn, time, basicSa, dest):
#     cursor = conn.cursor()
#     cursor.execute('INSERT INTO SimulationData (time, basicSa, dest) VALUES (?, ?, ?)', (time, basicSa, dest))
#     conn.commit()

def insert_visibility_record(conn, time, src, dest, distance):
    cursor = conn.cursor()
    cursor.execute('INSERT INTO SimulationData (time, basicSa, dest, distance) VALUES (?, ?, ?, ?)', (time, src, dest, distance))
    conn.commit()


import os
import sqlite3

def initialize_database_record(db_name):
    # 检查数据库文件是否存在
    if not os.path.exists(db_name):
        # 创建连接并初始化表
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS SimulationData (
            time INTEGER,
            nodeid INTEGER,
            x DOUBLE,
            y DOUBLE,
            z DOUBLE
        )
        ''')
        conn.commit()
        conn.close()
        print(f"Database '{db_name}' created and table initialized.")
    else:
        print(f"Database '{db_name}' already exists. No need to create it.")

# 示例调用
# db_name = 'simulation_data.db'
# initialize_database_record(db_name)


def insert_record_node(db_name, time, nodes):
    initialize_database_record(db_name)

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    for nodeid, node in enumerate(nodes):
        # 使用 get_position 方法获取 (x, y, z) 坐标
        x, y, z = node.get_position()

        cursor.execute('''
        INSERT INTO SimulationData (time, nodeid, x, y, z) VALUES (?, ?, ?, ?, ?)
        ''', (time, nodeid, x, y, z))

    conn.commit()
    conn.close()
    print(f"Inserted {len(nodes)} records into the database.")


# 示例调用
# db_name = 'simulation_data.db'
# time = 10  # Example time
# nodes = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]  # Example node data
# insert_record_node(db_name, time, nodes)

# conn = sqlite3.connect('simulation_data.db')
# insert_visibility_record(conn, 1, 2, 6)


def get_node_visibility(db_name, time):
    # 连接到数据库
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 查询给定时间的可见性数据
    cursor.execute('''
    SELECT basicSa, dest FROM SimulationData WHERE time = ?
    ''', (time,))

    # 初始化nodevisibility字典
    nodevisibility = {}

    # 处理查询结果
    rows = cursor.fetchall()

    for src, dest in rows:
        if src not in nodevisibility:
            nodevisibility[src] = set()
        nodevisibility[src].add(dest)

    # 关闭数据库连接
    conn.close()

    return nodevisibility


# #
# pwd = '/home/yfh/Desktop/NS3/manswn/satpython/data/'
# db_name =pwd+ 'simulation_data.db'
# time =1
# nodevisibility = get_node_visibility(db_name, time)
# print(nodevisibility[0])

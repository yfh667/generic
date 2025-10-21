import basicSa.Dataprocessing.readns3file as readns3file

import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'
import matplotlib.pyplot as plt

file_paths = '/home/yfh/Desktop/Data/7links_0.08_3000.xml'
db = readns3file.readxml(file_paths)  # 获取地面站数据

totaltime = len(db.snapshots)

stationnum = 8
OrbitNum = 18  # 总轨道数（原变量P）
SatPerOrbit = 36  # 每个轨道的卫星数（原变量N）
#
# def plot(end,db):
#    # end = 1
#     # 初始化数据存储
#     times, path_counts, orbit_counts = [], [], []
#     lastpath = []
#     for i in range(totaltime):
#         paths = db.snapshots[i].active_paths
#
#         # 初始化默认值
#         path_count = 0
#         orbit_count = 0
#
#         if end in paths and len(paths[end]) > 0:
#             # 提取路径列表（假设路径字典结构为 paths = {1: [{'path': [...]}]}）
#             path_list = paths[end][0]['path']
#             path_count = len(path_list) - 1
#             if not lastpath:
#                 lastpath = path_list
#             else:
#                 if lastpath!=path_list:
#
#
#             # 计算涉及的轨道数量（使用之前定义的 count_involved_orbits 函数）
#             orbit_count = count_involved_orbits(path_list)
#
#         # 记录数据
#         times.append(i)
#         path_counts.append(path_count)
#         orbit_counts.append(orbit_count)
#         print(f"Time: {i}, Paths: {path_count}, Orbits: {orbit_count}")


def plot(end, db):
    # 初始化数据存储
    times, path_counts, orbit_counts = [], [], []
    change_times = []  # 专门记录路径变化的时间点
    lastpath = []

    for i in range(totaltime):
        paths = db.snapshots[i].active_paths

        # 初始化默认值
        path_count = 0
        orbit_count = 0

        if end in paths and len(paths[end]) > 0:
            path_list = paths[end][0]['path']
            path_count = len(path_list) - 1  # 路径跳数

            # 检测路径变化
            if not lastpath:
                lastpath = path_list
                change_times.append(i)  # 初始路径记录为第一次变化
            else:
                if lastpath != path_list:
                    change_times.append(i)  # 记录变化时间点
                    lastpath = path_list

            orbit_count = count_involved_orbits(path_list)

        # 记录数据
        times.append(i)
        path_counts.append(path_count)
        orbit_counts.append(orbit_count)

    # 绘制双轴曲线图
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制路径跳数曲线（左轴）
    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Path Hops', color=color)
    ax1.plot(times, path_counts, color=color, label='Path Hops')
    ax1.tick_params(axis='y', labelcolor=color)

    # 绘制轨道数量曲线（右轴）
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Orbit Count', color=color)
    ax2.plot(times, orbit_counts, color=color, linestyle='--', label='Orbit Count')
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加变化竖线
    for t in change_times:
        ax1.axvline(x=t, color='gray', linestyle=':', alpha=0.7, linewidth=1)

    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Path Change Analysis with Transition Markers')
    plt.grid(alpha=0.3)
    plt.show()
    # 创建双轴图表
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制路径数量（左轴）
    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Number of Paths', color=color)
    ax1.plot(times, path_counts, 'b-', label='Paths')
    ax1.tick_params(axis='y', labelcolor=color)

    # 创建右轴绘制轨道数量
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Number of Orbits', color=color)
    ax2.plot(times, orbit_counts, 'r--', label='Orbits')
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加标题和图例
    plt.title('Path Count and Involved Orbits over Time')
    fig.tight_layout()
    plt.grid(True)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.show()


def count_involved_orbits(path_dicts):
    """
    计算路径中涉及的轨道数量
    :param path_dicts: 包含卫星节点ID的可迭代对象（如列表、字典键等）
    :return: 涉及的轨道数量
    """
    involved_orbits = set()  # 使用集合自动去重

    for node_id in path_dicts:
        if node_id >= stationnum:  # 卫星节点（地面站ID 0-5）
            sat_id = node_id - stationnum
            orbit_id = sat_id // SatPerOrbit  # 轨道编号从0开始

            # 校验轨道编号合法性
            if orbit_id < 0 or orbit_id >= OrbitNum:
                raise ValueError(f"卫星 {node_id} 的轨道编号 {orbit_id} 超出范围")

            involved_orbits.add(orbit_id)

    return len(involved_orbits)

plot(1,db)
plot(3,db)
plot(5,db)
plot(7,db)

import basicSa.Dataprocessing.readns3file as readns3file
import basicSa.Dataprocessing.readpyfile as readpyfile
import basicSa.solvedata.plots.kde as kde
import basicSa.solvedata.plots.cdf as cdf

import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# 设置中文字体

file_paths = '/home/yfh/Desktop/Data/simulation_paths.xml'
db = readpyfile.readxml(file_paths)  # 获取地面站数据

totaltime = len(db.snapshots)

stationnum = 8
OrbitNum = 18  # 总轨道数（原变量P）
SatPerOrbit = 36  # 每个轨道的卫星数（原变量N）
#
def plot_interval_distribution(intervals):
    """
    直接统计时间间隔分布（避免 KDE 的负值问题）
    :param intervals: 时间间隔列表，单位秒
    """
    if len(intervals) == 0:
        print("无数据可绘制")
        return

    # 过滤非正值（根据实际需求调整）
    intervals = [x for x in intervals if x > 0]

    # 计算最大间隔，动态设定分组区间
    max_interval = max(intervals)
    bins = np.arange(0, max_interval + 1, 1)  # 1秒为一个区间

    # 统计频数
    hist, bin_edges = np.histogram(intervals, bins=bins)

    # 归一化为概率
    prob = hist / hist.sum()

    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    plt.bar(bin_edges[:-1], prob, width=np.diff(bin_edges), align='edge', alpha=0.7)

    # 标记数据点（可选）
    plt.scatter(intervals, np.zeros_like(intervals), color='red', marker='x', label='实际间隔')

    plt.xlabel('时间间隔 (秒)')
    plt.ylabel('概率')
    plt.title('路径变更时间间隔分布')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()


# # 在 plot 函数中调用



def plot(end, db):
    # 初始化数据存储
    times, path_counts, orbit_counts = [], [], []
    change_times = []  # 专门记录路径变化的时间点
    lastpath = []
    lasttime =0

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


    qujian = []
    for i in range(len(change_times)-1):
        qujian.append(change_times[i+1]-change_times[i])

    #kde.plot_kde_curves(qujian, labels=None)
    # 绘制 KDE 曲线

    cdf.plot_cdf(qujian)
    # qujian = np.diff(change_times)
    # plot_interval_distribution(qujian)

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
# plot(3,db)
# plot(5,db)
# plot(7,db)

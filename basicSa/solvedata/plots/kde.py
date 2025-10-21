import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


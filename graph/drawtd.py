from vedo import *
import numpy as np


def plot_multi_layer_topology(P,N, t):
    """
    绘制多层拓扑结构

    参数:
    N: int, 拓扑的长度
    P: int, 拓扑的宽度
    t: int, 层数
    """
    # 创建一个Plotter对象用于显示
    plt = Plotter(N=1, axes=1, bg='white')

    # 定义参数
    spacing = 1.0  # 网格点之间的间距
    layer_height = 1.0  # 层与层之间的间距设置为1

    # 创建所有层的点和平面
    all_points = []
    all_planes = []

    # 设置统一的颜色
    point_color = 'blue'  # 所有点使用红色
    plane_color = 'lightgray'  # 所有平面使用浅灰色

    for layer in range(t):
        # 创建当前层的点
        points = []
        for i in range(N):
            for j in range(P):
                x = i * spacing
                y = j * spacing
                z = layer * layer_height  # 每层高度为1的整数倍
                points.append([x, y, z])

        # 将点转换为numpy数组
        points = np.array(points)

        # 创建点的可视化，使用统一的颜色
        spheres = Points(points, r=10, c=point_color)
        all_points.append(spheres)

        # 创建背景平面
        plane_size_x = (N - 1) * spacing + 1
        plane_size_y = (P - 1) * spacing + 1
        center_x = (N - 1) * spacing / 2
        center_y = (P - 1) * spacing / 2
        center_z = layer * layer_height - 0.1  # 平面略低于点所在的高度

        # 使用Plane而不是Rectangle
        plane = Plane(
            normal=(0, 0, 1),  # 平面朝上
            s=(plane_size_x, plane_size_y),  # 平面大小
            c=plane_color,  # 统一的平面颜色
            alpha=0.3,  # 透明度
        ).pos(center_x, center_y, center_z)  # 设置位置

        all_planes.append(plane)

    # 将所有对象添加到场景中
    for plane in all_planes:  # 先添加平面
        plt += plane
    for points in all_points:  # 再添加点
        plt += points

    # 设置相机位置以获得更好的视角
    plt.camera.SetPosition(10, 10, 10)
    plt.camera.SetFocalPoint(2, 2, 2)

    # 显示场景
    plt.show()


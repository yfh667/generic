from vedo import *
import numpy as np

def apply_region_colors(P, N, t, regions):
    """
    给多层拓扑的指定区域上色

    参数:
    P: int, 拓扑的宽度
    N: int, 拓扑的长度
    t: int, 层数
    regions: dict, 区域字典，包含区域号及区域的点的索引，指定不同的颜色
    """
    # 调用之前的绘制函数，获取到多层拓扑结构
    plt, all_points, all_coordinates = plot_multi_layer_topology(P, N, t)

    # 根据regions字典中的区域进行上色
    color_map = {
        0: 'red',
        1: 'green',
        2: 'yellow',

        # 你可以根据需要为其他区域指定颜色
    }
    for z_layer, layers in regions.items():  # z_layer 是层级，layers 是该层的区域列表
        for region_id, region in enumerate(layers):  # region_id 是区域编号，region 是该区域的点索引
            color = color_map.get(region_id, 'black')  # 获取颜色
            for index in region:
                # 计算在当前层中的实际索引
                actual_index = z_layer * (N * P) + index
                if actual_index < len(all_coordinates):
                    x, y, z = all_coordinates[actual_index]
                    points_obj = Points([[x, y, z]], r=10, c=color)  # 给点上色
                    all_points.append(points_obj)

    # 将所有点添加到场景中
    for points in all_points:  # 添加点
        plt += points

    # 设置相机位置以获得更好的视角
    plt.camera.SetPosition(10, 10, 10)
    plt.camera.SetFocalPoint(2, 2, 2)

    # 显示场景
    plt.show()

def get_region_color(region_id):
    """
    根据区域编号返回相应的颜色
    """
    color_map = {
        0: 'red',
        1: 'green',
        2: 'yellow',

        # 你可以根据需要为其他区域指定颜色
    }
    return color_map.get(region_id, 'blue')  # 默认颜色为蓝色

def plot_multi_layer_topology(P, N, t):
    """
    绘制多层拓扑结构，返回绘图对象和所有点的信息

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
    all_coordinates = []  # 存储所有点的坐标

    # 创建所有层的点
    for layer in range(t):
        points = []
        for i in range(N):
            for j in range(P):
                x = i * spacing
                y = j * spacing
                z = layer * layer_height  # 每层高度为1的整数倍
                points.append([x, y, z])
                all_coordinates.append([x, y, z])  # 存储点的坐标

        # 将点转换为numpy数组
        points = np.array(points)

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
            c='lightgray',  # 统一的平面颜色
            alpha=0.3,  # 透明度
        ).pos(center_x, center_y, center_z)  # 设置位置

        all_planes.append(plane)

        # 默认所有点为蓝色
        points_obj = Points(points, r=10, c="blue")
        all_points.append(points_obj)

    # 将平面和点对象返回
    for plane in all_planes:  # 先添加平面
        plt += plane
    for points in all_points:  # 再添加点
        plt += points

    return plt, all_points, all_coordinates

# 示例：区域字典，每个区域由多个子区域（每个子区域包含点的索引）组成，指定不同的颜色
regions = {
    0: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]],  # z=0的拓扑上色，包含3个区域
    1: [[0, 1, 2, 4], [5, 6, 7], [8, 9]],  # z=1的拓扑上色，包含3个区域
    2: [[0, 1, 2, 4], [5, 6, 7], [15, 10]],  # z=1的拓扑上色，包含3个区域
}

# 调用函数，绘制图形并上色
apply_region_colors(5, 5, 10, regions)

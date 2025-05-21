from vedo import *
#from vedo import vtk      # ← 新增

import numpy as np
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

# 定义全局参数
SPACING = 1.0  # 网格点之间的间距
LAYER_HEIGHT = 1.0  # 层与层之间的间距


def add_dashed_connections(plt, connections):
    """
    在已有拓扑结构上添加虚线连接

    参数:
    plt: Plotter对象
    connections: list of lists, 每个子列表包含两个三维坐标点，表示需要连接的点
    """
    for connection in connections:
        if len(connection) == 2:
            start_point, end_point = connection
            # 创建虚线连接
            line = DashedLine(start_point, end_point, c='red', lw=2, spacing=0.05)  # spacing调整虚线疏密
            plt += line  # 将虚线添加到绘图对象
    return plt

def calculate_coordinates(layer, point, N, P):
    """
    根据层级、点编号计算坐标。

    参数:
    layer: int, 当前层级
    point: int, 点编号（从 0 到 N*P-1）
    N: int, 拓扑的宽度
    P: int, 拓扑的长度

    返回：
    计算得到的三维坐标 [x, y, z]
    """
    x = (point % N) * SPACING
    y = (point // N) * SPACING
    z = layer * LAYER_HEIGHT  # 根据层数设置z坐标
    return (x, y, z)


def apply_region_colors(plt, P, N, t, regions, all_coordinates):
    """
    给多层拓扑的指定区域上色。
    注意：此函数直接在传入的 plt 对象上添加新的彩色点，而不是修改原始点。

    参数:
    plt: Plotter对象，将在其上添加彩色点
    P: int, 拓扑的宽度
    N: int, 拓扑的长度
    t: int, 层数
    regions: dict, 区域字典 {层索引: [[点索引列表1], [点索引列表2], ...]}
    all_coordinates: list, 所有点的原始坐标列表
    """
    colored_points_to_add = []  # 存储待添加的彩色点对象

    for z_layer, layer_regions_data in regions.items():  # z_layer 是层级，layer_regions_data 是该层的区域列表
        if not (0 <= z_layer < t):  # 检查层索引是否有效
            print(f"警告: 区域定义中的层索引 {z_layer} 超出范围 [0, {t - 1}]，已跳过。")
            continue
        for region_id, region_indices in enumerate(layer_regions_data):  # region_id 是区域编号，region_indices 是该区域的点索引
            color = get_region_color(region_id)  # 获取颜色
            for point_index_in_layer in region_indices:
                # 计算在 all_coordinates 中的实际索引
                # 每层有 N * P 个点
                if not (0 <= point_index_in_layer < N * P):  # 检查点索引是否有效
                    print(
                        f"警告: 层 {z_layer} 区域 {region_id} 中的点索引 {point_index_in_layer} 超出范围 [0, {N * P - 1}]，已跳过。")
                    continue

                actual_index = z_layer * (N * P) + point_index_in_layer
                if actual_index < len(all_coordinates):
                    coords = all_coordinates[actual_index]
                    # 创建一个新的 Points 对象来表示这个彩色点
                    # r=12 使其比原来的点稍大，更容易看到
                    point_obj = Points([coords], r=12, c=color)
                    colored_points_to_add.append(point_obj)
                else:
                    print(f"警告: 计算得到的实际索引 {actual_index} 超出坐标列表长度 {len(all_coordinates)}，已跳过。")

    # 将所有区域着色的点添加到场景中
    for point_obj in colored_points_to_add:
        plt += point_obj

    return plt


def get_region_color(region_id):
    """
    根据区域编号返回相应的颜色
    """
    color_map = {
        0: 'red',  # 红色
        1: 'green',  # 绿色
        2: 'yellow',  # 黄色
        # 你可以根据需要为其他区域指定颜色
    }
    return color_map.get(region_id, 'purple')  # 默认颜色为紫色
#
def plot_multi_layer_topology(P, N, t,z_down=True):
    """
    绘制多层拓扑结构，返回绘图对象和所有点的信息

    参数:
    N: int, 拓扑的长度
    P: int, 拓扑的宽度
    t: int, 层数
    """
    # 创建一个Plotter对象用于显示
    plt = Plotter(
        axes=1,  # 显示坐标轴 (模式1：简单坐标轴)
        bg='white',  # 背景颜色设置为白色
        interactive=True  # 确保启用交互
    )

    # 设置渲染器属性以优化性能
    renderer = plt.renderer
    if hasattr(renderer, 'SetUseFXAA'):
        renderer.SetUseFXAA(True)  # 启用FXAA抗锯齿
    if hasattr(renderer, 'SetUseDepthPeeling'):
        renderer.SetUseDepthPeeling(True)  # 启用深度剥离以改善透明度渲染
        renderer.SetMaximumNumberOfPeels(4)  # 设置最大剥离层数
        renderer.SetOcclusionRatio(0.0)  # 设置遮挡比率

    # 设置交互样式为轨迹球相机模式，允许鼠标拖动旋转、平移、缩放
    style = vtkInteractorStyleTrackballCamera()
    plt.interactor.SetInteractorStyle(style)

    all_initial_points_objects = []  # 存储每层原始点（蓝色）的 Points 对象
    all_planes = []  # 存储每层的背景平面
    all_coordinates = []  # 存储所有点的三维坐标

    # 创建所有层的点和平面
    for layer_idx in range(t):  # 遍历每一层
        points_coords_in_layer = []  # 当前层的点坐标列表
        for i in range(N):  # 遍历长度方向
            for j in range(P):  # 遍历宽度方向
                x = i * SPACING
                y = j * SPACING
                z = layer_idx * LAYER_HEIGHT  # 每层高度为 LAYER_HEIGHT 的整数倍
               # z = -layer_idx * LAYER_HEIGHT if z_down else layer_idx * LAYER_HEIGHT

                points_coords_in_layer.append([x, y, z])
                all_coordinates.append([x, y, z])  # 存储所有点的坐标

        points_coords_in_layer_np = np.array(points_coords_in_layer)  # 转换为numpy数组

        # 创建背景平面
        # 平面尺寸略大于点阵范围，确保点不会画在平面边缘外
        plane_size_x = (N - 1) * SPACING + SPACING if N > 0 else SPACING
        plane_size_y = (P - 1) * SPACING + SPACING if P > 0 else SPACING

        # 平面中心位置
        center_x = (N - 1) * SPACING / 2 if N > 0 else 0
        center_y = (P - 1) * SPACING / 2 if P > 0 else 0
        center_z = layer_idx * LAYER_HEIGHT - 0.05 * LAYER_HEIGHT  # 平面略低于点所在的高度，避免Z-fighting
      #  center_z = (-layer_idx if z_down else layer_idx) * LAYER_HEIGHT - 0.05 * LAYER_HEIGHT

        plane = Plane(
            pos=(center_x, center_y, center_z),  # 设置平面位置
            normal=(0, 0, 1),  # 平面法向量朝上 (z轴正方向)
           # normal=(0, 0, -1 if z_down else 1),

            s=(plane_size_x, plane_size_y),  # 平面大小 (x方向尺寸, y方向尺寸)
            c='lightgray',  # 平面颜色
            alpha=0.3,  # 透明度
        )
        all_planes.append(plane)

        # 创建当前层的所有点对象（默认为蓝色）
        if len(points_coords_in_layer_np) > 0:  # 确保有坐标点才创建Points对象
            points_obj = Points(points_coords_in_layer_np, r=10, c="lightgray")  # r是点半径
            all_initial_points_objects.append(points_obj)

    # 将所有平面和点对象添加到绘图场景中
    # 先添加平面，再添加点，可以避免点被平面遮挡（如果平面不透明）
    for plane_obj in all_planes:
        plt += plane_obj
    for points_obj in all_initial_points_objects:
        plt += points_obj

    return plt, all_initial_points_objects, all_coordinates



# # --- 示例使用 ---
# if __name__ == "__main__":
#     # 定义拓扑结构参数
#     P_val, N_val, t_val = 5, 5, 10  # 宽度P, 长度N, 层数t
#
#     # 定义区域上色信息
#     # 结构: {层索引: [[区域1点索引], [区域2点索引], ...]}
#     # 点索引是基于单层内的索引 (0 到 N*P-1)
#     regions_to_color = {
#         0: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]],  # z=0 (第0层) 的拓扑上色
#         1: [[0, 1, 2, 4], [5, 6, 7, 11], [12, 13, 14]],  # z=1 (第1层) 的拓扑上色
#         2: [[0, 1, 2, 4], [5, 6, 7, 15], [16, 17, 10]],  # z=2 (第2层) 的拓扑上色
#     }
#
#     # 1. 创建基本的多层拓扑结构
#     # plot_multi_layer_topology 返回 plotter 对象, 原始点对象列表, 和所有坐标
#     main_plotter, original_points_objs, all_coords = plot_multi_layer_topology(P_val, N_val, t_val)
#
#     # 2. 对指定区域进行上色
#     # apply_region_colors 会在 main_plotter 上添加新的彩色点
#     main_plotter = apply_region_colors(main_plotter, P_val, N_val, t_val, regions_to_color, all_coords)
#
#     # 3. 定义需要连接的点（使用实际三维坐标）
#     # 确保这些坐标与 SPACING 和 LAYER_HEIGHT 的设置一致
#     connections_list = [
#         [(0, 0, 0), (0, 0, 1)],  # 垂直连接
#         [(1, 1, 1), (1, 1, 2)],  # 跨层连接
#         [(2, 2, 0), (2, 2, 2)],  # 同层对角线连接
#     ]
#
#     # 4. 添加虚线连接
#     main_plotter = add_dashed_connections(main_plotter, connections_list)
#
#
#
#     # 6. 显示绘图窗口
#     # viewup="z" 表示Z轴朝上，这对于分层结构通常更自然
#     # title 设置窗口标题
#     print("Vedo 窗口即将显示。请使用鼠标进行交互：")
#     print("- 按住左键拖动：旋转")
#     print("- 按住中键拖动 (或 Shift + 左键拖动)：平移")
#     print("- 滚轮 (或 右键拖动)：缩放")
#     main_plotter.show(viewup="z", title="可交互3D拓扑图")

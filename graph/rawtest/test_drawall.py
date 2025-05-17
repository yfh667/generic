

import graph.drawall as drawall

if __name__ == "__main__":
    # 定义拓扑结构参数
    P_val, N_val, t_val = 5, 5, 10  # 宽度P, 长度N, 层数t

    # 定义区域上色信息
    # 结构: {层索引: [[区域1点索引], [区域2点索引], ...]}
    # 点索引是基于单层内的索引 (0 到 N*P-1)
    regions_to_color = {
        0: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]],  # z=0 (第0层) 的拓扑上色
        1: [[0, 1, 2, 4], [5, 6, 7, 11], [12, 13, 14]],  # z=1 (第1层) 的拓扑上色
        2: [[0, 1, 2, 4], [5, 6, 7, 15], [16, 17, 10]],  # z=2 (第2层) 的拓扑上色
    }

    # 1. 创建基本的多层拓扑结构
    # plot_multi_layer_topology 返回 plotter 对象, 原始点对象列表, 和所有坐标
    main_plotter, original_points_objs, all_coords = drawall.plot_multi_layer_topology(P_val, N_val, t_val)

    # 2. 对指定区域进行上色
    # apply_region_colors 会在 main_plotter 上添加新的彩色点
    main_plotter = drawall.apply_region_colors(main_plotter, P_val, N_val, t_val, regions_to_color, all_coords)

    # 3. 定义需要连接的点（使用实际三维坐标）
    # 确保这些坐标与 SPACING 和 LAYER_HEIGHT 的设置一致
    connections_list = [
        [(0, 0, 0), (1, 2, 4)],  # 垂直连接
        [(1, 1, 1), (1, 1, 2)],  # 跨层连接
        [(2, 2, 0), (2, 2, 2)],  # 同层对角线连接
    ]

    # 4. 添加虚线连接
    main_plotter = drawall.add_dashed_connections(main_plotter, connections_list)



    # 6. 显示绘图窗口
    # viewup="z" 表示Z轴朝上，这对于分层结构通常更自然
    # title 设置窗口标题
    print("Vedo 窗口即将显示。请使用鼠标进行交互：")
    print("- 按住左键拖动：旋转")
    print("- 按住中键拖动 (或 Shift + 左键拖动)：平移")
    print("- 滚轮 (或 右键拖动)：缩放")

    main_plotter.show(viewup="z", title="可交互3D拓扑图")

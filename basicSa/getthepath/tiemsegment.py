def calculate_time_segments(paths):
    """
    计算时间间隔分段

    参数:
        paths: 字典，包含各个路径的时间区间
        示例:
        {
            'path1': [(0, 55), (70, 100)],
            'path2': [(0, 20), (44, 100)],
            'path3': [(0, 40), (50, 100)]
        }

    返回:
        排序后的时间间隔列表
    """
    # 1. 收集所有时间点
    time_points = set()
    for i in range(len(paths)):
        for interval in paths[i]['time_intervals']:
            start, end = interval
            time_points.add(start)
            time_points.add(end)
    # for path_intervals in paths.values():
    #     for interval in path_intervals:
    #         basicSa, end = interval
    #         time_points.add(basicSa)
    #         time_points.add(end)

    # 2. 去重并排序
    sorted_time_points = sorted(time_points)

    # 3. 创建时间间隔
    time_segments = []
    for i in range(len(sorted_time_points) - 1):
        start = sorted_time_points[i]
        end = sorted_time_points[i + 1]
        time_segments.append((start, end))

    return time_segments


# # 示例数据
# paths = {
#     'path1': [(0, 55), (70, 100)],
#     'path2': [(0, 20), (44, 100)],
#     'path3': [(0, 40), (50, 100)]
# }
#
# paths = [    {'path1': [0,  22.0, 56, 19, 53, 16, 50, 13, 47, 10, 44, 43, 77, 1], 'intervals': [[0, 55], [70, 100]] },
#              {    'path2': [0, 22.0, 56, 19, 53, 16, 50, 13, 47, 10, 44, 43, 77, 1], 'intervals': [[0, 20], [44, 100]],
# },{    'path3': [0, 22.0, 56, 19, 53, 16, 50, 13, 47, 10, 44, 43, 77, 1], 'intervals': [[0, 40], [50, 100]],
# }
# ]
#
#
#
#
# #paths
# #
#
# # 计算时间间隔
# segments = calculate_time_segments(paths)
#
# # 打印结果
# print("计算得到的时间间隔:")
# for seg in segments:
#     print(seg)

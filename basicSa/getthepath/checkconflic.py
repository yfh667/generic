def check_path_conflict(path1_nodes, path2_nodes):
    """
    检查两条路径在卫星节点上的ISL是否冲突
    参数:
        path1_nodes: 第一条路径的节点列表，如 [1, 2, 3, 4]
        path2_nodes: 第二条路径的节点列表，如 [1, 7, 6, 4]
    返回:
        int: 1表示冲突，0表示不冲突
    """
    # 获取卫星节点（排除头尾的地面节点）
    sat1 = path1_nodes[1:-1]  # 第一条路径的卫星节点
    sat2 = path2_nodes[1:-1]  # 第二条路径的卫星节点

    # 找出共有的卫星节点
    common_sats = set(sat1) & set(sat2)
    return 1 if common_sats else 0


def build_conflict_matrix(paths):
    """
    构建冲突矩阵（仅考虑路径节点冲突，不考虑时间）
    返回格式：
    {
        0: {0: 0, 1: 1, 2: 0},  # 路径0 对比 自己(0) / 路径1(有冲突) /路径2(无冲突)
        1: {0: 1, 1: 0, 2: 0},  # 路径1 对比 路径0(有冲突)
        2: {0: 0, 1: 0, 2: 0}   # 路径2 和其他路径无冲突
    }
    """
    num_paths = len(paths)
    conflict_matrix = {
        p1_idx: {
            p2_idx: 1 if (p1_idx != p2_idx and check_path_conflict(paths[p1_idx]["path"], paths[p2_idx]["path"]))
                   else 0
                   for p2_idx in range(num_paths)
                   }
        for p1_idx in range(num_paths)
    }
    return conflict_matrix

#
# # 使用示例
# if __name__ == "__main__":
#     # 测试数据（使用数字索引）
#     paths = [{'path': [1, 2,3, 4], 'intervals': [[0, 55], [70, 100]]},
#              {'path': [1,2,5,4], 'intervals': [[0, 20], [44, 100]],
#               }, {'path': [1,7,6,4], 'intervals': [[0, 40], [50, 100]],
#                   }
#              ]
#
#     conflict_matrix = build_conflict_matrix(paths)
#
#     # 测试冲突检测
#     print("冲突矩阵:")
#     for p1 in sorted(conflict_matrix):
#         for p2 in sorted(conflict_matrix[p1]):
#             print(f"路径{p1}和路径{p2}: {'冲突' if conflict_matrix[p1][p2] else '不冲突'}")
#
#     # 快速查询示例
#     print("\n快速查询示例:")
#     print("路径0和路径1是否冲突:", conflict_matrix[0][1])  # 输出: 1
#     print("路径1和路径2是否冲突:", conflict_matrix[1][2])  # 输出: 0

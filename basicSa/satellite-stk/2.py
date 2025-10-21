import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import networkx as nx


def draw_grid_with_adjacency_tuples(N, P, adjacency):
    """
    根据给定的 N 和 P，先创建 (N+1)*(P+1) 网格节点 (以元组 (i,j) 形式)。
    然后根据 adjacency（邻接字典，key/value 均为 (i,j)）把指定节点之间用线连起来。

    这里 (i, j) 含义：i 表示列号, j 表示行号。
    """

    # ------------------- 1) 构建所有节点的坐标 (pos) -------------------
    pos = {}

    # 1.1) 原始 N 行 * P 列 节点
    #      行号 j 从 1 到 N
    #      列号 i 从 1 到 P
    for j in range(0, N):
        for i in range(0, P):
            pos[(i, j)] = (i, -j)  # x = i, y = -j（让行号越大越往下）

    # 1.2) 添加第 N+1 行 (物理上确实在最下面)
    #      对应行号 j = N+1, 列号 i 从 1 到 P
    for i in range(0, P):
        pos[(i, N)] = (i, -(N))

    # 1.3) 添加第 P+1 列 (物理上确实在最右边)
    #      对应列号 i = P+1, 行号 j 从 1 到 N
    # for j in range(1, N + 1):
    #     pos[(P + 1, j)] = (P + 1, -j)

    # --------------------- 2) 绘制所有节点 ----------------------------
    plt.figure(figsize=(16, 16))
    for (i, j), (x, y) in pos.items():
        # 先画出散点
        plt.scatter(x, y, color="blue")

        # 决定文字标签的显示方式：
        # 如果是第 N+1 行，提示它是“第 1 行”的复制
        if j == N and 0 <= i <= P - 1:
            plt.text(x, y, f"({i},{1}_row) ", fontsize=8,
                     ha="right", va="bottom")
        # 如果是第 P+1 列，提示它是“第 1 列”的复制
        # elif i == P+1 and 1 <= j <= N:
        #     plt.text(x, y, f"({1},{j})", fontsize=8,
        #              ha="right", va="bottom")
        else:
            # 普通点就正常标
            plt.text(x, y, f"({i},{j})", fontsize=8,
                     ha="right", va="bottom")

    # ---------------- 3) 根据 adjacency 绘制连线 ----------------------
    for node, neighbors in adjacency.items():
        # 如果 adjacency 里出现了不存在于 pos 的节点，则跳过或报错
        if node not in pos:
            continue
        x1, y1 = pos[node]
        for neighbor in neighbors:
            if neighbor not in pos:
                continue
            x2, y2 = pos[neighbor]
            plt.plot([x1, x2], [y1, y2], color='black', linewidth=1)

    # --------------------- 4) 收尾并显示 ------------------------------
    plt.title(f"Satellite Grid with N={N}, P={P}  ")
    plt.axis("off")
    plt.show()


def directed_to_undirected(directed_graph):
    """
    将有向图的邻接表转换为无向图的邻接表。

    参数:
        directed_graph (dict): 有向图邻接表，形式类似于:
            {
                (1, 1): [(1, 2), (2, 1)],
                (1, 2): [(1, 1)],
                (2, 1): [(1, 1)],
                ...
            }

    返回:
        dict: 无向图邻接表
    """
    undirected_graph = {}

    for node, neighbors in directed_graph.items():
        # 确保 undirected_graph 中有 node 的条目
        if node not in undirected_graph:
            undirected_graph[node] = []

        for neighbor in neighbors:
            # neighbor 也应该在 undirected_graph 中有条目
            if neighbor not in undirected_graph:
                undirected_graph[neighbor] = []

            # 将 neighbor 加入 node 的邻接列表
            if neighbor not in undirected_graph[node]:
                undirected_graph[node].append(neighbor)

            # 将 node 加入 neighbor 的邻接列表
            if node not in undirected_graph[neighbor]:
                undirected_graph[neighbor].append(node)

    return undirected_graph


# ---------------------------
# 使用示例
if __name__ == "__main__":
    path = r'E:\STK_file\angular'
    N = 3
    P = 4
    # # we get all the yigui
   # angular_velocity = readtxt.get_angular(path, N, P)

        # # 假设你想要 3 行、4 列
        # # 并让 (1,1) 与 (1,2)、(2,1) 连线
    adjacency_example = {
        (1, 1): [(1, 2), (2, 1)],
        (1, 2): [(1, 1)],
        (2, 1): [(1, 1)],
        # 其他没列出的节点则不连线
    }

    draw_grid_with_adjacency_tuples(N, P, adjacency=adjacency_example)

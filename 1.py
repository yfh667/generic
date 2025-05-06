import networkx as nx
import matplotlib.pyplot as plt

# 给定的参数
P = 3
N = 3

# 创建图对象
G = nx.Graph()

# 添加节点并标注坐标信息
for i in range(P * N):
    # 计算对应的二维坐标
    x = i // N
    y = i % N
    G.add_node(i, pos=(x, y))
    print(f"节点 {i} 对应坐标 ({x}, {y})")

# 定义邻接表（完整连接关系）
adjacency_list = {
    0: [1, 6],
    1: [0, 2, 3],
    2: [1, 5],
    3: [0, 4,7],
    4: [ 3, 5],
    5: [2, 4,8]
}

# 添加边
for node, neighbors in adjacency_list.items():
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

# 将邻接表转换为右邻居数组（仅记录右邻居）
a = [-1] * (P -1)* N  # 初始化数组，-1表示无右邻居)


for node, neighbors in adjacency_list.items():
    orbit = node//N
    for i in neighbors:
        negighbor_orbit=i//N
        if orbit != negighbor_orbit and negighbor_orbit> orbit :
            a[node]=i


def plot_graph_with_curved_edges(adjacency_list, N, P):
    G = nx.Graph()

    # 添加节点并标注坐标（保持原始网格布局）
    for i in range(P * N):
        x = i // N
        y = i % N
        G.add_node(i, pos=(x, y))

    # 添加所有边
    for node, neighbors in adjacency_list.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    # 获取节点坐标
    pos = nx.get_node_attributes(G, 'pos')

    # 绘制图形
    plt.figure(figsize=(8, 6))

    # 先绘制所有直线边（排除需要曲线的边）
    straight_edges = [edge for edge in G.edges() if edge != (0, 6) and edge != (6, 0)]
    nx.draw_networkx_edges(G, pos, edgelist=straight_edges, edge_color='black')

    # 单独绘制 0-6 的曲线边
    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(0, 6)],
        edge_color='red',
        connectionstyle="arc3,rad=0.9",  # 控制曲线弯曲程度
        style="solid"
    )

    # 绘制节点和标签
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    plt.title("Graph with Curved Edge (0-6)")
    plt.axis('off')
    plt.show()

print("\n右邻居数组 a:", a)

plot_graph_with_curved_edges(adjacency_list, N, P)

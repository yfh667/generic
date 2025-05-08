import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches # Needed for arrowstyle '-' potentially
import math # For checking float equality robustly

# --- Plotting Function (Corrected Layout) ---
def plot_graph_with_auto_curve(adjacency_list, N, P, default_rad=0.3, tolerance=1e-9):
    """
    绘制图形，自动计算节点位置，并将连接同一行且非相邻列节点的边绘制成曲线。
    节点布局：列按 node // N 排列，行内按 node % N 从下到上排列。

    Args:
        adjacency_list (dict): 图的邻接表.
        N (int): 网格的行数 (每列的节点数).
        P (int): 网格的列数.
        default_rad (float): 用于曲线边的默认弯曲度.
        tolerance (float): 用于浮点数比较的容差.
    """
    # --- Graph Creation and Edge Adding (Inside Function) ---
    graph = nx.Graph()
    num_total_nodes = P * N
    graph.add_nodes_from(range(num_total_nodes))
    print(f"初始化图，包含节点 0 到 {num_total_nodes - 1}")

    print("根据邻接表添加边:")
    # ... (Edge adding logic remains the same as your last version) ...
    for node, neighbors in adjacency_list.items():
        if node not in graph:
            # print(f"  警告: 邻接表中的节点 {node} ...") # Optional print
            continue
        for neighbor in neighbors:
            if neighbor not in graph:
                # print(f"  警告: 邻接表中的邻居 {neighbor} ...") # Optional print
                continue
            if not graph.has_edge(node, neighbor):
                # print(f"  添加边 ({node}, {neighbor})") # Optional print
                graph.add_edge(node, neighbor)


    # --- Position Calculation (Inside Function) ---
    num_expected_nodes = P * N
    positions = {}
    print("\n根据 N, P 计算图中节点的位置 (Y轴从下到上):")
    # ... (Position calculation logic remains the same: x = node // N, y = node % N) ...
    for node in graph.nodes():
        if 0 <= node < num_expected_nodes:
            x = node // N # Column index
            y = node % N  # Row index (0 at bottom, N-1 at top)
            positions[node] = (x, y)
        # else: # Handle nodes outside range if necessary

    if not positions:
        print("错误: 无法为任何节点计算位置。无法绘图。")
        return
    print(f"计算得到 {len(positions)} 个节点的位置。")


    # --- Edge Analysis for Curving (Inside Function) ---
    straight_edges = []
    curved_edges = []
    print("\n自动检测需要弯曲的边:")
    # ... (Edge analysis logic remains the same) ...
    if not graph.edges():
        print("图中没有边需要分析或绘制。")
    else:
        for u, v in graph.edges():
            if u in positions and v in positions:
                pos_u = positions[u]
                pos_v = positions[v]
                is_same_row = math.isclose(pos_u[1], pos_v[1], abs_tol=tolerance)
                is_non_adjacent_col = abs(pos_u[0] - pos_v[0]) > (1.0 + tolerance)
                if is_same_row and is_non_adjacent_col:
                    # print(f" - 边 ({u}, {v}): ...曲线。") # Optional print
                    curved_edges.append((u, v))
                else:
                    straight_edges.append((u, v))
            # else: # Handle edges with missing node positions if necessary


    # --- Drawing (Inside Function) ---
    plt.figure(figsize=(max(6, P*1.5), max(4, N*1.5)))
    ax = plt.gca()

    drawable_nodes = list(positions.keys())
    drawable_labels = {node: str(node) for node in drawable_nodes}

    # Draw straight edges
    if straight_edges:
        # print(f"绘制 {len(straight_edges)} 条直线边...") # Optional print
        nx.draw_networkx_edges(graph, positions, ax=ax, edgelist=straight_edges,
                               edge_color='black', alpha=0.6)
    # else: print("没有直线边需要绘制。")

    # Draw curved edges
    if curved_edges:
        connection_style = f"arc3,rad={default_rad}"
        # print(f"绘制 {len(curved_edges)} 条曲线边...") # Optional print
        nx.draw_networkx_edges(
            graph, positions, ax=ax, edgelist=curved_edges, edge_color='red',
            connectionstyle=connection_style, width=1.5, arrows=True, arrowstyle='-'
        )
    # else: print("没有曲线边需要绘制。")

    # Draw nodes and labels
    if drawable_nodes:
        # print(f"绘制 {len(drawable_nodes)} 个节点...") # Optional print
        nx.draw_networkx_nodes(graph, positions, ax=ax, nodelist=drawable_nodes,
                               node_size=500, node_color='skyblue')
        nx.draw_networkx_labels(graph, positions, ax=ax, labels=drawable_labels,
                                font_size=10, font_weight='bold')
    # else: print("没有可绘制的节点。")

    plt.title(f"Graph ({P}x{N} Grid Layout) with Auto-Curved Edges")
    plt.xlabel("Column Index (Node // N)")
    plt.ylabel("Row Index (Node % N)")
    plt.xlim(-0.5, P - 0.5)
    plt.ylim(-0.5, N - 0.5) # Y=0 is bottom, Y=N-1 is top
    ax.set_aspect('equal', adjustable='box')

    # --- REMOVED THIS LINE ---
    # plt.gca().invert_yaxis()

    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xticks(range(P)) # Ticks 0, 1,..., P-1 on X-axis
    plt.yticks(range(N)) # Ticks 0, 1,..., N-1 on Y-axis (bottom to top)
    plt.tight_layout()
    plt.show()



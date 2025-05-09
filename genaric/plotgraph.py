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


import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math


# --- Plotting Function (Corrected Layout & Coloring) ---
def plot_graph_with_auto_curve_distinct(adjacency_list, N, P, distinct=[], default_rad=0.3, tolerance=1e-9):
    """
    绘制图形，自动计算节点位置，并将连接同一行且非相邻列节点的边绘制成曲线。
    根据 distinct 参数为指定的节点区域上色。
    节点布局：列按 node // N 排列，行内按 node % N 从下到上排列。

    Args:
        adjacency_list (dict): 图的邻接表.
        N (int): 网格的行数 (每列的节点数).
        P (int): 网格的列数.
        distinct (list): 一个包含列表的列表，每个内部列表是一组需要特殊着色的节点.
        default_rad (float): 用于曲线边的默认弯曲度.
        tolerance (float): 用于浮点数比较的容差.
    """
    # --- Graph Creation and Edge Adding ---
    graph = nx.Graph()
    num_total_nodes = P * N
    # 只添加在预期范围内的节点，并确保邻接表中的节点也在范围内
    valid_nodes = set(range(num_total_nodes))
    nodes_to_add = set(valid_nodes)  # Start with all potential nodes
    nodes_to_add.update(adjacency_list.keys())  # Add nodes from adjacency list keys
    for neighbors in adjacency_list.values():
        nodes_to_add.update(neighbors)  # Add nodes from adjacency list values

    nodes_to_add = sorted(list(nodes_to_add))  # Sort for consistent ordering
    graph.add_nodes_from(nodes_to_add)
    print(f"初始化图，包含节点: {sorted(list(graph.nodes()))}")

    print("根据邻接表添加边:")
    for node, neighbors in adjacency_list.items():
        if node not in graph:
            # print(f"  警告: 邻接表中的节点 {node} 未添加到图，跳过其边.") # Optional print
            continue
        for neighbor in neighbors:
            if neighbor not in graph:
                # print(f"  警告: 邻接表中的邻居 {neighbor} 未添加到图，跳过边 ({node}, {neighbor}).") # Optional print
                continue
            if not graph.has_edge(node, neighbor):
                # print(f"  添加边 ({node}, {neighbor})") # Optional print
                graph.add_edge(node, neighbor)

    # --- Position Calculation ---
    num_expected_nodes = P * N
    positions = {}
    print("\n根据 N, P 计算图中节点的位置 (Y轴从下到上):")
    for node in graph.nodes():
        # 只为在网格范围内的节点计算位置
        if 0 <= node < num_expected_nodes:
            x = node // N  # Column index (0 to P-1)
            y = node % N  # Row index (0 at bottom, N-1 at top)
            positions[node] = (x, y)
        # else: # Handle nodes outside range if necessary, they won't be plotted
        # print(f"  节点 {node} 超出预期范围 [0, {num_expected_nodes-1})，不计算位置.")

    drawable_nodes = list(positions.keys())  # Nodes that have a calculated position
    if not drawable_nodes:
        print("错误: 无法为任何节点计算位置。无法绘图。")
        return
    print(f"计算得到 {len(drawable_nodes)} 个节点的位置，这些节点将被绘制。")

    drawable_labels = {node: str(node) for node in drawable_nodes}

    # --- Coloring Logic ---
    base_node_color = 'skyblue'  # Default color for nodes not in distinct areas
    # A list of distinct colors to cycle through for the distinct areas
    distinct_colors = [
        'lightcoral', 'lightgreen', 'plum', 'sandybrown',
        'khaki', 'teal', 'orchid', 'salmon'
    ]
    node_color_map = {node: base_node_color for node in drawable_nodes}  # Initialize map with base color

    print("\n根据 distinct 参数为可绘制节点着色:")
    if distinct:
        for i, group in enumerate(distinct):
            if not group:  # Skip empty groups
                continue
            # Cycle through available colors
            group_color = distinct_colors[i % len(distinct_colors)]
            print(f" - 组 {i + 1} (颜色: {group_color}): 尝试着色 {group}")
            colored_count = 0
            for node in group:
                if node in node_color_map:  # Check if the node is among the drawable ones
                    node_color_map[node] = group_color
                    colored_count += 1
                # else:
                # print(f"    警告: 组中的节点 {node} 没有位置信息或不在图中，跳过着色.") # Optional warning for nodes in distinct but not drawable
            print(f"   -> 成功为 {colored_count} 个节点着色.")

        # Create the final list of colors in the same order as drawable_nodes for nx.draw_networkx_nodes
        node_colors_list = [node_color_map[node] for node in drawable_nodes]
        print(f"生成了 {len(node_colors_list)} 个节点的颜色列表用于绘图.")
    else:
        print("没有指定 distinct 区域，所有可绘制节点使用默认颜色。")
        node_colors_list = [base_node_color for node in drawable_nodes]

    # --- Edge Analysis for Curving ---
    straight_edges = []
    curved_edges = []
    print("\n自动检测需要弯曲的边:")
    if not graph.edges():
        print("图中没有边需要分析或绘制。")
    else:
        for u, v in graph.edges():
            # Only analyze edges where both nodes are drawable
            if u in positions and v in positions:
                pos_u = positions[u]
                pos_v = positions[v]
                is_same_row = math.isclose(pos_u[1], pos_v[1], abs_tol=tolerance)
                # Check if column difference is greater than 1 (non-adjacent columns)
                is_non_adjacent_col = abs(pos_u[0] - pos_v[0]) > (1.0 + tolerance)
                if is_same_row and is_non_adjacent_col:
                    # print(f" - 边 ({u}, {v}): 同行非相邻列，曲线。") # Optional print
                    curved_edges.append((u, v))
                else:
                    straight_edges.append((u, v))
            # else:
            # print(f" - 边 ({u}, {v}): 节点位置缺失，不绘制.") # Optional print

    # --- Drawing ---
    plt.figure(figsize=(max(6, P * 1.5), max(4, N * 1.5)))
    ax = plt.gca()

    # Draw straight edges
    if straight_edges:
        print(f"绘制 {len(straight_edges)} 条直线边...")
        nx.draw_networkx_edges(graph, positions, ax=ax, edgelist=straight_edges,
                               edge_color='black', alpha=0.6)
    else:
        print("没有直线边需要绘制。")

    # Draw curved edges
    if curved_edges:
        connection_style = f"arc3,rad={default_rad}"
        print(f"绘制 {len(curved_edges)} 条曲线边...")
        # Note: arrows=True and arrowstyle='-' together draw lines without arrowheads.
        # If you want arrowheads, remove arrowstyle='-'. If you want no arrows, remove arrows=True.
        nx.draw_networkx_edges(
            graph, positions, ax=ax, edgelist=curved_edges, edge_color='red',
            connectionstyle=connection_style, width=1.5, arrows=True, arrowstyle='-'
        )
    else:
        print("没有曲线边需要绘制。")

    # Draw nodes using the calculated colors list
    if drawable_nodes:
        print(f"绘制 {len(drawable_nodes)} 个节点...")
        nx.draw_networkx_nodes(graph, positions, ax=ax, nodelist=drawable_nodes,
                               node_size=500, node_color=node_colors_list)  # Use the list here
        nx.draw_networkx_labels(graph, positions, ax=ax, labels=drawable_labels,
                                font_size=10, font_weight='bold')
    else:
        print("没有可绘制的节点。")

    plt.title(f"Graph ({P}x{N} Grid Layout) with Auto-Curved Edges and Colored Areas")
    plt.xlabel("Column Index (Node // N)")
    plt.ylabel("Row Index (Node % N)")
    # Set limits based on grid dimensions
    plt.xlim(-0.5, P - 0.5)
    plt.ylim(-0.5, N - 0.5)  # Y=0 is bottom, Y=N-1 is top
    ax.set_aspect('equal', adjustable='box')  # Keep aspect ratio equal

    # Do NOT invert y-axis if you want Y=0 at bottom
    # plt.gca().invert_yaxis()

    plt.grid(True, linestyle='--', alpha=0.4)
    # Set ticks to be centered on columns/rows
    plt.xticks(range(P))  # Ticks 0, 1,..., P-1 on X-axis
    plt.yticks(range(N))  # Ticks 0, 1,..., N-1 on Y-axis (bottom to top)

    plt.tight_layout()
    plt.show()



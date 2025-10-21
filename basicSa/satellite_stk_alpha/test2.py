import collections
def build_graph(N):
    """
    Builds the graph representation assuming ALL connections are BI-DIRECTIONAL.

    Args:
        N: The size of the N*N topology.

    Returns:
        A dictionary representing the adjacency list of the graph.
        Keys are node tuples (i, j), values are lists of neighbor tuples.
    """
    if N <= 0:
        return {}

    graph = collections.defaultdict(list)

    # Helper function to add an edge symmetrically
    def add_edge(u, v):
        graph[u].append(v)
        graph[v].append(u)

    for i in range(N):
        for j in range(N):
            current_node = (i, j)

            # 1. Add neighbors within the same orbit (Vertical Edges)
            # Connect (i, j) to (i, (j+1)%N)
            # The connection will be added symmetrically when the loop reaches the neighbor
            neighbor_v_next = (i, (j + 1) % N)
            graph[current_node].append(neighbor_v_next)

            # Note: The (j-1) connection is implicitly handled when the loop processes
            # the node (i, (j-1+N)%N) and connects it to (i,j). Adding it here
            # again would create duplicate list entries, which is okay but less clean.
            # Let's explicitly add both neighbors from current_node's perspective,
            # accepting that the reverse edge will be added again later.
            neighbor_v_prev = (i, (j - 1 + N) % N)

            graph[current_node].append(neighbor_v_prev)

            # 2. Add neighbors in the next orbits (Horizontal Edges)
            # Apply rules and add edges symmetrically
            if i <= N - 3:
                # Potential targets in i+1
                targets_i_plus_1 = [
                    (i + 1, j % N),
                    (i + 1, (j + 1) % N),
                    (i + 1, (j - 1 + N) % N),
                    (i + 1, (j - 2 + N) % N)
                ]
                 # Potential targets in i+2
                targets_i_plus_2 = [
                    (i + 2, j % N),
                    (i + 2, (j - 1 + N) % N),
                    (i + 2, (j - 2 + N) % N)
                ]
                # Add symmetric edges
                for target in targets_i_plus_1:
                    add_edge(current_node, target)
                for target in targets_i_plus_2:
                    add_edge(current_node, target)

            elif i == N - 2:
                 # Potential targets in i+1 (which is N-1)
                targets_n_minus_1 = [
                    (i + 1, j % N),
                    (i + 1, (j + 1) % N),
                    (i + 1, (j - 1 + N) % N),
                    (i + 1, (j - 2 + N) % N)
                ]
                # Add symmetric edges
                for target in targets_n_minus_1:
                     if target[0] < N: # Ensure target orbit is valid
                        add_edge(current_node, target)

            # elif i == N - 1:
                # No NEW horizontal edges starting from the last orbit are defined
                # But edges from N-2 and N-3 pointing TO N-1 have already been
                # added symmetrically.

    # Optional: Remove duplicate entries in adjacency lists if desired
    # for node in graph:
    #    graph[node] = list(set(graph[node])) # Using set removes duplicates

    return graph


def find_path_bfs(graph, start_node, end_node):
    """
    Finds the shortest path between start_node and end_node using BFS.

    Args:
        graph: The adjacency list representation of the graph.
        start_node: The starting node tuple (i, j).
        end_node: The ending node tuple (i, j).

    Returns:
        A list representing the path from start_node to end_node
        (including basicSa and end), or None if no path exists.
    """
    if start_node == end_node:
        return [start_node]
    if start_node not in graph:
         print(f"Warning: Start node {start_node} not found in graph keys.")
         # It might still be reachable if it's an endpoint with no outgoing edges
         # but let's assume valid basicSa nodes are keys for simplicity here.
         # Depending on N=1 case etc., this might need adjustment.
         # If N=1, graph[(0,0)] might point to itself.

    # Check if graph is empty or nodes are invalid early
    if not graph or end_node is None: # Basic checks
        return None

    # Queue for BFS: Stores nodes to visit
    queue = collections.deque([start_node])
    # Set to keep track of visited nodes
    visited = {start_node}
    # Dictionary to store the path predecessor of each node
    # key: node, value: node from which we reached key
    predecessor = {start_node: None}

    while queue:
        current_node = queue.popleft()

        # Check if we reached the end node
        if current_node == end_node:
            # Reconstruct path
            path = []
            node = end_node
            while node is not None:
                path.append(node)
                node = predecessor[node]
            return path[::-1] # Return reversed path (basicSa to end)

        # Explore neighbors
        # Use .get() to handle nodes that might exist but have no outgoing edges listed
        for neighbor in graph.get(current_node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                predecessor[neighbor] = current_node
                queue.append(neighbor)

    # If the queue becomes empty and we haven't found the end_node
    return None # No path found


# --- Main Example Usage ---
if __name__ == "__main__":
    N = 5 # Example size, adjust as needed

    # Define example basicSa and end nodes A, B, C
    # Remember: 0 <= i, j < N
    node_A = (0, 0)
    node_B = (N - 1, N - 1) # Furthest node example
    node_C = (2, 3)         # An intermediate node example

    print(f"Building graph for N={N}...")
    graph = build_graph(N)
    print(f"Graph built. Nodes: {len(graph)}") # Shows how many nodes have outgoing edges

    # --- Find Path A to B ---
    print(f"\nSearching for path from A={node_A} to B={node_B}...")
    path_A_to_B = find_path_bfs(graph, node_A, node_B)

    if path_A_to_B:
        print("Path found (A to B):")
        print(" -> ".join(map(str, path_A_to_B)))
        print(f"Path length (hops): {len(path_A_to_B) - 1}")
    else:
        print("No path found from A to B.")

    # --- Find Path A to C ---
    print(f"\nSearching for path from A={node_A} to C={node_C}...")
    path_A_to_C = find_path_bfs(graph, node_A, node_C)

    if path_A_to_C:
        print("Path found (A to C):")
        print(" -> ".join(map(str, path_A_to_C)))
        print(f"Path length (hops): {len(path_A_to_C) - 1}")
    else:
        print("No path found from A to C.")

    # --- Example: Path from C to B ---
    print(f"\nSearching for path from C={node_C} to B={node_B}...")
    path_C_to_B = find_path_bfs(graph, node_C, node_B)

    if path_C_to_B:
        print("Path found (C to B):")
        print(" -> ".join(map(str, path_C_to_B)))
        print(f"Path length (hops): {len(path_C_to_B) - 1}")
    else:
        print("No path found from C to B.")

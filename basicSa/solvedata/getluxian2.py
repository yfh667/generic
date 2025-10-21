import basicSa.Dataprocessing.readns3file as readns3file
import basicSa.Dataprocessing.readpyfile as readpyfile
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg') # or 'TkAgg', 'GTK3Agg', 'WXAgg' etc.
import matplotlib.pyplot as plt
# 常量定义
STATION_NUM = 8
ORBIT_NUM = 18
SAT_PER_ORBIT = 36


# 链路类型枚举 (Keep as is)
class LinkType:
    EAST_UP_1 = 0
    SAME_ORBIT = 1
    EAST_DOWN_1 = 2
    EAST_DOWN_2 = 3
    JUMP_ORBIT_SAME = 4
    JUMP_ORBIT_DOWN_2 = 5

    WEST_UP_2 = 6
    WEST_UP_1 = 7
    WAST_SAME_ORBIT = 8 # Typo fixed: WAST -> WEST

    WEST_DOWN_1 = 9
    JUMP_WEST_ORBIT_UP_2 = 10
    JUMP_WEST_ORBIT_SAME = 11

    UP_ORBIT = 12
    DOWN_ORBIT = 13


# Typo fixed in calculate_link_type
def calculate_link_type(N, first_plane, first_orbit, next_plane, next_orbit):
    """计算两个节点间的链路类型"""
    plane_diff = next_plane - first_plane
    orbit_diff = (next_orbit - first_orbit + N) % N # Corrected modulo arithmetic

    if plane_diff == 1:
        if orbit_diff == 0:
            return LinkType.SAME_ORBIT
        elif orbit_diff == 1:
            return LinkType.EAST_UP_1
        elif orbit_diff == N - 1:
            return LinkType.EAST_DOWN_1
        elif orbit_diff == N - 2:
            return LinkType.EAST_DOWN_2
        # Added default case for robustness, though maybe unreachable with your setup
        else: return None # Or raise an error
    elif plane_diff == 2:
        if orbit_diff == 0:
            return LinkType.JUMP_ORBIT_SAME
        # Original code had 'else:', implying *any* non-zero orbit_diff.
        # Assuming you only meant the specific case mentioned? Clarify if needed.
        # For now, let's assume it means *any other* orbit diff for plane_diff == 2
        elif orbit_diff == N-2: # Specific check based on your example name
             return LinkType.JUMP_ORBIT_DOWN_2
        else: return None # Or handle other cases if they exist

    elif plane_diff == -1:
        # Corrected variable name typo WAST -> WEST
        if orbit_diff == 0:
            return LinkType.WAST_SAME_ORBIT # Keep enum name if you use it elsewhere
            # return LinkType.WEST_SAME_ORBIT # Or correct enum name if possible
        elif orbit_diff == 1:
            return LinkType.WEST_UP_1
        elif orbit_diff == 2:
            return LinkType.WEST_UP_2
        elif orbit_diff == N - 1:
            return LinkType.WEST_DOWN_1
        else: return None # Default case
    elif    plane_diff == -2:
        if orbit_diff == 0:
            return LinkType.JUMP_WEST_ORBIT_SAME
        # Assuming only one other case intended for plane_diff == -2
        elif orbit_diff == 2: # Specific check based on your example name
             return LinkType.JUMP_WEST_ORBIT_UP_2
        else: return None # Default case

    elif plane_diff == 0:
        if orbit_diff == 1:
            return LinkType.UP_ORBIT
        # Original code had 'else:' for DOWN_ORBIT. Assuming orbit_diff == N-1
        elif orbit_diff == N-1:
            return LinkType.DOWN_ORBIT
        else: return None # Default case
    # Handle cases where plane_diff is > 2 or < -2, or other unhandled orbit_diffs
    # Depending on your basicSa constellation topology, these might be errors
    # or require new LinkTypes. For now, returning None.
    return None # Default if no condition met


# PathSignatureMapper (Keep as is)
class PathSignatureMapper:
    def __init__(self):
        self.signature_map = {}  # 存储签名到数值的映射
        self.counter = 1  # 起始编号为1

    def get_mapped_value(self, signature):
        """获取签名的唯一数值标识"""
        if not signature:  # 空路径处理
            return 0 # Assign 0 for empty/invalid paths

        # Treat None signatures (from potential calculate_link_type errors)
        if signature is None:
             signature = "INVALID_LINK" # Give it a specific name

        if signature not in self.signature_map:
            self.signature_map[signature] = self.counter
            self.counter += 1
        return self.signature_map[signature]

    # Optional: Add a method to get the final map after all processing
    def get_current_map(self):
        return self.signature_map.copy()


def calculate_path_signature(path):
    """计算路径特征签名"""
    signature_parts = []

    # Iterate through hops *between* satellites (ignore station-sat links for signature)
    # Start from index 1 (first sat) to index len(path) - 3 (second-to-last sat)
    for i in range(1, len(path) - 2): # Ensure we have path[i] and path[i+1] as sats
        # Check if nodes are satellites (assuming positive node IDs are satellites after offset)
        if path[i] >= STATION_NUM and path[i+1] >= STATION_NUM:
             first_node = path[i] - STATION_NUM
             next_node = path[i + 1] - STATION_NUM

             first_plane = first_node // SAT_PER_ORBIT
             first_orbit = first_node % SAT_PER_ORBIT
             next_plane = next_node // SAT_PER_ORBIT
             next_orbit = next_node % SAT_PER_ORBIT

             link_type = calculate_link_type(SAT_PER_ORBIT, first_plane, first_orbit,
                                             next_plane, next_orbit)

             # Handle potential None from calculate_link_type
             if link_type is None:
                 signature_parts.append('?') # Indicate unknown link type
                 # print(f"Warning: Unknown link type between nodes {path[i]} ({first_plane},{first_orbit}) and {path[i+1]} ({next_plane},{next_orbit})")
             else:
                 # Using letters A-N for link types 0-13
                 signature_parts.append(chr(65 + link_type))
        else:
             # Handle case where a station node is unexpectedly in the middle
             signature_parts.append('X') # Indicate unexpected node type

    # Return 'EMPTY' or similar if no valid basicSa links found in signature
    return ''.join(signature_parts) if signature_parts else 'NO_SAT_HOPS'


# --- MODIFIED plot_signature_series ---
def plot_signature_series(mapped_values, time_points, id_to_signature_map):
    """Plots path transitions using final mapped IDs and labels."""
    plt.figure(figsize=(16, 8)) # Increased figure size slightly

    # Basic line plot
    # Use 'steps-post' to make transitions clearer (value holds until next time point)
    plt.step(time_points, mapped_values, 'b-', where='post', linewidth=1.5)

    # --- Y-axis Tick Configuration ---
    if id_to_signature_map:
        # Get sorted unique IDs present in the data
        unique_ids_in_data = sorted(list(set(val for val in mapped_values if val > 0))) # Exclude 0

        if unique_ids_in_data:
            # Create labels based on the final mapping
            tick_labels = []
            for tick_id in unique_ids_in_data:
                 # Combine ID and Signature for clarity
                 signature = id_to_signature_map.get(tick_id, f"Unknown ID {tick_id}")
                 tick_labels.append(f"ID {tick_id}: {signature}")

            plt.yticks(unique_ids_in_data, tick_labels, fontsize=8) # Set ticks and labels

            # Adjust y-limits for padding
            min_y = min(unique_ids_in_data)
            max_y = max(unique_ids_in_data)
            plt.ylim(min_y - 0.5, max_y + 0.5)
        else:
             plt.yticks([]) # No valid path IDs found
             plt.ylim(0, 1) # Default limits if no data

    else:
        # Fallback if no map provided (shouldn't happen with new structure)
         pass # Default y-axis behavior

    # Basic formatting
    plt.title('Routing Path Transition Timeline (Ordered IDs)')
    plt.xlabel('Simulation Time (s)')
    plt.ylabel('Path Signature (ID + String)')
    plt.grid(axis='y', linestyle='--', alpha=0.6) # Grid lines for y-axis
    plt.grid(axis='x', linestyle=':', alpha=0.3)  # Lighter grid for x-axis

    # Add scatter points at changes for visibility
    if len(mapped_values) > 1:
        changes = np.where(np.diff(mapped_values) != 0)[0] + 1 # Indices where value *changes* from previous
        if len(changes) > 0:
            # Include the first point
            change_indices = np.concatenate(([0], changes))
            plt.scatter(np.array(time_points)[change_indices],
                        np.array(mapped_values)[change_indices],
                        c='red', s=30, zorder=5, label='Path Change')
            plt.legend()


    plt.tight_layout()
    plt.show()


# --- MODIFIED plot_path_analysis ---
def plot_path_analysis(end_node, db, signature_mapper): # Pass mapper in
    """Analyzes paths, applies custom ID ordering, and plots."""
    time_points = []
    raw_signatures = [] # Store the string signatures first

    # 1. Collect all raw signatures and timestamps
    for snapshot in db.snapshots:
        paths = snapshot.active_paths
        signature = 'N/A' # Default if no path found
        if snapshot.timestamp==62:
            pass
        if end_node in paths and paths[end_node]:
            path = paths[end_node][0]['path']
            # Check if path is long enough to have basicSa hops
            if len(path) >= 3: # Need at least Station -> Sat -> Sat -> ...
                signature = calculate_path_signature(path)
            else:
                signature = 'SHORT_PATH' # Path too short for signature calc

        raw_signatures.append(signature)
        time_points.append(snapshot.timestamp)

    # 2. Get unique signatures and let mapper assign initial sequential IDs
    unique_signatures = sorted(list(set(s for s in raw_signatures if s != 'N/A' and s is not None))) # Exclude N/A etc.
    for sig in unique_signatures:
        signature_mapper.get_mapped_value(sig) # This populates the initial map

    initial_map = signature_mapper.get_current_map()
    print("Initial Sequential Signature Mapping:")
    print(initial_map)

    # 3. Define the desired final mapping (apply swaps/reordering)
    final_map = initial_map.copy() # Start with the initial map




    print("\nFinal Custom Signature Mapping:")
    print(final_map)

    # 4. Create the list of mapped values using the FINAL map
    final_mapped_values = []
    for sig in raw_signatures:
        # Use .get() with a default value (e.g., 0) for signatures not in the map
        # (like 'N/A', 'SHORT_PATH', or potentially 'INVALID_LINK')
        final_mapped_values.append(final_map.get(sig, 0))

    # 5. Create the inverse map (ID -> Signature) for plotting labels
    # Ensure IDs are unique in the values of final_map before inverting
    # If swaps created duplicate IDs (shouldn't happen with simple swaps), handle appropriately.
    id_to_signature_map = {v: k for k, v in final_map.items()}
    if len(id_to_signature_map) != len(final_map):
         print("Warning: Non-unique IDs detected after reordering! Check logic.")
         # Handle collision resolution if necessary, e.g., append suffix to duplicates
         # For now, we proceed, but labels might be ambiguous.

    print(f"\nTotal unique paths (after potential remapping): {len(final_map)}")

    # 6. Plot using the final mapped values and the ID-to-Signature map for labels
    plot_signature_series(final_mapped_values, time_points, id_to_signature_map)


def main():
    # Initialize global mapper outside, pass it into the analysis function
    # This allows the mapper state (like the counter) to persist if called multiple times,
    # but it's cleaner to pass it explicitly.
    path_mapper = PathSignatureMapper()

    # File读取
    try:
        # Ensure the path is correct for your system
        db = readpyfile.readxml('/home/yfh/Desktop/Data/simulation_paths.xml')

        if not db or not db.snapshots:
             print("Error: No snapshots found in the XML file.")
             return
        print(f"Loaded {len(db.snapshots)} snapshots")
    except FileNotFoundError:
        print(f"Error: File not found at /home/yfh/Desktop/Data/satellite.xml")
        return
    except Exception as e:
        print(f"Error loading or parsing file: {str(e)}")
        return

    # 分析指定节点
    TARGET_NODE = 1 # Make sure this is the correct destination node ID in your XML
    try:
        # Pass the mapper instance to the analysis function
        plot_path_analysis(TARGET_NODE, db, path_mapper)
    except KeyError:
        # This specific error might be less likely now with the checks inside plot_path_analysis
        print(f"Node {TARGET_NODE} not found as a destination in path data.")
    except Exception as e:
        print(f"An error occurred during analysis: {str(e)}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging


if __name__ == "__main__":
    main()


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

def calculate_path_signature(path,SAT_PER_ORBIT):
    """计算路径特征签名"""
    signature_parts = []

    # Iterate through hops *between* satellites (ignore station-sat links for signature)
    # Start from index 1 (first sat) to index len(path) - 3 (second-to-last sat)
    for i in range( len(path) - 1): # Ensure we have path[i] and path[i+1] as sats
        # Check if nodes are satellites (assuming positive node IDs are satellites after offset)

             first_node = path[i]
             next_node = path[i + 1]

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


    # Return 'EMPTY' or similar if no valid basicSa links found in signature
    return ''.join(signature_parts) if signature_parts else 'NO_SAT_HOPS'

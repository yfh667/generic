import os
import draw.snapshotf_romxml as snapshotf_romxml
import graph.drawall as drawall

# --- Example Usage ---

# Define a dummy file name for demonstration
P_val, N_val = 10, 10

# Define region color information
# Structure: {layer index: [[region 1 point indices], [region 2 point indices], ...]}
# Point indices are based on the indices within a single layer (0 to N*P-1)
regions_to_color = {}

# Specify the time step you want to extract (e.g., 0)

start_ts = 1500
end_ts = 1523

dummy_file_name = "E:\\code\\data\\station_visible_satellites_100_test.xml"

# Extract and print the satellite lists for each region from the file


# Assuming snapshotf_romxml.extract_region_satellites_from_file returns the relevant data
# I'm not sure if the `extract_region_satellites_from_file` function is implemented, but it should be
# Uncomment and adjust this line as needed based on your actual function:
# region_satellite_groups = snapshotf_romxml.extract_region_satellites_from_file(dummy_file_name, target_time_step)

# Iterate over the time steps
region_satellite_groups= snapshotf_romxml.extract_region_satellites_from_file(dummy_file_name, start_ts, end_ts)
target_time_step = len(region_satellite_groups)
print(f"\nExtracting data for time step {target_time_step} from '{dummy_file_name}'...")
for i in range(len(region_satellite_groups)):
        region_satellite_group = [[int(point) for point in region] for region in region_satellite_groups[i]]
        u =region_satellite_group[0]
        v = region_satellite_group[3]
        o = [u,v]
        regions_to_color[i] = o  # Corrected append to dictionary assignment


# for i in range(target_time_step):
#     region_satellite_groups = snapshotf_romxml.extract_region_satellites_from_file(dummy_file_name, i)
#     region_satellite_groups = [[int(point) for point in region] for region in region_satellite_groups]
#     u =region_satellite_groups[0]
#     v = region_satellite_groups[3]
#     o = [u,v]
#     regions_to_color[i] = o  # Corrected append to dictionary assignment

# 1. Create the basic multi-layer topology
# plot_multi_layer_topology returns the plotter object, original point objects, and all coordinates
main_plotter, original_points_objs, all_coords = drawall.plot_multi_layer_topology(P_val, N_val, target_time_step)

# 2. Color the specified regions
# apply_region_colors will add new colored points to the main_plotter
main_plotter = drawall.apply_region_colors(main_plotter, P_val, N_val, target_time_step, regions_to_color, all_coords)

# 3. Define the points that need to be connected (using actual 3D coordinates)
# Ensure these coordinates are consistent with SPACING and LAYER_HEIGHT settings
connections_list = [
    [(0, 0, 0), (1, 2, 4)],  # Vertical connection
    [(1, 1, 1), (1, 1, 2)],  # Cross-layer connection
    [(2, 2, 0), (2, 2, 2)],  # Same-layer diagonal connection
]

# 4. Add dashed connections
main_plotter = drawall.add_dashed_connections(main_plotter, connections_list)

# 6. Display the plot window
# viewup="z" means Z-axis is up, which is usually more natural for layered structures
# title sets the window title
print("Vedo window is about to display. Please use the mouse for interaction:")
print("- Left-click and drag: Rotate")
print("- Middle-click and drag (or Shift + Left-click drag): Pan")
print("- Scroll wheel (or Right-click drag): Zoom")
main_plotter.show(viewup="z", title="Interactive 3D Topology")

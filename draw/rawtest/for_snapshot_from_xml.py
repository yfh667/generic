import os
import draw.snapshotf_romxml as snapshotf_romxml
# --- Example Usage ---

# Define a dummy file name for demonstration


# Write the XML data to the dummy file
try:

    # Specify the time step you want to extract (e.g., 0)
    target_time_step = 0
    dummy_file_name = "E:\code\data\station_visible_satellites.xml"
    # Extract and print the satellite lists for each region from the file
    print(f"\nExtracting data for time step {target_time_step} from '{dummy_file_name}'...")
    region_satellite_groups = snapshotf_romxml.extract_region_satellites_from_file(dummy_file_name, target_time_step)

    print(f"Satellite groups for time step {target_time_step}:")
    for i, satellite_list in enumerate(region_satellite_groups):
        print(f"Region {i}: {satellite_list}")



except Exception as e:
    print(f"An error occurred during example execution: {e}")

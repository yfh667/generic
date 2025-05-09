import xml.etree.ElementTree as ET
import os # Needed for file operations in the example

def extract_region_satellites_from_file(xml_file_path, target_ts):
    """
    从 XML 文件中读取快照数据，提取指定时间步内各区域站点可见的唯一卫星列表。

    Args:
        xml_file_path (str): 包含快照数据的 XML 文件路径。
        target_ts (int): 要提取数据的时间步 (step)。

    Returns:
        list: 一个包含列表的列表，每个内部列表是对应区域的唯一卫星ID (float类型)，
              按区域索引 0 到 5 排列。如果指定时间步不存在或文件读取/解析失败，
              返回 [set(), set(), ..., set()] 转换成的列表，即所有区域的卫星列表都为空。
    """
    # Define the regions based on station IDs
    # Map region index to a set of station IDs
    regions_map = {
        0: set(range(9)),     # Region 0: stations 0-8
        1: {9},               # Region 1: station 10
        2: {10},               # Region 2: station 11
        3: set(range(11, 15)), # Region 3: stations 12-15
        4: set(range(15, 17)), # Region 4: stations 16-17
        5: set(range(17, 19))  # Region 5: stations 18-20
    }

    # Initialize a list of sets to store unique satellite IDs for each region
    # Using sets automatically handles uniqueness
    region_satellites = [set() for _ in range(len(regions_map))]

    try:
        # Parse the XML file directly
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # Find the specific time step
        time_element = None
        for time_snap in root.findall('time'):
            if time_snap.get('step') == str(target_ts):
                time_element = time_snap
                break

        # If the target time step is found
        if time_element is not None:
            stations_element = time_element.find('stations')
            if stations_element is not None:
                # Iterate through each station in the time step
                for station_element in stations_element.findall('station'):
                    station_id_str = station_element.get('id')
                    if station_id_str is None:
                        continue # Skip stations without an ID

                    try:
                        station_id = int(station_id_str)
                    except ValueError:
                        print(f"Warning: Skipping station with non-integer id: {station_id_str}")
                        continue # Skip if id is not a valid integer

                    # Check which region the station belongs to
                    for region_idx, station_ids_in_region in regions_map.items():
                        if station_id in station_ids_in_region:
                            # If the station is in this region, collect its satellites
                            for satellite_element in station_element.findall('satellite'):
                                satellite_id_str = satellite_element.get('id')
                                if satellite_id_str is not None:
                                    try:
                                        # Convert satellite ID to float and add to the region's set
                                        region_satellites[region_idx].add(float(satellite_id_str))
                                    except ValueError:
                                        print(f"Warning: Skipping satellite with non-float id: {satellite_id_str} in station {station_id}")
                                        continue # Skip if satellite id is not a valid float
                            # Once a station is found in a region, no need to check other regions
                            break
        else:
            print(f"Warning: Time step '{target_ts}' not found in the XML data.")

    except FileNotFoundError:
        print(f"Error: XML file not found at path '{xml_file_path}'")
        # Return empty lists if file not found
        region_satellites = [set() for _ in range(len(regions_map))]
    except ET.ParseError as e:
        print(f"Error parsing XML file '{xml_file_path}': {e}")
        # Return empty lists if parsing fails
        region_satellites = [set() for _ in range(len(regions_map))]
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
         # Return empty lists for other errors
        region_satellites = [set() for _ in range(len(regions_map))]


    # Convert the sets of satellites to sorted lists
    result_lists = []
    for sat_set in region_satellites:
        # Sort the satellite IDs numerically before converting back to list
        sorted_sats = sorted(list(sat_set))
        result_lists.append(sorted_sats)

    return result_lists
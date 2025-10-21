# -*- coding: utf-8 -*-
# Note: Matplotlib imports are still here for the XML parsing part,
# but the plotting will use Plotly.
import matplotlib
try:
    matplotlib.use('TkAgg') # Or Qt5Agg, etc.
except ImportError:
    try:
        matplotlib.use('Qt5Agg')
    except ImportError:
        matplotlib.use('Agg')

# matplotlib.pyplot and matplotlib.widgets.Slider are not needed for Plotly
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider
# matplotlib.colors, matplotlib.ticker are generally not needed for Plotly's
# default color handling or custom scales.
# from matplotlib.colors import ListedColormap, BoundaryNorm
# from matplotlib.ticker import MaxNLocator
# mpl_toolkits.mplot3d, matplotlib.cm, matplotlib.cm.ScalarMappable are not needed for Plotly
# from matplotlib.cm import ScalarMappable

import numpy as np
import xml.etree.ElementTree as ET
import sys

# --- Import Plotly ---
import plotly.graph_objects as go
import plotly.express as px
# import plotly.colors

# --- Parameters ---
N = 36
P = 18

# --- 1. Parse XML Data (Keep the same function with robustness checks and prints) ---
def parse_xml(xml_file, N, P):
    """Parses the XML file to extract connection data per time step."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"Error: XML file not found at {xml_file}")
        sys.exit(1)
    except ET.ParseError:
        print(f"Error: Could not parse XML file {xml_file}. Check its format.")
        sys.exit(1)

    data = {}
    all_steps_elem = root.findall('time')
    if not all_steps_elem:
        print("Error: No 'time' elements found in the XML.")
        sys.exit(1)

    min_step, max_step = None, None # Initialize min/max steps
    all_steps_int = []

    try:
        all_steps_int = sorted([int(time_elem.get('step')) for time_elem in all_steps_elem if time_elem.get('step')])
        if not all_steps_int:
             print("Warning: 'step' attribute missing or invalid in all 'time' elements.")
             # If no valid steps are found, min/max remain None, range loop won't run
        else:
            min_step, max_step = all_steps_int[0], all_steps_int[-1]
             # Pre-initialize data dictionary for the full range of steps if steps were found
            for step in range(min_step, max_step + 1):
                data[step] = np.zeros((N, P), dtype=int)

    except ValueError:
         print("Error: Could not convert 'step' attribute to integer during initial scan.")
         # min/max remain None, range loop won't run based on initial scan

    # --- DEBUG PRINT ---
    print(f"Initial scan found steps: {all_steps_int}")
    print(f"Initial min_step: {min_step}, max_step: {max_step}")


    max_connections_overall = 0

    for time_elem in all_steps_elem:
        step_str = time_elem.get('step')
        if not step_str:
             print("Warning: 'step' attribute missing in a 'time' element. Skipping.")
             continue
        try:
            step = int(step_str)
        except ValueError:
             print(f"Warning: Invalid integer value for 'step': {step_str}. Skipping time element.")
             continue

        # If step is outside the initially found min/max range or if min/max were None, add it
        if step not in data:
             print(f"Warning: Step {step} found in XML but not in initial range. Adding step data.")
             data[step] = np.zeros((N, P), dtype=int)
             # Update min/max steps based on newly found step if necessary
             min_step = min(min_step, step) if min_step is not None else step
             max_step = max(max_step, step) if max_step is not None else step


        for sat in time_elem.findall('sat'):
            try:
                sat_id = int(sat.get('satid'))
                connections_str = sat.get('connections')
                if connections_str is None:
                    # print(f"Warning: 'connections' attribute missing for satid {sat_id} in step {step}. Assuming 0 connections.")
                    connections = 0
                else:
                    connections = int(connections_str)


                row = sat_id % N
                col = sat_id // N

                if 0 <= row < N and 0 <= col < P:
                    data[step][row, col] = connections
                    if connections > max_connections_overall:
                        max_connections_overall = connections
                else:
                    print(f"Warning: Satellite ID {sat_id} at step {step} out of bounds ({N}x{P}). Row={row}, Col={col}. Skipping.")
            except (ValueError, TypeError) as e:
                print(f"Warning: Invalid satid or connections data for basicSa in step {step}: {sat.attrib}. Error: {e}. Skipping basicSa.")
            except Exception as e:
                 print(f"Warning: An unexpected error occurred processing basicSa data in step {step}: {sat.attrib}. Error: {e}. Skipping basicSa.")

    # --- DEBUG PRINT ---
    print(f"Data dictionary contains {len(data)} steps after parsing.")


    sorted_steps = sorted(data.keys()) if data else []
    sorted_data = {step: data[step] for step in sorted_steps}

    final_min_step = sorted_steps[0] if sorted_steps else 0
    final_max_step = sorted_steps[-1] if sorted_steps else 0

    # --- DEBUG PRINT ---
    print(f"Final data range: min_step={final_min_step}, max_step={final_max_step}")
    print(f"Maximum connections found during parsing: {max_connections_overall}")


    return sorted_data, max_connections_overall, final_min_step, final_max_step


# --- 2. No Discrete Colormap Module needed for Plotly this way ---


# --- 3. Create Plotly 3D Time Evolution Plot (Modified for Red 0 and include all points) ---
def plot_plotly_3d_time_evolution(data, N, P, max_connections, min_step, max_step):
    """
    Creates an interactive Plotly 3D scatter plot showing connection evolution over time.
    X-axis: Orbit Plane (P)
    Y-axis: Satellite Index (N)
    Z-axis: Time Step
    Color: Number of Connections (0 is red, >0 is blue scale)
    """
    print("Preparing data for Plotly 3D plot...")
    x_coords = [] # Orbit Plane (Column Index)
    y_coords = [] # Satellite Index (Row Index)
    z_coords = [] # Time Step
    conn_values = [] # Connection count (for color)
    hover_text = [] # Text to show on hover

    # Iterate through time steps and grid points
    for step, grid_data in data.items():
        # grid_data is an (N, P) array
        for (row, col), connections in np.ndenumerate(grid_data):
             # --- MODIFICATION: Include ALL points, even with connections == 0 ---
             # Removed the 'if connections > 0:' check
             x_coords.append(col) # P is horizontal axis (column)
             y_coords.append(row) # N is vertical axis (row)
             z_coords.append(step) # Time is Z axis
             conn_values.append(connections) # Connection count for color
             hover_text.append(f'Step: {step}<br>Sat N: {row}<br>Sat P: {col}<br>Connections: {connections}')

    # --- DEBUG PRINT ---
    print(f"Prepared {len(x_coords)} data points for plotting.")

    if not x_coords:
        print("No data points found to plot in 3D.")
        return

    # --- MODIFICATION: Define a custom colorscale with Red for 0 ---
    # Plotly colorscales are lists of [normalized_value, color] pairs, where value is between 0 and 1.
    # We need to map our connection values (0 to max_connections) to the 0-1 range.
    # A connection count of 'v' is normalized to v / max_connections (when cmin=0, cmax=max_connections).

    # Map 0 connections (normalized value 0.0) to Red.
    # Map > 0 connections (normalized values > 0.0) to a sequential scale (e.g., Blues).

    custom_colorscale = [[0.0, 'red']] # Start the colorscale with Red for 0.0

    if max_connections > 0:
        # Use a sequential colormap from Plotly for values > 0
        blues_scale = px.colors.sequential.Blues
        # num_blues_colors = len(blues_scale) # Not directly used when sampling

        # Add points to the colorscale for connections 1 to max_connections.
        # These values will map to the Blues scale.
        # Ensure the normalized values for > 0 connections are slightly above 0.0
        # to prevent interpolation issues with the 0.0 point.
        epsilon = 1e-9

        for i in range(1, max_connections + 1):
            # Normalized value for connection count 'i' (i > 0)
            # Map the range [1, max_connections] to the range [epsilon, 1.0] for colorscale values.
            if max_connections > 1:
                 norm_val_for_blues = epsilon + (i - 1) / (max_connections - 1) * (1.0 - epsilon)
            else: # max_connections == 1
                 norm_val_for_blues = 0.5

            color_from_blues = px.colors.sample_colorscale(blues_scale, norm_val_for_blues)[0]

            custom_colorscale.append([norm_val_for_blues, color_from_blues])

        custom_colorscale.sort(key=lambda x: x[0])

    # Ensure the colorscale has at least the red point if max_connections is 0
    if max_connections == 0 and not custom_colorscale:
         custom_colorscale = [[0.0, 'red']]

    # --- Create Plotly 3D Scatter Plot ---
    fig = go.Figure(data=[go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='markers',
        marker=dict(
            size=5, # Adjust marker size as needed
            color=conn_values, # Color points based on connection count
            colorscale=custom_colorscale, # Use the custom scale
            colorbar=dict(title='Connections (0: Red)'),
            cmin=0, # Ensure cmin is 0 so 0 connections map to 0.0 normalized value
            cmax=max_connections if max_connections > 0 else 1, # Ensure cmax is at least 1 if max_conn is 0
            opacity=0.8 # Adjust opacity
        ),
        text=hover_text, # Text to show on hover
        hoverinfo='text'
    )])

    # --- Configure Layout ---
    fig.update_layout(
        title='3D Satellite Connection Evolution (0 Connections: Red)',
        scene=dict(
            xaxis_title='Orbit Plane (P)',
            yaxis_title='Satellite Index (N)',
            zaxis_title='Time Step',
            xaxis=dict(
                tickvals=np.arange(P),
            ),
            yaxis=dict(
                tickvals=np.arange(N),
            ),
             zaxis=dict(
                 tickvals=np.arange(min_step, max_step + 1, step=max(1, (max_step - min_step) // 10)),
             )
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    print("Displaying Plotly 3D plot (opens in browser or inline)...")
    fig.show()

    print("Plotting complete.")


# --- Main Execution ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        xml_file_path = sys.argv[1]
    else:
        xml_file_path = "/home/yfh/Desktop/Data/sat_access_stats.xml" # MODIFY THIS PATH

    print(f"Loading data from: {xml_file_path}")
    print(f"Grid dimensions: N={N} (rows/vertical), P={P} (columns/horizontal)")

    # 1. Parse the XML data
    parsed_data, max_conn, min_t, max_t = parse_xml(xml_file_path, N, P)

    if not parsed_data:
         print("No data parsed. Exiting.")
         # The parse_xml now exits if critical errors occur early, but this check remains
         sys.exit(1)

    print(f"Maximum connections found: {max_conn}")
    print(f"Data covers time steps {min_t} to {max_t}.")
    print(f"Number of steps in parsed data dictionary: {len(parsed_data)}")


    # 2. Create and display the Plotly 3D plot
    plot_plotly_3d_time_evolution(parsed_data, N, P, max_conn, min_t, max_t)

    print("Plotting complete.")

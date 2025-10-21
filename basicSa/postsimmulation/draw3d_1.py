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
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.cm import ScalarMappable

import numpy as np
import xml.etree.ElementTree as ET
import sys

# --- Import Plotly ---
import plotly.graph_objects as go
import plotly.express as px # Often useful, but go.Scatter3d is explicit here
# import plotly.colors # Potentially useful for custom colorscales

# --- Parameters ---
N = 36
P = 18

# --- 1. Parse XML Data (Keep the same function) ---
# [Keep the exact same parse_xml function here]
def parse_xml(xml_file, N, P):
    """Parses the XML file to extract connection data per time step."""
    # ... (code from previous answer - copy the function definition exactly) ...
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

    # Convert steps to integers and find min/max
    try:
        all_steps_int = sorted([int(time_elem.get('step')) for time_elem in all_steps_elem if time_elem.get('step')])
        if not all_steps_int:
             print("Error: 'step' attribute missing or invalid in all 'time' elements.")
             sys.exit(1)
        min_step, max_step = min(all_steps_int), max(all_steps_int)
    except ValueError:
         print("Error: Could not convert 'step' attribute to integer.")
         sys.exit(1)


    max_connections_overall = 0

    # Pre-initialize data dictionary for the full range of steps
    for step in range(min_step, max_step + 1):
        data[step] = np.zeros((N, P), dtype=int)

    # Populate data from XML
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

        if step not in data:
             print(f"Warning: Step {step} found in XML but not in initial range ({min_step}-{max_step}). Adding.")
             # This case should be rare if min/max calculation is correct, but defensive.
             data[step] = np.zeros((N, P), dtype=int)


        for sat in time_elem.findall('sat'):
            try:
                sat_id = int(sat.get('satid'))
                connections = int(sat.get('connections'))
                row = sat_id % N
                col = sat_id // N

                if 0 <= row < N and 0 <= col < P:
                    data[step][row, col] = connections
                    if connections > max_connections_overall:
                        max_connections_overall = connections
                else:
                    print(f"Warning: Satellite ID {sat_id} at step {step} out of bounds ({N}x{P}). Row={row}, Col={col}. Skipping.")
            except (ValueError, TypeError) as e:
                print(f"Warning: Invalid data for basicSa in step {step}: {sat.attrib}. Error: {e}. Skipping basicSa.")
            except Exception as e: # Catch any other unexpected errors per basicSa
                 print(f"Warning: An unexpected error occurred processing basicSa data in step {step}: {sat.attrib}. Error: {e}. Skipping basicSa.")


    # Ensure missing steps between min/max found have zero data
    # This is handled by the pre-initialization loop, but let's double check.
    # However, the sorted() call below ensures we only include steps present in data.
    # Let's rely on the pre-initialization for correctness.

    # Sort the data by step
    sorted_steps = sorted(data.keys())
    sorted_data = {step: data[step] for step in sorted_steps}


    return sorted_data, max_connections_overall, min_step, max_step # Return min_step, max_step from the parsed data


# --- 2. No Discrete Colormap Module needed for Plotly this way ---
# Plotly handles color mapping directly using the 'color' attribute and 'colorscale'.
# We will define the colorscale within the plotting function.


# --- 3. Create Plotly 3D Time Evolution Plot ---
def plot_plotly_3d_time_evolution(data, N, P, max_connections, min_step, max_step):
    """
    Creates an interactive Plotly 3D scatter plot showing connection evolution over time.
    X-axis: Orbit Plane (P)
    Y-axis: Satellite Index (N)
    Z-axis: Time Step
    Color: Number of Connections
    """
    print("Preparing data for Plotly 3D plot...")
    # Prepare data points for scatter plot
    x_coords = [] # Orbit Plane (Column Index)
    y_coords = [] # Satellite Index (Row Index)
    z_coords = [] # Time Step
    conn_values = [] # Connection count (for color)
    hover_text = [] # Text to show on hover

    # Iterate through time steps and grid points
    for step, grid_data in data.items():
        # grid_data is an (N, P) array
        for (row, col), connections in np.ndenumerate(grid_data):
             # OPTIONAL: Plot only points with connections > 0 to reduce data transfer
             # and rendering points that might not be interesting.
             # If you need to show where connections are 0, remove this if condition.
             if connections > 0:
                x_coords.append(col) # P is horizontal axis (column)
                y_coords.append(row) # N is vertical axis (row)
                z_coords.append(step) # Time is Z axis
                conn_values.append(connections) # Connection count for color
                hover_text.append(f'Step: {step}<br>Sat N: {row}<br>Sat P: {col}<br>Connections: {connections}')


    if not x_coords:
        print("No connection data (with > 0 connections) found to plot in 3D.")
        # If you removed the 'if connections > 0:' check, this message might need adjustment
        # or you should still check if conn_values has data if you want to plot zeros.
        return


    # Define a custom colorscale: White for 0, then a sequential scale for > 0
    # Plotly colorscales are lists of [value, color] pairs, where value is between 0 and 1.
    # We need to map our connection values (0 to max_connections) to the 0-1 range.
    # 0 connections maps to value 0.
    # 1 connection maps to value 1/max_connections (if max_connections > 0).
    # max_connections maps to value 1.

    # Use a standard sequential colormap from Plotly for values > 0
    # px.colors.sequential.Blues[::-1] is the 'Blues' scale reversed (dark to light blue)
    # We want light blue for low connections and dark blue for high. Let's use the standard 'Blues'.
    # The number of colors needed is max_connections + 1.
    # We will pick colors from a continuous colormap range.

    colors = px.colors.sequential.Blues # A list of hex color strings
    num_sequential_colors = len(colors)

    custom_colorscale = [[0.0, 'rgb(255, 255, 255)']] # Start with white for 0

    if max_connections > 0:
        # Create steps for the sequential part of the colorscale
        # Map connection counts 1 to max_connections to the range (epsilon, 1.0]
        # We use a small epsilon > 0 to ensure 0.0 is exclusively white.
        epsilon = 1e-9
        for i in range(1, max_connections + 1):
            # Normalized value between 0 and 1 (exclusive of 0)
            normalized_value = i / max_connections
            # Ensure it's slightly above 0 if max_connections is 1 and i is 1
            normalized_value = max(normalized_value, epsilon)

            # Map this normalized value to a position within the sequential colorscale (0 to 1)
            # Use the normalized_value directly to sample the sequential colormap range
            color_index = normalized_value * (num_sequential_colors - 1)
            # Get the color by interpolating or picking from the list (Plotly handles this mapping internally)
            # We just need to provide the [normalized_value, color] pairs.
            # Plotly's colorscale mapping is based on linear interpolation between provided points.
            # We need points for each integer value from 1 to max_connections.

            # To get distinct colors for each integer > 0, define steps at each integer boundary
            # Normalized value for connection count 'i' (where i > 0)
            # Map 1 -> small value > 0, 2 -> slightly larger, ..., max_conn -> 1.0
            # Let's map i to a value in [epsilon, 1.0]
            normalized_conn_value = (i - 1) / (max_connections - 1) if max_connections > 1 else 0.5 # Map 1 to 0.5 if max_conn is 1
            normalized_conn_value = epsilon + normalized_conn_value * (1.0 - epsilon)


            # Select a color from the Blues scale corresponding to this normalized value
            # This requires mapping the normalized_conn_value (0 to 1) to the blues list indices (0 to num_sequential_colors - 1)
            # Plotly's colorscale definition is more direct: you give it the [normalized_value, color] pairs.
            # Let's create pairs for each integer value 1 to max_connections.
            # Value for connection 'i' will be i / max_connections.
            # We need to ensure the color for 'i' is distinct.

            # A simpler approach for discrete colors > 0:
            # Map connection 1 to value (1/max_conn), connection 2 to (2/max_conn), ..., max_conn to (max_conn/max_conn=1)
            # Use Plotly's built-in sequential scale which will interpolate.
            # Define the stops at the boundaries of the integer values.

            # Let's try defining stops at the *midpoints* for better discrete mapping with a continuous scale
            # Stop for connections 'i' is at normalized value (i - 0.5) / max_connections ? No, this is complex.

            # Plotly colorscales work by interpolating between the provided value/color pairs.
            # To make integer values map deterministically, define points at each integer value.
            # Normalized value for connection 'i' (i > 0) is i / max_connections.
            # The color should correspond to this value on the sequential scale.
            # We need points like [1/max_conn, color_at_1/max_conn], [2/max_conn, color_at_2/max_conn], ...
            # Plotly's px.colors.sequential.Blues is a list of hex colors.
            # We can select colors from this list based on the normalized value.

            # A simple custom scale for discrete integer values 0 to max_connections:
            # Value 0 maps to white.
            # Values 1 to max_connections are spread linearly across the rest of the 0-1 range (e.g., 0.1 to 1.0)
            # and mapped to colors from a sequential colormap.
            if max_connections > 1:
                # Map connection i (from 1 to max_conn) to a normalized value from 0.1 to 1.0
                norm_val = 0.1 + (i - 1) / (max_connections - 1) * 0.9
            else: # max_connections == 1
                 norm_val = 0.5 # Map 1 connection to the middle of the scale > 0

            # Get color from Blues scale based on norm_val
            # Use px.colors.sample_colorscale
            color_for_i = px.colors.sample_colorscale(px.colors.sequential.Blues, norm_val)[0] # sample_colorscale returns a list

            custom_colorscale.append([norm_val, color_for_i])

        # Sort the colorscale by value
        custom_colorscale.sort(key=lambda x: x[0])

    # Ensure there's at least one point if max_connections was 0, otherwise the scale might be empty
    if max_connections == 0 and not custom_colorscale:
         custom_colorscale = [[0.0, 'rgb(255, 255, 255)']]


    # --- Create Plotly 3D Scatter Plot ---
    fig = go.Figure(data=[go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='markers',
        marker=dict(
            size=5, # Adjust marker size as needed
            color=conn_values, # Color points based on connection count
            colorscale=custom_colorscale, #'Blues', # Use the custom scale
            colorbar=dict(title='Connections'),
            cmin=0,
            cmax=max_connections,
            opacity=0.8 # Adjust opacity
        ),
        text=hover_text, # Text to show on hover
        hoverinfo='text'
    )])

    # --- Configure Layout ---
    fig.update_layout(
        title='3D Satellite Connection Evolution',
        scene=dict(
            xaxis_title='Orbit Plane (P)',
            yaxis_title='Satellite Index (N)',
            zaxis_title='Time Step',
            xaxis=dict(
                tickvals=np.arange(P), # Explicit ticks for P
                # ticktext=[str(i) for i in range(P)] # Optional: labels
            ),
            yaxis=dict(
                tickvals=np.arange(N), # Explicit ticks for N
                # ticktext=[str(i) for i in range(N)] # Optional: labels
            ),
             zaxis=dict(
                # Plotly often handles Z ticks well, but you can set them explicitly if needed
                # tickvals=np.arange(min_step, max_step + 1, step=max(1, (max_step - min_step) // 10)), # Example: show max 10 ticks
                # ticktext=[str(i) for i in np.arange(min_step, max_step + 1, step=max(1, (max_step - min_step) // 10))]
             )
        ),
        margin=dict(l=0, r=0, b=0, t=40) # Adjust margins
    )

    print("Displaying Plotly 3D plot (opens in browser or inline)...")
    # Display the plot
    # If in a Jupyter environment, it will display inline.
    # Otherwise, it typically opens in your default web browser.
    fig.show()

    print("Plotly plot displayed. Interaction happens in the plot window/browser tab.")


# --- Main Execution ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        xml_file_path = sys.argv[1]
    else:
        xml_file_path = "/home/yfh/Desktop/Data/sat_access_stats.xml" # MODIFY THIS PATH

    print(f"Loading data from: {xml_file_path}")
    print(f"Grid dimensions: N={N} (rows/vertical), P={P} (columns/horizontal)")

    # 1. Parse the XML data
    # min_t and max_t are the actual min/max steps found in the data, not the requested range if data is sparse.
    parsed_data, max_conn, min_t, max_t = parse_xml(xml_file_path, N, P)

    if not parsed_data:
         print("No data parsed. Exiting.")
         sys.exit(1)

    print(f"Maximum connections found: {max_conn}")
    print(f"Data covers time steps {min_t} to {max_t}.")

    # 2. Create and display the Plotly 3D plot
    plot_plotly_3d_time_evolution(parsed_data, N, P, max_conn, min_t, max_t)

    print("Plotting complete.")

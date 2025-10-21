# -*- coding: utf-8 -*-
import matplotlib
try:
    matplotlib.use('TkAgg')
except ImportError:
    print("TkAgg backend not available, trying Qt5Agg...")
    try:
        matplotlib.use('Qt5Agg')
    except ImportError:
        print("Qt5Agg backend also not available, using default Agg (non-interactive).")
        matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import xml.etree.ElementTree as ET
# Import color and ticker functionalities
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MaxNLocator # For colorbar ticks
import sys

# --- Parameters ---
N = 36
P = 18

# --- 1. Parse XML Data (Same as before) ---
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
    all_steps = sorted([int(time_elem.get('step')) for time_elem in root.findall('time')])
    if not all_steps:
        print("Error: No 'time' elements found in the XML.")
        sys.exit(1)

    min_step, max_step = min(all_steps), max(all_steps)
    max_connections_overall = 0

    # Pre-initialize data dictionary
    for step in range(min_step, max_step + 1):
        data[step] = np.zeros((N, P), dtype=int)

    for time_elem in root.findall('time'):
        step = int(time_elem.get('step'))
        if step not in data:
             print(f"Warning: Step {step} found in XML but not in initial range. Adding.")
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

    steps_present = set(data.keys())
    for step in range(min_step, max_step + 1):
        if step not in steps_present:
            data[step] = np.zeros((N, P), dtype=int) # Ensure missing steps have zero data

    sorted_data = dict(sorted(data.items()))
    return sorted_data, max_connections_overall, min_step, max_step


# --- 2. Create Discrete Colormap Module/Function ---
def create_discrete_colormap(max_connections):
    """
    Creates a discrete colormap and normalization for integer connection values.
    Ensures 0 maps to white, and other integers map to distinct colors from a sequential map.

    Args:
        max_connections (int): The maximum connection count found in the data.

    Returns:
        tuple: (matplotlib.colors.ListedColormap, matplotlib.colors.BoundaryNorm)
               The colormap and normalization objects to use.
    """
    # Handle edge case where there are no connections
    if max_connections <= 0:
        # Only need white color for value 0
        cmap = ListedColormap(['white'])
        # Boundaries encompass only 0: [-0.5, 0.5]
        boundaries = [-0.5, 0.5]
        norm = BoundaryNorm(boundaries, cmap.N)
        return cmap, norm

    # Choose a base sequential colormap (e.g., 'Blues', 'YlGnBu', 'viridis', 'plasma', 'Greens')
    # 'Blues' goes from light blue to dark blue
    # 'YlGnBu' goes from light yellow through green to blue
    base_cmap_name = 'Blues'
    base_cmap = plt.cm.get_cmap(base_cmap_name)

    # We need N+1 colors for values 0, 1, ..., N (where N=max_connections)
    num_colors = max_connections + 1

    # Generate colors from the base map.
    # We sample slightly away from the absolute basicSa (often too light/white)
    # unless we only need 1 color beyond white.
    if num_colors > 1 :
       # Sample from a range that avoids the very lightest color if desired
       # linspace(0.1, 1.0, num_colors -1) samples non-white colors
       # We then prepend white
       color_list = base_cmap(np.linspace(0.1, 1.0, num_colors - 1))
       # Prepend white for the 0 value
       colors = np.vstack(([1, 1, 1, 1], color_list)) # Use RGBA white
    else: # Only need color for 0
         colors = np.array([[1,1,1,1]]) # White

    # Create a ListedColormap
    cmap = ListedColormap(colors, name=f'discrete_{base_cmap_name}')

    # Define boundaries for the normalization.
    # For values 0, 1, ..., N, boundaries are -0.5, 0.5, 1.5, ..., N+0.5
    boundaries = np.arange(-0.5, max_connections + 1.5, 1)
    norm = BoundaryNorm(boundaries, cmap.N) # cmap.N is the number of colors

    return cmap, norm


# --- 3. Create Visualization ---
def plot_orbital_map(data, N, P, max_connections, min_step, max_step):
    """Creates the interactive 2D scatter plot with a time slider."""

    # --- Setup Coordinates ---
    orbit_indices = np.arange(P)
    sat_indices = np.arange(N)
    xx, yy = np.meshgrid(orbit_indices, sat_indices)
    x_coords_flat = xx.flatten()
    y_coords_flat = yy.flatten()

    # --- Get Discrete Colormap and Normalization ---
    cmap, norm = create_discrete_colormap(max_connections)

    # --- Create Plot ---
    fig, ax = plt.subplots(figsize=(14, 9))
    plt.subplots_adjust(bottom=0.25)

    initial_step = min_step
    initial_connections = data[initial_step].flatten()

    # --- Create Scatter Plot ---
    # Use the generated cmap and norm. No need for vmin/vmax now.
    scatter = ax.scatter(
        x_coords_flat,
        y_coords_flat,
        s=300,
        c=initial_connections,
        cmap=cmap,        # Use the discrete colormap
        norm=norm,        # Use the boundary normalization
        edgecolors='grey',
        linewidths=0.5
    )

    # --- Configure Axes ---
    ax.set_xlabel('Orbit Plane Number (P)')
    ax.set_ylabel('Satellite Index in Orbit (N)')
    ax.set_title(f'Satellite Connections (Time Step {initial_step})')
    ax.set_xlim(-0.5, P - 0.5)
    ax.set_ylim(-0.5, N - 0.5) # 0 at bottom, N-1 at top
    ax.set_xticks(np.arange(P))
    ax.set_yticks(np.arange(N))
    # ax.invert_yaxis() # Removed as requested
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgrey')

    # --- Color Bar ---
    # The colorbar will automatically use the cmap and norm from the scatter plot
    cbar = plt.colorbar(scatter, label='Number of Connections')

    # Improve colorbar ticks for discrete integer values
    # Make ticks appear centered within the color blocks for each integer
    tick_locs = np.arange(0, max_connections + 1) # Ticks at 0, 1, 2...
    cbar.set_ticks(tick_locs)
    # Optional: if max_connections is very large, use MaxNLocator
    # locator = MaxNLocator(integer=True)
    # cbar.locator = locator
    # cbar.update_ticks()


    # --- Time Slider ---
    ax_time = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    time_slider = Slider(
        ax=ax_time,
        label='Time Step',
        valmin=min_step,
        valmax=max_step,
        valinit=initial_step,
        valstep=1
    )

    # --- Update Function for Slider ---
    def update(val):
        step = int(time_slider.val)
        connections_at_step = data.get(step, np.zeros((N, P))).flatten()
        # Update the colors of the scatter plot points
        scatter.set_array(connections_at_step)
        ax.set_title(f'Satellite Connections (Time Step {step})')
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        xml_file_path = sys.argv[1]
    else:
        xml_file_path = "/home/yfh/Desktop/Data/1/station_visible_satellites.xml" # MODIFY THIS PATH

    print(f"Loading data from: {xml_file_path}")
    print(f"Grid dimensions: N={N} (rows/vertical), P={P} (columns/horizontal)")

    parsed_data, max_conn, min_t, max_t = parse_xml(xml_file_path, N, P)

    if max_conn == 0:
        print("Warning: Maximum connections found is 0. All points will be white.")
    else:
        print(f"Maximum connections found: {max_conn}")

    print(f"Plotting data for time steps {min_t} to {max_t}...")
    plot_orbital_map(parsed_data, N, P, max_conn, min_t, max_t)

    print("Plot closed.")

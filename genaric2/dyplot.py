from vedo import *
import numpy as np

class TopologyVisualizer:
    def __init__(self, N, P, t, adj_list_array):
        self.N = N
        self.P = P
        self.t = t
        self.adj_list_array = adj_list_array
        self.current_time = 0
        self.plt = None
        self.points_dict = {}
        self.lines_dict = {}

        self.setup_plotter()

    def setup_plotter(self):
        self.plt = Plotter(
            axes=0,
            bg='white',
            interactive=True,
            size=(800, 800)
        )
        self.plt.camera.SetPosition(self.P / 2, self.N / 2, 15)
        self.plt.camera.SetFocalPoint(self.P / 2, self.N / 2, 0)
        self.plt.camera.SetViewUp(0, 1, 0)
        self.plt.camera.ParallelProjectionOn()

    def create_topology_for_time(self, time_idx):
        self.plt.clear()
        self.points_dict[time_idx] = []
        self.lines_dict[time_idx] = []

        time_info = Text2D(f"Time: {time_idx}", pos="top-left", c="black", bg="white", font="Arial")
        self.plt += time_info

        # Coordinate axes from (-1, -1)
        x_axis = Line([-1, -1, 0], [self.P, -1, 0], c='black', lw=2)
        y_axis = Line([-1, -1, 0], [-1, self.N, 0], c='black', lw=2)
        self.plt += [x_axis, y_axis]

        # X ticks and labels
        for i in range(-1, self.P + 1):
            tick = Line([i, -1, 0], [i, -1.2, 0], c='black', lw=1)
            self.plt += tick
            label = Text3D(str(i), pos=[i, -1.5, 0], s=0.3, c='black', justify='center')
            self.plt += label

        # Y ticks and labels
        for j in range(-1, self.N + 1):
            tick = Line([-1, j, 0], [-1.2, j, 0], c='black', lw=1)
            self.plt += tick
            label = Text3D(str(j), pos=[-1.5, j, 0], s=0.3, c='black', justify='center')
            self.plt += label

        # Axis labels
        x_label = Text3D("X", pos=[self.P / 2, -2, 0], s=0.4, c='black', justify='center')
        y_label = Text3D("Y", pos=[-2, self.N / 2, 0], s=0.4, c='black', justify='center')
        self.plt += [x_label, y_label]

        # Nodes
        for i in range(self.P):
            for j in range(self.N):
                point = Point([i, j, 0], r=8, c='blue')
                self.points_dict[time_idx].append(point)
                self.plt += point

        # Edges
        if time_idx < len(self.adj_list_array):
            edges = self.adj_list_array[time_idx]
            for edge in edges:
                start_node, end_node = edge
                start_x = start_node % self.P
                start_y = start_node // self.P
                end_x = end_node % self.P
                end_y = end_node // self.P

                line = Line([start_x, start_y, 0], [end_x, end_y, 0], c='red', lw=2)
                self.lines_dict[time_idx].append(line)
                self.plt += line

    def show_time(self, time_idx):
        if 0 <= time_idx < self.t:
            self.current_time = time_idx
            self.create_topology_for_time(time_idx)

    def show_animation(self):
        def slider_callback(widget, event):
            value = widget.GetRepresentation().GetValue()
            time_idx = int(value)
            self.show_time(time_idx)
            self.plt.render()

        slider = self.plt.add_slider(
            slider_callback,
            0,
            self.t - 1,
            value=0,
            pos='bottom-center',
            title="Time",
            show_value=True
        )

        self.show_time(0)
        self.plt.show()

# # Example usage
# N = 5
# P = 5
# t = 10
# adj_list_array = [
#     [(0, 4), (2, 3), (10, 15)],
#     [(1, 3), (2, 4), (5, 20)],
#     [(0, 2), (3, 4), (7, 12)],
#     [(2, 12), (8, 18), (1, 21)],
# ]
#
# visualizer = TopologyVisualizer(N, P, t, adj_list_array)
# visualizer.show_animation()

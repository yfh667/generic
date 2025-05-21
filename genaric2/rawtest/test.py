from vedo import Plotter, Points, Lines, Axes
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
import numpy as np

# -------------------------------------------------
# 1) 原始坐标（数学 z 向上）
coords = np.array([
    [1, 2, 3],
    [2, 3, 7],
])

# 2) 为了让 z 轴正方向“向下”，把显示用的 z 取负
coords_display = coords.copy()
coords_display[:, 2] *= -1          # (x, y, -z)

# -------------------------------------------------
# 3) 创建 Plotter，并用 Trackball 相机交互
plt = Plotter(title='Z-Axis Down Demo', size=(800, 600), bg='white')
plt.interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())

# 4) 画点和连线
pts  = Points(coords_display,  r=12, c='red')        # 小球
pts.labels(['(1,2,3)', '(2,3,7)'], font='Courier')   # 原始坐标标签

link = Lines(coords_display[0], coords_display[1], c='blue', lw=2)

# 5) 自定义坐标轴：z 轴范围给负值，显式注明 “Down”
axes = Axes(
    xtitle='X',
    ytitle='Y',
    ztitle='Z (Down)',
    xrange=(0, 3),
    yrange=(0, 4),
    zrange=(-8, 0),     # 负到零 → 向下
    axesLineWidth=2,
    numberOfDivisions=4,
)

# 6) 一并显示
plt.show(pts, link, axes, interactive=True)

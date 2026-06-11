# satellite_topology_viewer_3D examples

This module keeps reusable 3D/2D synchronization code under `module`.
The example files provide concrete Starlink paths and group schemes.

For interactive module usage, see:

```text
E:\paper11\generic\src\satellite_topology_viewer_3D\examples\satellite_topology_viewer_3D_usage.ipynb
```

Run the 100s GUI:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\satellite_topology_viewer_3D\examples\run_starlink_2d3d_viewer.py'
```

Run a check without opening the GUI:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\satellite_topology_viewer_3D\examples\run_starlink_2d3d_viewer.py' --check-only --start 1 --end 20
```

Run the long cache:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\satellite_topology_viewer_3D\examples\run_starlink_2d3d_viewer.py' --config 'E:\paper11\generic\src\satellite_topology_viewer_3D\examples\configs\starlink_2d3d_viewer_full.yaml'
```

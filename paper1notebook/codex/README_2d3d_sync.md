# Synced 2D + 3D Topology Viewer

This is an incremental framework under `generic/paper1notebook/codex`. It does not modify `generic/src`.

## Files

- `synced_2d3d_viewer.py`: reusable PyQt window that combines the new 2D topology viewer with a PyVista 3D globe.
- `run_starlink_2d3d_viewer.py`: runnable Starlink_72_22 example using the YAML/config/cache test pack.

The 3D backend is PyVista/VTK only, matching the style of `src/viz/vis3d.py`.

## Run Starlink 0..100s Example

```powershell
cd E:\paper11
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\paper1notebook\codex\run_starlink_2d3d_viewer.py'
```

Useful options:

```powershell
--start 0 --end 100 --stride 1
--link-stride 2
--no-3d-links
--no-orbits
--check-only
```

The 2D and 3D views share the bottom timeline. Selecting an edge in the 2D view highlights the same edge in 3D.

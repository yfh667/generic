# G60 0..86164s Run Results

Command:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\paper1notebook\codex2\build_g60_group_betweenness_86164.py' --start 0 --end 86164 --workers 8 --progress-every 100 --force
```

Output directory:

`E:\paper11\generic\paper1notebook\codex2\outputs\g60_group_betweenness_t0_86164`

Group cache:

`E:\paper11\generic\paper1notebook\codex2\cache\group_data_cache\station_visible_satellites_20250106_G60_t0_86164_stride1.json`

Result:

- Steps: `86165`
- Edges: `3060`
- Unique China-Europe group states: `12224`
- Matrix shape: `(86165, 3060)`
- Value range: `0.0 .. 92.35961151123047`
- Main matrix size: about `1005.8 MB`
- Unique state cache size: about `142.69 MB`

Validation:

- Full first 11 rows match the direct 0..10s computation exactly: max absolute diff `0.0`.
- For the first 20 rows, sum of edge betweenness equals total shortest-path hop count: max error `0.0`.

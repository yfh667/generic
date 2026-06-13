# motif_generator

This module enumerates self-contained exact-box inter-plane motifs.

## Structure

- `module/exact_box.py`: parameter-transparent motif generation functions.
- `module/tiling.py`: tile one motif onto a full `p x n` 2D grid and draw/write the result.
- `module/viewer_adapter.py`: convert tiled motif results to the `EdgeTable` used by `SatelliteTopology2DViewer`.
- `module/support.py`: read/write the recommended support-style motif representation.
- `module/config_io.py`: small YAML loading helper for examples.
- `examples/enumerate_exact_box_motifs.py`: runnable example that writes CSV and JSON outputs.
- `examples/tile_motif_on_grid.py`: runnable example for tiling and drawing one motif on a full grid.
- `examples/run_tiled_motif_2d_viewer.py`: runnable example that opens the tiled motif with the shared 2D topology viewer.
- `examples/configs/dad_cxx_support.yaml`: support-style motif YAML example.
- `examples/motif_generator_usage.ipynb`: self-contained Jupyter usage guide.

## Definition

A motif box has width `w` and height `h`.

- The first `w - 1` columns are planning columns.
- The last column is a target boundary inside the box.
- Candidate symbols are:
  - `A = (dp=1, dy=0)`
  - `B = (dp=1, dy=-1)`
  - `C = (dp=1, dy=1)`
  - `D = (dp=2, dy=0)`
  - `- = idle outgoing terminal`
- Constraints:
  - source out-degree `<= 1`
  - target in-degree `<= 1`
  - maximality: an idle source is allowed only if all legal targets are already occupied

This module enumerates exact-box motifs. It does not merge motifs by global translation equivalence.

## Recommended motif config

Use support-style YAML for a motif that will be tiled or shown:

```yaml
motif:
  name: DAD_Cxx
  w: 3
  h: 3
  offsets:
    A: [1, 0]
    B: [1, -1]
    C: [1, 1]
    D: [2, 0]
  support:
    - [0, 0, D]
    - [0, 1, A]
    - [0, 2, D]
    - [1, 0, C]
```

This is equivalent to Python data like:

```python
motif = {
    "w": 3,
    "h": 3,
    "support": [
        (0, 0, "D"),
        (0, 1, "A"),
        (0, 2, "D"),
        (1, 0, "C"),
    ],
}
```

The `offsets` block makes the coordinate convention explicit, so the support entries only need to say where each edge starts and which symbol it uses.

## Usage

```python
from src.motif_generator.module.exact_box import (
    enumerate_maximal_exact_box,
    enumerate_primitive_exact_box,
    motif_edges_as_user_ids,
    pretty_motif,
)

motifs = enumerate_maximal_exact_box(w=3, h=3, phase_count=36)
primitive = enumerate_primitive_exact_box(w=3, h=3, phase_count=36)

print(len(motifs), len(primitive))
print(pretty_motif(motifs[0]))
print(motif_edges_as_user_ids(motifs[0], row_pitch=36))
```

Run the example:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\motif_generator\examples\enumerate_exact_box_motifs.py' --w 3 --h 3 --phase-count 36 --out-dir 'E:\paper11\generic\src\motif_generator\examples\outputs\w3_h3'
```

Tile one motif on a full `p=18,n=36` grid and open it with the shared 2D topology viewer:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\motif_generator\examples\run_tiled_motif_2d_viewer.py' --config 'E:\paper11\generic\src\motif_generator\examples\configs\dad_cxx_support.yaml' --out-dir 'E:\paper11\generic\src\motif_generator\examples\outputs\viewer_yaml_dad_cxx'
```

Static PNG output is also available for quick non-GUI checks:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\motif_generator\examples\tile_motif_on_grid.py' --config 'E:\paper11\generic\src\motif_generator\examples\configs\dad_cxx_support.yaml' --out-dir 'E:\paper11\generic\src\motif_generator\examples\outputs\tile_yaml_dad_cxx'
```

Notebook usage guide:

`E:\paper11\generic\src\motif_generator\examples\motif_generator_usage.ipynb`

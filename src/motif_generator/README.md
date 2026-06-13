# motif_generator

This module enumerates self-contained exact-box inter-plane motifs.

## Structure

- `module/exact_box.py`: parameter-transparent motif generation functions.
- `module/library.py`: canonical motif-library generation, i.e. primitive exact-box motifs followed by translation/graph deduplication.
- `module/drawing.py`: draw one local motif from support-style input or an enumerated motif matrix.
- `module/tiling.py`: tile one motif onto a full `p x n` 2D grid and draw/write the result.
- `module/viewer_adapter.py`: convert tiled motif results to the `EdgeTable` used by `SatelliteTopology2DViewer`.
- `module/support.py`: read/write the recommended support-style motif representation.
- `module/config_io.py`: small YAML loading helper for examples.
- `examples/enumerate_exact_box_motifs.py`: runnable example that writes CSV and JSON outputs.
- `examples/generate_canonical_motif_library.py`: runnable example for the experiment motif library used by batch simulations.
- `examples/tile_motif_on_grid.py`: runnable example for tiling and drawing one motif on a full grid.
- `examples/run_tiled_motif_2d_viewer.py`: runnable example that opens the tiled motif with the shared 2D topology viewer.
- `examples/configs/dad_cxx_support.yaml`: support-style motif YAML example.
- `examples/configs/motif_000056_support.yaml`: support-style YAML example used in the notebook and 2D viewer demo.
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

## YAML-First Workflow

The recommended user-facing input is a YAML file, for example:

`E:\paper11\generic\src\motif_generator\examples\configs\motif_000056_support.yaml`

Read the YAML motif:

```python
import sys
from pathlib import Path

GENERIC_ROOT = Path(r"E:\paper11\generic")
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.motif_generator.module.config_io import load_motif_support_yaml, load_yaml_dict
from src.motif_generator.module.support import motif_support_label

CONFIG_PATH = GENERIC_ROOT / "src" / "motif_generator" / "examples" / "configs" / "motif_000056_support.yaml"

raw_config = load_yaml_dict(CONFIG_PATH)
motif_support = load_motif_support_yaml(CONFIG_PATH)

print(raw_config["grid"])
print(motif_support.name)
print(motif_support_label(motif_support))
```

Tile it to a full `p x n` topology and write CSV/JSON/PNG outputs:

```python
from pathlib import Path

from src.motif_generator.module.config_io import load_motif_support_yaml, load_yaml_dict
from src.motif_generator.module.tiling import tile_motif_on_grid, write_tiled_motif_outputs

CONFIG_PATH = Path(r"E:\paper11\generic\src\motif_generator\examples\configs\motif_000056_support.yaml")
OUT_DIR = Path(r"E:\paper11\data\linshi\motif_000056_from_yaml")

raw_config = load_yaml_dict(CONFIG_PATH)
grid = raw_config.get("grid", {})
tiling = raw_config.get("tiling", {})
motif_support = load_motif_support_yaml(CONFIG_PATH)

result = tile_motif_on_grid(
    p=int(grid.get("p", 18)),
    n=int(grid.get("n", 36)),
    motif=motif_support,
    horizontal_step=tiling.get("horizontal_step"),
    allow_vertical_overlap=bool(tiling.get("allow_vertical_overlap", True)),
    allow_clipped_right=bool(tiling.get("allow_clipped_right", True)),
)
write_tiled_motif_outputs(result, OUT_DIR)
```

Show the tiled result from YAML with the shared 2D topology viewer:

```python
from PyQt5 import QtWidgets

from src.config.viewer_config import G60_CONFIG
from src.motif_generator.module.config_io import load_motif_support_yaml
from src.motif_generator.module.tiling import tile_motif_on_grid
from src.motif_generator.module.viewer_adapter import tiled_result_to_edge_table
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer

CONFIG_PATH = Path(r"E:\paper11\generic\src\motif_generator\examples\configs\motif_000056_support.yaml")
motif_support = load_motif_support_yaml(CONFIG_PATH)

result = tile_motif_on_grid(p=G60_CONFIG.P, n=G60_CONFIG.N, motif=motif_support)
edge_table = tiled_result_to_edge_table(result)

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
viewer = SatelliteTopology2DViewer(
    G60_CONFIG,
    steps=[1],
    edge_table=edge_table,
    window_title=f"G60 tiled motif 2D: {result.motif_label}",
    group_data={},
    show_groups=False,
)
viewer.resize(1200, 760)
viewer.show()
```

Draw one local motif directly from the support-style Python data:

```python
from pathlib import Path

from src.motif_generator.module.drawing import draw_motif_support

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

draw_motif_support(
    motif,
    out_path=Path(r"E:\paper11\data\linshi\motif_local_preview.png"),
    row_pitch=36,
    title="local motif preview",
)
```

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

Generate the canonical experiment library for one `w,h`:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\motif_generator\examples\generate_canonical_motif_library.py' --w 4 --h 3 --out-dir 'E:\paper11\data\linshi\motif_w4h3_canonical_from_src'
```

Tile one motif on a full `p=18,n=36` grid and open it with the shared 2D topology viewer:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\motif_generator\examples\run_tiled_motif_2d_viewer.py' --config 'E:\paper11\generic\src\motif_generator\examples\configs\motif_000056_support.yaml' --out-dir 'E:\paper11\data\linshi\motif_000056_2d_viewer_from_yaml'
```

Static PNG output is also available for quick non-GUI checks:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\motif_generator\examples\tile_motif_on_grid.py' --config 'E:\paper11\generic\src\motif_generator\examples\configs\motif_000056_support.yaml' --out-dir 'E:\paper11\data\linshi\motif_000056_tile_from_yaml'
```

Open the same YAML in the shared 2D viewer:

```powershell
& 'C:\ProgramData\miniconda3\envs\paper11\python.exe' 'E:\paper11\generic\src\motif_generator\examples\run_tiled_motif_2d_viewer.py' --config 'E:\paper11\generic\src\motif_generator\examples\configs\motif_000056_support.yaml' --out-dir 'E:\paper11\data\linshi\motif_000056_2d_viewer_from_yaml'
```

Notebook usage guide:

`E:\paper11\generic\src\motif_generator\examples\motif_generator_usage.ipynb`

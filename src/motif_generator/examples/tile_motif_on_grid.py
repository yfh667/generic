from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.motif_generator.module.config_io import load_yaml_dict
from src.motif_generator.module.exact_box import motif_from_columns, pretty_motif
from src.motif_generator.module.support import (
    motif_support_from_dict,
    motif_support_label,
    motif_support_to_edge_records,
)
from src.motif_generator.module.tiling import (
    draw_tiled_motif,
    tile_edge_records_on_grid,
    tile_motif_on_grid,
    write_tiled_motif_outputs,
)


DEFAULT_OUT_DIR = THIS_DIR / "outputs" / "tile_motif_on_grid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tile a motif onto a p x n 2D grid and draw it.")
    parser.add_argument("--config", type=Path, default=None, help="YAML file using motif.w/h/support format.")
    parser.add_argument("--p", type=int, default=None, help="Number of plane/x columns.")
    parser.add_argument("--n", type=int, default=None, help="Number of y rows.")
    parser.add_argument(
        "--motif-columns",
        nargs="+",
        default=["DAD", "C--"],
        help="Motif planning columns, e.g. --motif-columns DAD C--",
    )
    parser.add_argument("--horizontal-step", type=int, default=None)
    parser.add_argument("--no-vertical-overlap", action="store_true")
    parser.add_argument("--no-clipped-right", action="store_true")
    parser.add_argument("--show-node-labels", action="store_true")
    parser.add_argument("--hide-patch-boxes", action="store_true")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.config is not None:
        raw_config = load_yaml_dict(args.config)
        grid_raw = raw_config.get("grid", {})
        tiling_raw = raw_config.get("tiling", {})
        if not isinstance(grid_raw, dict) or not isinstance(tiling_raw, dict):
            raise ValueError("YAML grid and tiling sections must be mappings when present")
        motif_support = motif_support_from_dict(raw_config)
        p = int(args.p if args.p is not None else grid_raw.get("p", 18))
        n = int(args.n if args.n is not None else grid_raw.get("n", 36))
        horizontal_step = args.horizontal_step if args.horizontal_step is not None else tiling_raw.get("horizontal_step")
        result = tile_edge_records_on_grid(
            p=p,
            n=n,
            motif_width=motif_support.w,
            motif_height=motif_support.h,
            local_edges=motif_support_to_edge_records(motif_support),
            horizontal_step=None if horizontal_step is None else int(horizontal_step),
            allow_vertical_overlap=False
            if args.no_vertical_overlap
            else bool(tiling_raw.get("allow_vertical_overlap", True)),
            allow_clipped_right=False
            if args.no_clipped_right
            else bool(tiling_raw.get("allow_clipped_right", True)),
        )
        motif_label = motif_support.name or motif_support_label(motif_support)
    else:
        motif = motif_from_columns(tuple(args.motif_columns))
        result = tile_motif_on_grid(
            p=int(args.p if args.p is not None else 18),
            n=int(args.n if args.n is not None else 36),
            motif=motif,
            horizontal_step=args.horizontal_step,
            allow_vertical_overlap=not bool(args.no_vertical_overlap),
            allow_clipped_right=not bool(args.no_clipped_right),
        )
        motif_label = pretty_motif(motif)

    out_dir = Path(args.out_dir)
    write_tiled_motif_outputs(result, out_dir)
    png_path = out_dir / "tiled_motif.png"
    draw_tiled_motif(
        result,
        png_path,
        show_patch_boxes=not bool(args.hide_patch_boxes),
        show_node_labels=True if args.show_node_labels else None,
        dpi=int(args.dpi),
    )

    print(f"[motif-generator-tiling] motif={motif_label}")
    print(f"[motif-generator-tiling] p={result.p}, n={result.n}")
    print(f"[motif-generator-tiling] accepted={result.accepted_count}")
    print(f"[motif-generator-tiling] rejected={result.rejected_count}")
    print(f"[motif-generator-tiling] edges={result.edge_count}")
    print(f"[motif-generator-tiling] out_dir={out_dir}")
    print(f"[motif-generator-tiling] png={png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

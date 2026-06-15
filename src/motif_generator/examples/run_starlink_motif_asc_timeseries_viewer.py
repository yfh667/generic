from __future__ import annotations

import sys
from pathlib import Path


GENERIC_ROOT = Path(__file__).resolve().parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.satellite_topology_viewer.examples.run_starlink_motif_asc_timeseries_viewer import main


if __name__ == "__main__":
    raise SystemExit(main())

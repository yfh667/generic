from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.link_setup_time.examples.run_g60_motif_2x2_link_setup_lst_sweep import main


if __name__ == "__main__":
    raise SystemExit(main())

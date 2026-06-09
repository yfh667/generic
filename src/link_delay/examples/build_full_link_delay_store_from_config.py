from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path


GENERIC_ROOT = Path(__file__).resolve().parents[3]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.link_delay.module.build_full_link_delay_store import main


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path


GENERIC_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "g60_full_link_delay_store.yaml"
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.link_delay.module.build_full_link_delay_store import main as module_main


def with_default_config(argv: list[str]) -> list[str]:
    if "--config" in argv:
        return argv
    return ["--config", str(DEFAULT_CONFIG_PATH), *argv]


def main(argv: list[str] | None = None) -> int:
    return module_main(with_default_config(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

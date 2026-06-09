from __future__ import annotations

from prepare_full_link_delay_store_10000s import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--start",
                "0",
                "--end",
                "86164",
                "--workers",
                "8",
                "--progress-every",
                "32",
                "--chunk-steps",
                "512",
            ]
        )
    )

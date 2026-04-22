from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable
import traceback


@dataclass
class MotifRunResult:
    motif: str
    ok: bool
    payload: dict[str, Any] | None = None
    error: str | None = None


def run_motifs_parallel(
    motifs: list[str],
    worker: Callable[..., dict[str, Any]],
    *,
    max_workers: int = 1,
    worker_kwargs: dict[str, Any] | None = None,
) -> list[MotifRunResult]:
    """
    Run one independent motif per process.

    Keep this helper small and generic so the same motif-level parallel pattern
    can be reused by path export, probability statistics, and region aggregation.
    """
    worker_kwargs = dict(worker_kwargs or {})
    max_workers = max(1, int(max_workers))

    if max_workers == 1 or len(motifs) <= 1:
        out: list[MotifRunResult] = []
        for motif in motifs:
            try:
                payload = worker(motif, **worker_kwargs)
                out.append(MotifRunResult(motif=motif, ok=True, payload=payload))
            except Exception:
                out.append(
                    MotifRunResult(
                        motif=motif,
                        ok=False,
                        error=traceback.format_exc(),
                    )
                )
        return out

    results: list[MotifRunResult] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(worker, motif, **worker_kwargs): motif
            for motif in motifs
        }
        for fut in as_completed(futures):
            motif = futures[fut]
            try:
                results.append(MotifRunResult(motif=motif, ok=True, payload=fut.result()))
            except Exception:
                results.append(
                    MotifRunResult(
                        motif=motif,
                        ok=False,
                        error=traceback.format_exc(),
                    )
                )
    results.sort(key=lambda r: motifs.index(r.motif))
    return results


def print_motif_results(results: list[MotifRunResult]) -> None:
    for res in results:
        if res.ok:
            print(f"[OK] {res.motif}: {res.payload}")
        else:
            print(f"[FAILED] {res.motif}\n{res.error}")

    failed = [r for r in results if not r.ok]
    if failed:
        raise SystemExit(f"{len(failed)} motif task(s) failed.")

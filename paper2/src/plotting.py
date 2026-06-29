from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_phase_envelope(
    path: str | Path,
    *,
    title: str,
    steps: np.ndarray,
    original_values: np.ndarray,
    envelope_values: np.ndarray,
    ylabel: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(steps, dtype=np.float64) / 3600.0
    fig, ax = plt.subplots(figsize=(16.2, 6.8), dpi=180)
    ax.plot(x, original_values, label="single constellation", linewidth=1.0, alpha=0.78)
    ax.plot(x, envelope_values, label="two constellation envelope", linewidth=1.35)
    ax.set_title(title)
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_phase_sweep(
    path: str | Path,
    *,
    title: str,
    sweep_means: np.ndarray,
    stride_seconds: int,
    period_seconds: int,
    ylabel: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(sweep_means), dtype=np.float64) * float(stride_seconds) / float(period_seconds) * 360.0
    fig, ax = plt.subplots(figsize=(14.5, 5.4), dpi=180)
    ax.plot(x, sweep_means, linewidth=1.25)
    ax.set_title(title)
    ax.set_xlabel("phase offset (degree)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path

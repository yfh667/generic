import numpy as np
import pandas as pd
def reliability_global_stats(df, rel_col="rel_0999_099"):
    x = pd.to_numeric(df[rel_col], errors="coerce").dropna()
    return pd.Series({
        "count": int(x.size),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "std": float(x.std(ddof=1)),
        "min": float(x.min()),
        "p05": float(x.quantile(0.05)),
        "p10": float(x.quantile(0.10)),
        "p90": float(x.quantile(0.90)),
        "p95": float(x.quantile(0.95)),
        "max": float(x.max()),
        "time_ratio_rel_lt_0.95": float((x < 0.95).mean()),
        "time_ratio_rel_lt_0.98": float((x < 0.98).mean()),
        "time_ratio_rel_ge_0.99": float((x >= 0.99).mean()),
    }, name=rel_col)

def reliability_window_stats(df, rel_col="rel_0999_099", time_col="time", window_sec=300):
    d = df[[time_col, rel_col]].copy()
    d[time_col] = d[time_col].astype(int)
    d[rel_col] = pd.to_numeric(d[rel_col], errors="coerce")
    t0 = int(d[time_col].min())
    d["win_id"] = ((d[time_col] - t0) // int(window_sec)).astype(int)
    out = d.groupby("win_id", as_index=False).agg(
        start_time=(time_col, "min"),
        end_time=(time_col, "max"),
        rel_mean=(rel_col, "mean"),
        rel_min=(rel_col, "min"),
        rel_p05=(rel_col, lambda s: s.quantile(0.05)),
        rel_p95=(rel_col, lambda s: s.quantile(0.95)),
    )
    out["duration_sec"] = out["end_time"] - out["start_time"] + 1
    return out

def path_switch_stats(df, path_col="path", time_col="time"):
    d = df[[time_col, path_col]].sort_values(time_col).copy()
    d[path_col] = d[path_col].fillna("")
    d["switch"] = d[path_col].ne(d[path_col].shift(1))
    if len(d) > 0:
        d.iloc[0, d.columns.get_loc("switch")] = False
    d["seg_id"] = d["switch"].cumsum()
    seg = d.groupby("seg_id", as_index=False).agg(
        path=(path_col, "first"),
        start_time=(time_col, "min"),
        end_time=(time_col, "max"),
    )
    seg["duration_sec"] = seg["end_time"] - seg["start_time"] + 1
    summary = pd.Series({
        "path_switch_count": int(d["switch"].sum()),
        "segment_count": int(len(seg)),
        "avg_dwell_sec": float(seg["duration_sec"].mean()) if len(seg) else 0.0,
        "median_dwell_sec": float(seg["duration_sec"].median()) if len(seg) else 0.0,
        "max_dwell_sec": int(seg["duration_sec"].max()) if len(seg) else 0,
    }, name="path_switch_summary")
    return summary, seg

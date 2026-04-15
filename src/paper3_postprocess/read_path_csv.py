from pathlib import Path
import re
import pandas as pd


def read_pair_csv(
    csv_path,
    *,
    time_col="time",
    station_a_col="station_a",
    station_b_col="station_b",
    path_col="path",
    ensure_int=True,
):
    """
    读取一份 station 对的最短路径 CSV。

    支持两种情况：
    1) CSV 本身包含 station_a / station_b 列
    2) CSV 没有这两列时，从文件名中提取（如 region1--station12-region2--station19.csv）
    """
    path = Path(csv_path)
    df = pd.read_csv(path)

    # 缺列校验
    if time_col not in df.columns:
        raise ValueError(f"CSV 缺少列: {time_col} -> {path}")

    # 文件名兜底解析 station 对
    if station_a_col not in df.columns or station_b_col not in df.columns:
        m = re.match(r".*station(\d+)-[^-]*-station(\d+)", path.stem)
        if not m:
            raise ValueError(
                f"CSV 缺少 {station_a_col}/{station_b_col}，且文件名无法解析: {path.name}"
            )
        df[station_a_col] = int(m.group(1))
        df[station_b_col] = int(m.group(2))

    # 可选字段保证存在
    if path_col not in df.columns:
        df[path_col] = ""

    if ensure_int:
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce").astype("Int64")
        df[station_a_col] = pd.to_numeric(df[station_a_col], errors="coerce").astype("Int64")
        df[station_b_col] = pd.to_numeric(df[station_b_col], errors="coerce").astype("Int64")

    # 简单标准列
    return (
        df
        .loc[:, [time_col, station_a_col, station_b_col, path_col] + [
            c for c in df.columns
            if c not in {time_col, station_a_col, station_b_col, path_col}
        ]]
        .copy()
    )

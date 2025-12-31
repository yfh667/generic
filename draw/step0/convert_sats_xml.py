import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from lxml import etree  # pip install lxml

# 英文月份缩写 -> 月份数字（避免 Windows/locale 问题）
MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}

def _parse_datetime_tokens(day_str: str, mon_str: str, year_str: str, hms_str: str) -> datetime:
    """
    输入类似: day='26', mon='Dec', year='2025', hms='12:34:56.123456789'
    输出 datetime（微秒精度，截断/补齐到6位）
    """
    day = int(day_str)
    mon = MONTH_MAP.get(mon_str[:3], None)
    if mon is None:
        raise ValueError(f"Unknown month: {mon_str}")
    year = int(year_str)

    # time with optional fractional seconds
    if "." in hms_str:
        time_part, frac = hms_str.split(".", 1)
        frac6 = (frac[:6]).ljust(6, "0")  # 截断/补齐到6位
    else:
        time_part = hms_str
        frac6 = "000000"

    hh, mm, ss = time_part.split(":")
    return datetime(year, mon, day, int(hh), int(mm), int(ss), int(frac6))

def _parse_raw_line(line: str):
    """
    原始行格式假设：
      <DD> <Mon> <YYYY> <HH:MM:SS[.frac]> <x_km> <y_km> <z_km> ...
    返回: (datetime, x_km, y_km, z_km) 或 None
    """
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    dt = _parse_datetime_tokens(parts[0], parts[1], parts[2], parts[3])
    x_km, y_km, z_km = float(parts[4]), float(parts[5]), float(parts[6])
    return dt, x_km, y_km, z_km

def convert_raw_txt_to_xml(raw_txt_path: str, xml_path: str, skip_header_lines: int = 1):
    """
    直接把“原始txt”转成xml：
      - t: 从第一条有效数据起算的相对秒（取 int）
      - x/y/z: km -> m，保留3位小数
    """
    sat_id = os.path.splitext(os.path.basename(raw_txt_path))[0]
    root = etree.Element("sat", id=sat_id)
    tree = etree.ElementTree(root)

    with open(raw_txt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    data_lines = lines[skip_header_lines:] if skip_header_lines > 0 else lines

    # 找到第一条有效数据行作为 start_time
    start_time = None
    for line in data_lines:
        if not line.strip():
            continue
        try:
            parsed = _parse_raw_line(line)
            if parsed is None:
                continue
            start_time = parsed[0]
            break
        except Exception:
            continue

    if start_time is None:
        raise ValueError(f"No valid data lines found in {raw_txt_path}")

    # 逐行写入 <p>
    for line in data_lines:
        if not line.strip():
            continue
        try:
            parsed = _parse_raw_line(line)
            if parsed is None:
                continue
            current_time, x_km, y_km, z_km = parsed
            seconds_since_start = int((current_time - start_time).total_seconds())

            x_m, y_m, z_m = x_km * 1000.0, y_km * 1000.0, z_km * 1000.0
            p = etree.Element(
                "p",
                t=str(seconds_since_start),
                x=f"{x_m:.3f}",
                y=f"{y_m:.3f}",
                z=f"{z_m:.3f}",
            )
            root.append(p)
        except Exception as e:
            print(f"[WARN] skip bad line in {raw_txt_path}: {line.strip()} | {e}")

    os.makedirs(os.path.dirname(xml_path), exist_ok=True)
    with open(xml_path, "wb") as out:
        tree.write(out, encoding="utf-8", pretty_print=True, xml_declaration=False)

    print(f"OK: {raw_txt_path} -> {xml_path}")

def convert_all_raw_txt_to_xml(input_dir: str, output_dir: str, skip_header_lines: int = 1, max_workers: int | None = None):
    os.makedirs(output_dir, exist_ok=True)
    txt_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".txt")]

    tasks = []
    for fname in txt_files:
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, os.path.splitext(fname)[0] + ".xml")
        tasks.append((in_path, out_path, skip_header_lines))

    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 4) * 2)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        ex.map(lambda args: convert_raw_txt_to_xml(*args), tasks)

if __name__ == "__main__":
    convert_all_raw_txt_to_xml(
        input_dir=r"C:\usrspace\mywork\data_paper2\position_raw\g60",
        output_dir=r"C:\usrspace\mywork\data_paper2\position_modify\g60_xml",
        skip_header_lines=1,   # 你原脚本是 lines[1:]；如果确实要跳过前7行就改成 7
    )

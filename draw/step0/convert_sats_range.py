import os
import locale
from datetime import datetime

def _best_effort_set_english_locale():
    """尽量切到英文月份环境，避免中文系统解析失败。"""
    tried = ["C", "en_US.UTF-8", "en_US.utf8", "en_US", "English_United States.1252"]
    current = locale.setlocale(locale.LC_TIME)  # 记录当前
    for name in tried:
        try:
            locale.setlocale(locale.LC_TIME, name)
            return current  # 成功就返回旧值，后面可还原
        except Exception:
            continue
    return current  # 都失败也返回旧值，strptime 可能仍可用
from datetime import datetime

def _parse_ts(ts: str) -> datetime:
    """
    兼容多种时间戳：
    - 2012-02-24 18:00:00.000
    - 2012/02/24 18:00:00.000
    - 24 Feb 2012 18:00:00.000000000
    - 24 Feb 2012 18:00:00
    """
    s = ts.strip().replace("T", " ")

    # 统一小数秒到 6 位（微秒）
    if "." in s:
        head, frac = s.split(".", 1)
        frac = "".join(ch for ch in frac if ch.isdigit())
        frac = (frac + "000000")[:6]
        s6 = f"{head}.{frac}"
    else:
        s6 = s

    formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%d %b %Y %H:%M:%S.%f",
        "%d %b %Y %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s6, fmt)
        except Exception:
            pass
    raise ValueError(f"无法识别的时间格式: {ts}")

def convert_file_to_step_csv(input_file: str, output_file: str, value_name: str = "Range"):
    """
    读取两列文本（Time\t<value>），输出 CSV:
    header: time,<value_name>
    step = 相对首行时间的整秒（四舍五入），首行为 0
    """
    # 读取
    lines = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "cp1252"):
        try:
            with open(input_file, "r", encoding=enc) as f:
                lines = f.readlines()
            break
        except Exception:
            continue
    if lines is None:
        raise RuntimeError(f"无法读取文件编码：{input_file}")

    # 如果首行是表头，自动拿到第二列名（如 Range / AngleRate）
    if lines and "\t" in lines[0] and lines[0].lower().lstrip().startswith("time"):
        try:
            value_name = lines[0].split("\t", 1)[1].strip() or value_name
        except Exception:
            pass

    data = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("time") and "\t" in line:
            continue
        if "\t" not in line:
            continue
        t_str, v_str = line.split("\t", 1)
        t_str = t_str.strip()
        v_str = v_str.strip()
        if not t_str or not v_str:
            continue
        try:
            ts = _parse_ts(t_str)
            val = float(v_str)
            data.append((ts, val))
        except Exception:
            continue

    if not data:
        raise ValueError(f"未在文件中解析到有效数据：{input_file}")

    t0 = data[0][0]
    rows = []
    for ts, val in data:
        dt = (ts - t0).total_seconds()
        step = int(round(dt))
        rows.append((step, val))

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fout:
        fout.write(f"time,{value_name}\n")
        for step, val in rows:
            fout.write(f"{step},{val}\n")


# ===== 批量处理 =====
folder_path = r"C:\usrspace\mywork\data\fixed_coordination\range"
folder_path_out = r"C:\usrspace\mywork\data\fixed_coordination_modify\range"
os.makedirs(folder_path_out, exist_ok=True)

txt_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".txt")])

for filename in txt_files:
    in_path = os.path.join(folder_path, filename)
    out_name = os.path.splitext(filename)[0] + "_step.csv"
    out_path = os.path.join(folder_path_out, out_name)
    convert_file_to_step_csv(in_path, out_path)
    print(f"OK: {filename}  ->  {out_name}")

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

def _parse_ts(ts: str) -> datetime:
    """
    解析形如 '24 Feb 2012 18:00:00.000000000' 的时间戳。
    Python 仅支持到微秒(6位)，这里把小数部分截断/补齐到6位。
    """
    ts = ts.strip()
    # 拆分到秒和小数
    if "." in ts:
        head, frac = ts.split(".", 1)
        frac = ''.join(ch for ch in frac if ch.isdigit())  # 保留数字
        if len(frac) < 6:
            frac = frac.ljust(6, "0")
        else:
            frac = frac[:6]
        ts6 = f"{head}.{frac}"
    else:
        ts6 = ts + ".000000"
    return datetime.strptime(ts6, "%d %b %Y %H:%M:%S.%f")

def convert_file_to_step_csv(input_file: str, output_file: str):
    """
    读取含有两列（Time\tAngleRate）的文本文件，输出 CSV：
    header: step,AngleRate
    step = 相对首行时间的整秒（四舍五入），首行为 0
    """
    # 可能存在 UTF-8 BOM 或 ANSI，做个兜底
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

    old_locale = _best_effort_set_english_locale()
    try:
        data = []
        # 跳过非数据行；数据行为“时间\t数值”
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            # 跳过标题行
            if line.lower().startswith("time") and "\t" in line:
                continue
            # 必须包含一个制表符作为分隔
            if "\t" not in line:
                continue
            t_str, v_str = line.split("\t", 1)
            t_str = t_str.strip()
            v_str = v_str.strip()
            # 有些导出会用多个空格代替 \t，这里再兜底一次
            if not v_str and "  " in line:
                parts = [p for p in line.split() if p]
                t_str = " ".join(parts[:4])  # 'DD Mon YYYY HH:MM:SS.fffffffff'
                v_str = parts[4] if len(parts) > 4 else ""
            if not t_str or not v_str:
                continue
            try:
                ts = _parse_ts(t_str)
                val = float(v_str)
                data.append((ts, val))
            except Exception:
                # 解析失败就跳过该行
                continue

        if not data:
            raise ValueError(f"未在文件中解析到有效数据：{input_file}")

        # 以首条时间为 0 秒
        t0 = data[0][0]
        rows = []
        for ts, val in data:
            dt = (ts - t0).total_seconds()
            step = int(round(dt))  # 不是严格 1s 采样也能对齐最近的秒
            rows.append((step, val))

        # 写 CSV
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as fout:
            fout.write("time,AngleRate\n")
            for step, val in rows:
                fout.write(f"{step},{val}\n")
    finally:
        # 还原 locale
        try:
            locale.setlocale(locale.LC_TIME, old_locale)
        except Exception:
            pass

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

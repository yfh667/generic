from pathlib import Path

# 项目根目录（graph_ga 文件夹的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"

# 子目录
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

# 确保这些目录存在
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#
# from config import INPUT_DIR, OUTPUT_DIR
#
# file_in = INPUT_DIR / "example.csv"
# file_out = OUTPUT_DIR / "result.json"
#
# # 用法
# with open(file_in) as f:
#     data = f.read()

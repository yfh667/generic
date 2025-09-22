# simulation_daa/app.py
import sys, os
from pathlib import Path

# —— 确保能导入项目根的 config.py（包含 DATA_DIR / INPUT_DIR）
ROOT = Path(__file__).resolve().parents[3]  # 根据你的层级调整
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt5 import QtWidgets
from core.dashboard import DashboardWindow

def main():
    # Windows 多进程安全
    try:
        import multiprocessing as mp
        if sys.platform.startswith("win"):
            mp.freeze_support()
            mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # 可改环境 TIME_2_BUILD 或 TOPOLOGY_VERSION
    # os.environ["TIME_2_BUILD"] = "30"  # 示例

    win = DashboardWindow()
    win.resize(1400, 900)
    win.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

# simulation_daa/app.py
import sys
from pathlib import Path
from PyQt5 import QtWidgets

# 确保能 import 到你项目根的 config.py（提供 DATA_DIR/INPUT_DIR）
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.window_manager import DashboardWindow

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
    win = DashboardWindow()
    win.resize(1400, 900)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

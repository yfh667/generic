import sys
import os
from pathlib import Path
from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget

class CesiumViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.load_cesium()
        
    def init_ui(self):
        self.setWindowTitle("Cesium with PyQt6")
        self.resize(1280, 720)
        
        self.browser = QWebEngineView()
        
        # 启用WebGL和硬件加速
        settings = self.browser.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)
        
        layout = QVBoxLayout()
        layout.addWidget(self.browser)
        self.setLayout(layout)
        
    def load_cesium(self):
        # 获取cesium.html绝对路径
       	#current_dir = Path(__file__).parent
       # html_path = current_dir / "Cesium_module" / "pyqt.html"
        html_path = '/home/yfh/Desktop/NS3/manswn/Cesium_module/pyqt.html'
        # 加载本地Cesium页面
        self.browser.load(QUrl.fromLocalFile(str(html_path)))

if __name__ == "__main__":
    # 配置环境变量（必须在QApplication之前）
    os.environ.update({
        "QTWEBENGINE_CHROMIUM_FLAGS": "--enable-webgl --ignore-gpu-blocklist --enable-gpu-rasterization",
        "QT_QPA_PLATFORM_PLUGIN_PATH": f"{sys.prefix}/lib/python3.10/site-packages/PyQt6/Qt6/plugins",
        "QT_ANGLE_PLATFORM": "vulkan"  # Linux用vulkan，Windows用d3d11
    })
    
    # 启用OpenGL共享上下文
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseOpenGLES)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    
    app = QApplication(sys.argv)
    viewer = CesiumViewer()
    viewer.show()
    sys.exit(app.exec())


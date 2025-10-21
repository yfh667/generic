import sys
import os

from PyQt6.QtCore import QUrl
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget


class Widget(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("QWebEngineView, PyQt6, Python")
        self.resize(650, 400)

        layout = QVBoxLayout()
        view = QWebEngineView()
        layout.addWidget(view)
        self.setLayout(layout)

        # view.load(QUrl("file:///web-project/index.html"))
        view.settings().setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        print("----------------------")
        print(view.settings().testAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled))
        view.setUrl(QUrl("https://get.webgl.org/"))

if __name__ == "__main__":
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = f"{sys.prefix}/lib/python3.10/site-packages/PyQt6/Qt6/plugins"
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--enable-webgl --ignore-gpu-blocklist"
    
 
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec())

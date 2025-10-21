import sys
import os
from PyQt6.QtCore import QUrl
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout

class WebGLTest(QWidget):
    def __init__(self, url):
        super().__init__()
        self.view = QWebEngineView()
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)
        
        # 统一配置
        settings = self.view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGL2Enabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
        
        # 加载诊断页面
        self.view.load(QUrl(url))
        self.view.loadFinished.connect(self.on_load)

    def on_load(self):
        self.view.page().runJavaScript("""
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
            const info = {
                webgl1: !!canvas.getContext('webgl'),
                webgl2: !!canvas.getContext('webgl2'),
                renderer: gl ? gl.getParameter(gl.RENDERER) : 'N/A',
                maxTextureSize: gl ? gl.getParameter(gl.MAX_TEXTURE_SIZE) : 0,
                antialiasing: gl ? gl.getContextAttributes().antialias : false
            };
            console.log(JSON.stringify(info, null, 2));
            info;
        """, self.print_webgl_info)

    def print_webgl_info(self, info):
        print("WebGL Capabilities:")
        print(f"  WebGL 1.0: {info['webgl1']}")
        print(f"  WebGL 2.0: {info['webgl2']}")
        print(f"  Renderer: {info['renderer']}")
        print(f"  Max Texture Size: {info['maxTextureSize']}")
        print(f"  Antialiasing: {info['antialiasing']}")

if __name__ == "__main__":
    os.environ.update({
        "QTWEBENGINE_CHROMIUM_FLAGS": "--enable-webgl --ignore-gpu-blocklist --enable-logging --v=1",
        "QT_QPA_PLATFORM_PLUGIN_PATH": f"{sys.prefix}/lib/python3.10/site-packages/PyQt6/Qt6/plugins"
    })
    
    app = QApplication(sys.argv)
    
    # 对比测试不同URL
    test_cases = [
        ("https://get.webgl.org", "基础测试"),
        
    ]
    
    for url, title in test_cases:
        window = WebGLTest(url)
        window.setWindowTitle(title)
        window.show()
        app.exec()


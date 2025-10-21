from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSlider, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, QTimer


class TimeControl(QWidget):
    time_changed = pyqtSignal(float)  # 发送归一化的时间值[0.0-1.0]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.max_time = 0.0
        self.is_playing = False
        self._init_ui()
        self._init_timer()
        self._set_style()

    def _init_ui(self):
        # 主布局
        layout = QVBoxLayout()
        self.setLayout(layout)

        # 控制行布局
        control_layout = QHBoxLayout()

        # 播放/暂停按钮
        self.play_btn = QPushButton()
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self.toggle_play)
        control_layout.addWidget(self.play_btn)

        # 时间标签
        self.time_label = QLabel("Time: 0.0s / 0.0s")
        control_layout.addWidget(self.time_label)

        layout.addLayout(control_layout)

        # 进度条
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.valueChanged.connect(self.update_time_label)
        self.slider.sliderPressed.connect(self.pause)  # 手动拖动时暂停
        layout.addWidget(self.slider)

    def _init_timer(self):
        self.timer = QTimer()
        self.timer.setInterval(50)  # 20fps
        self.timer.timeout.connect(self.advance_playback)

    def _set_style(self):
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #eee;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QPushButton {
                min-width: 60px;
                padding: 5px;
                border-radius: 4px;
                background: #2ecc71;
                color: white;
            }
            QPushButton:checked {
                background: #e74c3c;
            }
        """)
        self.play_btn.setText("▶ Play")

    def set_time_range(self, max_time):
        """设置时间范围（秒）"""
        self.max_time = max_time
        self.slider.setValue(0)
        self.update_time_label()
        self.setEnabled(max_time > 0)

    def _on_slider_changed(self, value):
        """处理滑动事件"""
        normalized = value / 1000.0
        current_time = normalized * self.max_time
        self.time_label.setText(f"Time: {current_time:.1f}s / {self.max_time:.1f}s")
        self.time_changed.emit(normalized)
    def toggle_play(self, checked):
        """切换播放状态"""
        self.is_playing = checked
        self.play_btn.setText("⏸ Pause" if checked else "▶ Play")
        if checked:
            if self.slider.value() >= 1000:
                self.slider.setValue(0)
            self.timer.start()
        else:
            self.timer.stop()

    def advance_playback(self):
        """推进播放进度"""
        new_value = self.slider.value() + 2  # 每次前进0.2%的进度
        if new_value >= 1000:
            new_value = 1000
            self.toggle_play(False)
        self.slider.setValue(new_value)

    def update_time_label(self):
        """更新时间显示"""
        current = self.slider.value() / 1000 * self.max_time
        self.time_label.setText(f"Time: {current:.1f}s / {self.max_time:.1f}s")
        self.time_changed.emit(current / self.max_time)

    def pause(self):
        """暂停播放"""
        if self.is_playing:
            self.toggle_play(False)

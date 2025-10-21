from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout,
    QPushButton, QLineEdit, QTextEdit, QLabel
)
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from core.satellite import  SatelliteManager
import basicSa.utilis.Node as Node
import basicSa.utilis.readdata as readdata
import basicSa.fileread.readsatellite as readsatellite
from config.settings import SatelliteConfig  # 新增导入

import  os
class ControlPanel(QWidget):
  #  satellite_added = pyqtSignal(float, float, float)
    trajectory_loaded = pyqtSignal()  # 添加缺失的信号
    orbit_params_changed = pyqtSignal(int, int)  # 新增信号

    def __init__(self, manager: SatelliteManager, parent=None):  # 添加manager参数
        super().__init__(parent)
        self.manager = manager  # 初始化manager
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        # 文件加载按钮
        self.load_btn = QPushButton('Load Trajectory')
        self.load_btn.clicked.connect(self.load_trajectory_file)
        layout.addWidget(self.load_btn)  # 添加到布局

        # 添加批量加载按钮
        self.load_btn2 = QPushButton('Load ALL Trajectory')
        self.load_btn2.clicked.connect(self.load_trajectory_filenew)
        layout.insertWidget(1, self.load_btn2)  # 添加到加载单个按钮下方

        # 输入表单

        self.setLayout(layout)

        # 新增轨道参数表单
        param_form = QFormLayout()
        self.orbit_num_input = QLineEdit()
        self.sats_per_orbit_input = QLineEdit()
        self.sats_angle = QLineEdit()
        self.track_angle = QLineEdit()
        param_form.addRow("Number of Orbits:", self.orbit_num_input)
        param_form.addRow("Satellites per Orbit:", self.sats_per_orbit_input)
        param_form.addRow("Satangle :", self.sats_angle)
        param_form.addRow("track_angle :", self.track_angle)


        # 参数显示标签
        # Status display
        self.param_display = QLabel("Current parameters: Not set")
      #  self.param_display = QLabel("当前参数: 未设置")
        self.param_display.setStyleSheet("color: #666; font-style: italic;")

        # 设置参数按钮
        # Buttons
        self.set_param_btn = QPushButton('Set Orbit Parameters')

       # self.set_param_btn = QPushButton('设置轨道参数')
        self.set_param_btn.clicked.connect(self._set_orbit_params)

        # 在布局中添加组件（调整顺序）
        layout.addWidget(QLabel("轨道参数配置"))
        layout.addLayout(param_form)
        layout.addWidget(self.set_param_btn)
        layout.addWidget(self.param_display)
        layout.addSpacing(20)  # 添加间距


    def _set_orbit_params(self):
        """处理参数设置"""
        try:
            num_orbits = int(self.orbit_num_input.text())
            sats_per_orbit = int(self.sats_per_orbit_input.text())
            sats_angle = int(self.sats_angle.text())
            track_angle = int(self.track_angle.text())

            if num_orbits <= 0 or sats_per_orbit <= 0:
                raise ValueError("数值必须大于0")

            # 更新显示
            self.param_display.setText(
                f"当前参数: {num_orbits}个轨道，每个轨道{sats_per_orbit}颗卫星，卫星半椎角{90-sats_angle}，轨道倾角{track_angle}"
            )
            self.param_display.setStyleSheet("color: green;")

            # TODO: 存储参数供后续使用
            self.orbit_params = (num_orbits, sats_per_orbit)


            self.orbit_params_changed.emit(num_orbits, sats_per_orbit)  # 触发信号


        except ValueError as e:
            self.param_display.setText(f"错误: {str(e)}")
            self.param_display.setStyleSheet("color: red;")
            self.log.append(f"参数设置失败: {str(e)}")




    def load_trajectory_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Trajectory File", "", "Text Files (*.txt)"
        )
        if not path:
            return

        try:
            # 从文件名提取sat_id
            filename = os.path.basename(path)
            sat_id = int(filename.split('.')[0])  # 文件名格式要求：数字开头

            with open(path, 'r') as f:
                points = []
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) != 4:  # 现在每行是time, x, y, z
                        self.log.append(f"Ignore line {line_num}: 需要4个数值")
                        continue

                    try:
                        time = float(parts[0])
                        x, y, z = map(float, parts[1:4])  # 调整索引
                    except ValueError as e:
                        self.log.append(f"行{line_num} 数据错误: {str(e)}")
                        continue

                    points.append(Node.Node(time=time, x=x, y=y, z=z))

                if points:
                    self.manager.add_trajectory(sat_id, points)
                    self.trajectory_loaded.emit()
                    QMessageBox.information(
                        self,
                        "成功",
                        f"加载卫星{sat_id}，包含{len(points)}个轨迹点"
                    )
                else:
                    QMessageBox.warning(self, "警告", "文件未包含有效数据")

        except ValueError:
            QMessageBox.critical(self, "错误", "文件名必须为数字开头（如123.txt）")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"文件加载失败: {str(e)}")
            self.log.append(f"错误: {str(e)}")

    def load_trajectory_filenew(self):
        """加载文件夹下所有txt文件"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Trajectory Folder", ""
        )
        if not dir_path:
            return
        #
        # 在原来的GUI代码中替换为：
        total_files, success_count, error_files = readsatellite.readsatellite(self.manager,
            dir_path,
            int(self.sats_angle.text()),
            int(self.track_angle.text()),
            int(self.orbit_num_input.text()),
            int(self.sats_per_orbit_input.text()),
            SatelliteConfig.RAAN_INCREMENT
        )


        # satsnodes = readdata.readsats_multi(dir_path, int(self.sats_angle.text()))  # 获取地面站数据
        #
        # track_angle = int(self.track_angle.text())
        #
        # total_files = 0
        # success_count = 0
        # error_files = []
        #
        # P =  int(self.orbit_num_input.text())
        # N =  int(self.sats_per_orbit_input.text())
        #
        # for satsnode in satsnodes:
        #     id = satsnode[0].nodeid-1
        #     i_index = id //N
        #
        #     RAAN = (i_index) * SatelliteConfig.RAAN_INCREMENT
        #     self.manager.add_trajectory(id, satsnode)
        #
        #     readdata.add_number_RAAN(satsnode, RAAN, track_angle)
        #     success_count += 1
        #
        #
        #
        # for filename in os.listdir(dir_path):
        #     if not filename.lower().endswith('.txt'):
        #         continue
        #     total_files += 1


        # 显示汇总结果
        summary = [
            "批量加载完成：",
            f"总文件数: {total_files}",
            f"成功加载: {success_count}",
            f"失败文件: {len(error_files)}"
        ]
        if error_files:
            summary.append("\n错误详情：")
            summary.extend(error_files[:5])  # 最多显示5个错误
            if len(error_files) > 5:
                summary.append(f"（其余{len(error_files) - 5}个错误详见日志）")

        QMessageBox.information(
            self,
            "批量加载结果",
            "\n".join(summary)
        )

        if success_count > 0:
            self.trajectory_loaded.emit()



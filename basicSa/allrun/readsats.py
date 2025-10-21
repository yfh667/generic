import basicSa.utilis.readdata as readdata

import basicSa.satellite_stk_alpha.config.settings as settings  # 新增导入
import os
def load_trajectory_filenew(dir_path,satsangle,track_angle,P,N):
    """加载文件夹下所有txt文件"""

    if not dir_path:
        return

    satsnodes = readdata.readsats_multi(dir_path, satsangle)  # 获取地面站数据

   # track_angle = int(self.track_angle.text())

    total_files = 0
    success_count = 0
    error_files = []

    # P =  int(self.orbit_num_input.text())
    # N =  int(self.sats_per_orbit_input.text())
    realnodes= [None]*len(satsnodes)

    for satsnode in satsnodes:
        id = satsnode[0].nodeid -1
        i_index = id //N

        RAAN = (i_index) * settings.SatelliteConfig.RAAN_INCREMENT
       # self.manager.add_trajectory(id, satsnode)

        readdata.add_number_RAAN(satsnode, RAAN, track_angle)
        realnodes[id]= satsnode
        success_count += 1



    for filename in os.listdir(dir_path):
        if not filename.lower().endswith('.txt'):
            continue
        total_files += 1


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
    return realnodes

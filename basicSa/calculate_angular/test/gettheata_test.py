from datetime import datetime
import numpy as np
import basicSa.calculate_angular.gettheta as gettheta
# ECEF 坐标（单位任意一致）
r_ecef = np.array([4404.651206 , -5971.834353  , 592.805101])  # km

# UTC 时间
utc_time = datetime(2012, 2, 24, 18, 0, 0)

# 转换
r_eci = gettheta.ecef_to_eci(r_ecef, utc_time)
print("ECI Coordinates:", r_eci)

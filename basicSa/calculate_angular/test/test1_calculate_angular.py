
import basicSa.calculate_angular.calculate_angular as calculate_angular


if __name__ == "__main__":
    xa1 =  7421189.291
    ya1 = 10193.282
    za1 =583970.277
    xa2 =   7420611.583
    ya2 = 10320.591
    za2 =  591263.808
    xb1 =    6986350.779
    yb1 = 2291372.540
    zb1 =1164341.165
    xb2 = 6985219.746
    yb2 =  2291137.662
    zb2 = 1171566.893
    timestep = 1
    RAAN_deg = 0 # 输入为度数
    beta_deg = 89.0  # 输入为度数

    params = (xa1, ya1, za1, xa2, ya2, za2,
              xb1, yb1, zb1, xb2, yb2, zb2,
              beta_deg, timestep, RAAN_deg)
    w = calculate_angular.Get_angular_velocity(params)



    print(w)

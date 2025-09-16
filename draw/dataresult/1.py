import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ====== 示例数据（换成你自己的） ======
lats = [40, -20, 35, 10, 50]   # 纬度
lons = [-100, -60, 30, 120, 80] # 经度
hops = [20, 35, 28, 18, 32]     # 平均跳数

# ====== 绘图部分 ======
plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# 添加地图元素
ax.set_global()
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.4)
ax.add_feature(cfeature.LAND, facecolor='lightgray')  # 灰色陆地
ax.add_feature(cfeature.OCEAN, facecolor='white')     # 白色海洋

# 画散点，颜色映射到 hops
sc = ax.scatter(lons, lats, c=hops, cmap='Spectral_r', s=60, edgecolor='k', transform=ccrs.PlateCarree())

# 添加颜色条
cbar = plt.colorbar(sc, orientation='vertical', shrink=0.8, pad=0.05)
cbar.set_label("Average Hop Count to other end users")

plt.title("Global Distribution of Average Hop Count", fontsize=14)
plt.show()

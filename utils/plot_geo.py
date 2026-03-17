import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

config = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 16,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
}
plt.rcParams.update(config)



# 读取文件
data_dir = "../data/sz/Shenzhen_geo_data/"
shp_file = "Shenzhen_Street.shp"
data = gpd.read_file(data_dir + shp_file)

# 绘制地图
data.plot()

# 显示地图
plt.show()
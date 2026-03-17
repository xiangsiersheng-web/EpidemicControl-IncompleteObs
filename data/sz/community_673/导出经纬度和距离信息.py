import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from geopy.distance import geodesic

def get_lon_lat():
    # geo文件
    geo_path = '../Shenzhen_geo_data/Shenzhen_Community.shp'

    # Mapping 文件
    community_path = '../community_654'
    map_path = community_path + '/mapping.csv'

    gdf = gpd.read_file(geo_path)
    mapping = pd.read_csv(map_path)

    # 计算中心点经纬度
    gdf['longitude'] = gdf.geometry.centroid.x  # 经度
    gdf['latitude'] = gdf.geometry.centroid.y  # 纬度

    # 提取OBJECTID和经纬度
    result = gdf[['OBJECTID', 'longitude', 'latitude']]

    # 合并mapping文件，按filtered_idx排序
    result = result.merge(mapping, on='OBJECTID', how='right')
    result = result.sort_values('filtered_idx')

    # 保存到CSV文件
    result = result[['filtered_idx', 'longitude', 'latitude']]
    result.to_csv(community_path + '/lon_lat.csv', index=False)

    # 计算一个距离矩阵
    coords = result[['longitude', 'latitude']].values

    # 计算球面距离矩阵
    n = len(coords)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i][j] = geodesic(coords[i][::-1], coords[j][::-1]).km

    # 导出为NumPy数组
    np.save(community_path + '/distance_matrix.npy', distance_matrix)

if __name__ == '__main__':
    get_lon_lat()
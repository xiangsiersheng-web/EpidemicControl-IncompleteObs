import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from geopy.distance import geodesic

def get_lon_lat():
    # geo file
    geo_path = '../Shenzhen_geo_data/Shenzhen_Community.shp'

    # Mapping file
    community_path = '../community_654'
    map_path = community_path + '/mapping.csv'

    gdf = gpd.read_file(geo_path)
    mapping = pd.read_csv(map_path)

    # Calculate centroid longitude and latitude
    gdf['longitude'] = gdf.geometry.centroid.x  # Longitude
    gdf['latitude'] = gdf.geometry.centroid.y  # Latitude

    # Extract OBJECTID and coordinates
    result = gdf[['OBJECTID', 'longitude', 'latitude']]

    # Merge mapping file, sort by filtered_idx
    result = result.merge(mapping, on='OBJECTID', how='right')
    result = result.sort_values('filtered_idx')

    # Save to CSV file
    result = result[['filtered_idx', 'longitude', 'latitude']]
    result.to_csv(community_path + '/lon_lat.csv', index=False)

    # Calculate a distance matrix
    coords = result[['longitude', 'latitude']].values

    # Calculate geodesic distance matrix
    n = len(coords)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i][j] = geodesic(coords[i][::-1], coords[j][::-1]).km

    # Export as NumPy array
    np.save(community_path + '/distance_matrix.npy', distance_matrix)

if __name__ == '__main__':
    get_lon_lat()
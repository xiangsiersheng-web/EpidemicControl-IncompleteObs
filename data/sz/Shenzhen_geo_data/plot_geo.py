import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import LineString

# 1. Read map file
shp_file = "Shenzhen_Community.shp"
data = gpd.read_file(shp_file)
data.sort_values(by="OBJECTID", inplace=True)
data.reset_index(drop=True, inplace=True)  # Reset index to ensure index starts from 0
num_areas = len(data)
print(f"Number of communities: {num_areas}")

# 2. Generate flow matrix
# Here we generate a random flow matrix as an example, you can replace it with actual data
np.random.seed(42)  # Set random seed to ensure reproducibility
flow_matrix = np.random.randint(0, 500, size=(num_areas, num_areas))
np.fill_diagonal(flow_matrix, 0)  # Set diagonal to 0, meaning no self-flow
print("Flow matrix example:")
print(flow_matrix)

# If you have actual flow data, you can load it directly, for example from CSV file or .npy file
# Example loading methods (please adjust according to actual file path and format):
# flow_matrix = np.load('flow_matrix.npy')  # Load from .npy file
# or
# flow_df = pd.read_csv("flow_matrix.csv", index_col=0)
# flow_matrix = flow_df.values

# 3. Set flow threshold for filtering
flow_threshold = 990  # Connections with flow less than this value will not be plotted

# 4. Create an empty graph
G = nx.Graph()

# Add each region as a node to the graph
for idx, row in data.iterrows():
    region_id = idx  # Use index as region ID
    G.add_node(region_id, geometry=row.geometry)

# 5. Filter and add flow edges
for i in range(num_areas):
    for j in range(i + 1, num_areas):  # Only traverse upper triangular matrix to avoid duplicates
        flow_value = flow_matrix[i, j] + flow_matrix[j, i]  # Sum of bidirectional flow

        if flow_value >= flow_threshold:
            # Add edge to graph, flow as edge attribute
            G.add_edge(i, j, weight=flow_value)
print("Number of edges after filtering:", len(G.edges))
# 6. Plot map
fig, ax = plt.subplots(figsize=(12, 12))

# Plot base map layer
data.plot(ax=ax, color='lightblue', edgecolor='black')

 # Plot flow edges
for u, v, edge_data in G.edges(data=True):
    # Get region geometry
    u_geom = G.nodes[u]["geometry"]
    v_geom = G.nodes[v]["geometry"]

    # Get representative point (centroid) of region
    u_point = u_geom.centroid
    v_point = v_geom.centroid

    # Calculate line connecting two regions
    line = LineString([u_point, v_point])

    # Plot edge, line width proportional to flow value
    ax.plot(
        *line.xy,
        color="red",
        linewidth=edge_data["weight"] / 1000,  # Adjust scale factor to fit actual flow range
        alpha=0.6
    )

# Set map title and other parameters
ax.set_title("Population Flow Map", fontsize=15)
plt.axis('off')  # Turn off axis

# Display map
plt.show()

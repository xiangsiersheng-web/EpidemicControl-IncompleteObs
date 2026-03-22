"""
For displaying flow relationships on a map (refactored version: more readable, modular, functionality unchanged)
"""
# === Standard library / Third-party dependencies ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from shapely.geometry import LineString
from shapely.ops import unary_union
from pyproj import CRS, Geod

# === Matplotlib global styles ===
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 15,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
})

# === Path configuration (consistent with original script) ===
SHP_FILE = "../data/sz/Shenzhen_geo_data/Shenzhen_Community.shp"
POP_FILE = "../data/sz/community_654/population.npy"
OD_FILE  = "../data/sz/community_654/flow.npy"
MAP_CSV  = "../data/sz/community_654/mapping.csv"


# ---------------------------------------------------------------------
# Utility functions (plotting components)
# ---------------------------------------------------------------------
def plot_city_outline(ax, gdf: gpd.GeoDataFrame, heal_tol=None, **line_kw):
    """
    Only plot city outline (outer ring), avoiding thickening internal adjacent boundaries.
    heal_tol: Optional, unit is the length unit of current projection (meters/degrees), used to fix small gaps:
              First buffer(+tol) then buffer(-tol).
    line_kw: Style parameters passed to plot, such as color, linewidth, zorder.
    """
    geom = gdf.geometry
    if heal_tol and heal_tol > 0:
        geom = geom.buffer(heal_tol).buffer(-heal_tol)

    union = unary_union(geom).buffer(0)  # Union and fix geometry validity

    outlines = []
    if union.geom_type == "Polygon":
        outlines.append(LineString(union.exterior.coords))
    elif union.geom_type == "MultiPolygon":
        for p in union.geoms:
            outlines.append(LineString(p.exterior.coords))
    else:
        return  # Not a polygon, skip drawing

    gpd.GeoSeries(outlines, crs=gdf.crs).plot(ax=ax, **line_kw)


def add_north(ax, x=0.05, y=0.9, size=0.06):
    """
    Draw north arrow (pointing up): arrow from (x,y) to (x,y+size), coordinate system is axes fraction.
    """
    ax.annotate(
        'N',
        xy=(x, y + size),           # Arrow tip (higher position)
        xytext=(x, y),              # Text position (lower position)
        xycoords='axes fraction',
        textcoords='axes fraction',
        ha='center', va='center',
        arrowprops=dict(arrowstyle='-|>', lw=1.2, color='k', mutation_scale=15)
    )


def add_scalebar(ax, gdf, length_km=20, where=(0.35, 0.04),
                 tick_fracs=(0, 0.25, 0.5, 1.0), tick_labels=None,
                 unit="km", unit_on_last_only=True,
                 linewidth=2, fontsize=9, color="k"):
    """
    Add scale bar to current coordinate system (projected coordinates in meters; geographic coordinates use geodesic calculation).
    Note: To be consistent with original script, tick_fracs will be overridden to 9 equal divisions (0,1/8,...,1.0).

    - length_km: Total length of scale bar (kilometers)
    - where: Relative position of scale bar left endpoint in axes (axes fraction)
    - tick_fracs: Tick relative positions (will be overridden to 9 equal divisions to maintain original functionality)
    - tick_labels: Tick labels, length must match tick_fracs; auto-generated when None
    - unit_on_last_only: Only add unit to the last tick label
    """
    # —— Maintain original functionality: force 9 equal division ticks —— (remove this line if not wanted)
    tick_fracs = (0, 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8, 1.0)

    if tick_labels is None:
        labs = []
        for i, f in enumerate(tick_fracs):
            # Put empty labels at 1/8, 3/8, 5/8, 7/8 positions (maintain original display style)
            if i in {1, 3, 5, 7}:
                labs.append(" ")
                continue
            v = length_km * f
            if f == 0:
                labs.append("0")
            else:
                if unit_on_last_only and i != len(tick_fracs) - 1:
                    labs.append(f"{v:g}")
                else:
                    labs.append(f"{v:g} {unit}")
        tick_labels = labs
    else:
        if len(tick_labels) != len(tick_fracs):
            raise ValueError("tick_labels length must match tick_fracs.")

    minx, miny, maxx, maxy = gdf.total_bounds
    x0 = minx + where[0] * (maxx - minx)
    y0 = miny + where[1] * (maxy - miny)

    dy = (maxy - miny)
    tick_h = 0.004 * dy
    text_off = 0.01 * dy

    crs = CRS.from_user_input(gdf.crs) if gdf.crs else None

    if crs is not None and crs.is_projected:
        # —— Projected coordinates: in meters ——
        L = length_km * 1000.0
        x1 = x0 + L
        ax.plot([x0, x1], [y0, y0], color=color, lw=linewidth)

        for frac, lab in zip(tick_fracs, tick_labels):
            xx = x0 + L * float(frac)
            ax.plot([xx, xx], [y0 - tick_h, y0 + tick_h], color=color, lw=1)
            ax.text(xx, y0 - text_off, lab, ha="center", va="top", fontsize=fontsize)
    else:
        # —— Geographic coordinates (lat/lon): use geodesic to convert km -> longitude difference ——
        geod = Geod(ellps="WGS84")
        lon0, lat0 = float(x0), float(y0)

        # Main line (keep horizontal: azimuth 90°)
        lon1, lat1, _ = geod.fwd(lon0, lat0, 90, length_km * 1000.0)
        ax.plot([lon0, lon1], [lat0, lat0], color=color, lw=linewidth)

        for frac, lab in zip(tick_fracs, tick_labels):
            loni, lati, _ = geod.fwd(lon0, lat0, 90, length_km * 1000.0 * float(frac))
            ax.plot([loni, loni], [lat0 - tick_h, lat0 + tick_h], color=color, lw=1)
            ax.text(loni, lat0 - text_off, lab, ha="center", va="top", fontsize=fontsize)

    ax.text(x0, y0 + 3 * tick_h, "Scale", ha="left", va="bottom", fontsize=fontsize)


# ---------------------------------------------------------------------
# Data preparation functions
# ---------------------------------------------------------------------
def load_map(shp_path: str) -> gpd.GeoDataFrame:
    """Read community polygons and sort by OBJECTID."""
    gdf = gpd.read_file(shp_path)
    gdf.sort_values(by="OBJECTID", inplace=True)
    print(len(gdf))
    return gdf


def load_flow_and_remap(pop_path: str, od_path: str, map_csv: str, n_target: int) -> np.ndarray:
    """
    Read population and flow matrix, map from original indices to target (shp) index space according to mapping.csv,
    generate new OD consistent with gdf row count. Functionality remains consistent with original script.
    """
    POP = np.load(pop_path)
    OD  = np.load(od_path)
    OD  = OD * POP.reshape(POP.shape[0], 1)

    mapping = pd.read_csv(map_csv)
    original_to_filtered = {}
    for i in range(len(mapping)):
        original_idx = mapping.iloc[i]["original_idx"]
        filtered_idx = mapping.iloc[i]["filtered_idx"]
        original_to_filtered[original_idx] = filtered_idx

    new_OD = np.zeros((n_target, n_target))
    for i in range(n_target):
        for j in range(n_target):
            if i not in original_to_filtered or j not in original_to_filtered:
                continue
            new_OD[i][j] = OD[original_to_filtered[i]][original_to_filtered[j]]

    return new_OD


# ---------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------
def plot_flow_map(
    gdf,                    # Community polygons GeoDataFrame (sorted by OBJECTID)
    OD,                     # Flow matrix (weighted by *POP, mapped to gdf index space)
    bins=(0, 1e5, 2.5e5, 4.5e5, 8.5e5, 1.75e6),
    topk=None,              # Only plot top k edges with largest flow; None means disabled
    min_flow=None,          # Only plot edges >= min_flow; None means disabled
    crs_proj="EPSG:3857",   # Projected coordinate system (for centroid and scale bar; pass None if already in meters)
    figsize=(12, 6),
    save_path=None
):
    assert OD.shape[0] == len(gdf), "OD dimensions must match number of communities"

    # —— Projection and centroid —— (maintain original logic)
    if crs_proj is not None and (gdf.crs is None or gdf.crs.to_string() != crs_proj):
        gdf_plot = gdf.to_crs(crs_proj)
    else:
        gdf_plot = gdf

    cent = gdf_plot.geometry.centroid
    xs = cent.x.to_numpy()
    ys = cent.y.to_numpy()

    # —— Undirected flow (i↔j merged), only upper triangle —— (maintain original logic)
    flow = OD + OD.T
    iu = np.triu_indices_from(flow, k=1)
    pairs = np.stack([iu[0], iu[1]], axis=1)
    flows = flow[iu]

    # —— Filter by threshold / TopK —— (maintain original logic)
    mask = np.ones_like(flows, dtype=bool)
    if min_flow is not None:
        mask &= (flows >= float(min_flow))
    if topk is not None and topk > 0:
        top_idx = np.argpartition(flows, -topk)[-topk:]
        top_mask = np.zeros_like(flows, dtype=bool)
        top_mask[top_idx] = True
        mask &= top_mask
    pairs = pairs[mask]
    flows = flows[mask]
    if len(flows) == 0:
        print("No lines meet the conditions for plotting.")
        return
    print(f"flows max: {flows.max()}, min: {flows.min()}")

    # —— Binning and color/linewidth —— (maintain original logic)
    bins = np.asarray(bins, dtype=float)
    bin_ids = np.digitize(flows, bins, right=True) - 1  # 0..len(bins)-2
    n_bins = len(bins) - 1
    cmap = plt.get_cmap("RdYlGn_r", n_bins)            # Low flow in green, high flow in red
    colors = [cmap(i) for i in range(n_bins)]
    widths = np.linspace(0.7, 3.2, n_bins)             # High flow is thicker

    # —— Plot preparation —— (boundaries, outline)
    fig, ax = plt.subplots(figsize=figsize)
    # Community boundaries
    gdf_plot.boundary.plot(ax=ax, linewidth=0.6, color="0.7", zorder=1)
    # City outline (replaces dissolve().boundary, avoids thickening internal boundaries)
    plot_city_outline(ax, gdf_plot, heal_tol=None, color="k", linewidth=1.2, zorder=2)

    # —— Batch plot binned line segments (better performance than individual plot) —— (maintain original logic)
    for b in range(n_bins):
        sel = (bin_ids == b)
        if not np.any(sel):
            continue
        segs = [[(xs[i], ys[i]), (xs[j], ys[j])] for i, j in pairs[sel]]
        lc = LineCollection(segs, colors=[colors[b]], linewidths=widths[b], alpha=0.9, zorder=3)
        ax.add_collection(lc)

    # —— Legend (bin labels) —— (maintain original logic)
    handles, labels = [], []
    for i in range(n_bins):
        lo = int(bins[i])
        hi = int(bins[i + 1])
        lab = f"< {hi:,}" if i == 0 else f"{lo:,} - {hi:,}"
        h = Line2D([0], [0], color=colors[i], lw=widths[i])
        handles.append(h)
        labels.append(lab)
    # leg = ax.legend(handles, labels, title="Flow", loc="upper right", frameon=True)
    leg = ax.legend(handles, labels, title="Flow (persons/day)", loc="upper right", frameon=True)
    leg._legend_box.align = "left"

    # —— North arrow —— (maintain original logic)
    add_north(ax)

    # —— Generic scale bar (works with any CRS) —— (maintain original call)
    add_scalebar(ax, gdf_plot, length_km=20, where=(0.35, 0.03), fontsize=16)

    # —— Axis style —— (maintain original logic)
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Map data
    data = load_map(SHP_FILE)

    # 2) Flow matrix (mapped to data index space via mapping)
    new_OD = load_flow_and_remap(POP_FILE, OD_FILE, MAP_CSV, n_target=len(data))

    # 3) Plot (parameters consistent with original script)
    for topk in [2000]:
        # topk = 800
        save_path = f"flow_map_{topk}_en.png"
        plot_flow_map(
            data,
            new_OD,
            bins=(0, 2e3, 3e3, 5e3, 8e3, 1.2e4, 2e4),
            # min_flow=1e5,          # Or topk=150
            topk=topk,
            crs_proj="EPSG:3857",
            figsize=(12, 6),
            save_path=save_path
        )

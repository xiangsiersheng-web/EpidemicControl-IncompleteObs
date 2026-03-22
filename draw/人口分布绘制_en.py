"""
For displaying population distribution on a map (refactored version: more readable, modular, style consistent with flow map)
"""
# === Standard library / Third-party dependencies ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from shapely.geometry import LineString
from shapely.ops import unary_union
from pyproj import CRS, Geod

# === Matplotlib global styles (consistent) ===
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
OD_FILE  = "../data/sz/community_654/flow.npy"        # No longer used, but path kept for compatibility
MAP_CSV  = "../data/sz/community_654/mapping.csv"


# ---------------------------------------------------------------------
# Utility functions (plotting components) - consistent with original script
# ---------------------------------------------------------------------
def plot_city_outline(ax, gdf: gpd.GeoDataFrame, heal_tol=None, **line_kw):
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
        return

    gpd.GeoSeries(outlines, crs=gdf.crs).plot(ax=ax, **line_kw)


def add_north(ax, x=0.05, y=0.9, size=0.06):
    ax.annotate(
        'N',
        xy=(x, y + size),
        xytext=(x, y),
        xycoords='axes fraction',
        textcoords='axes fraction',
        ha='center', va='center',
        arrowprops=dict(arrowstyle='-|>', lw=1.2, color='k', mutation_scale=15)
    )


def add_scalebar(ax, gdf, length_km=20, where=(0.35, 0.04),
                 tick_fracs=(0, 0.25, 0.5, 1.0), tick_labels=None,
                 unit="km", unit_on_last_only=True,
                 linewidth=2, fontsize=9, color="k"):
    # —— Maintain original functionality: force 9 equal division ticks ——
    tick_fracs = (0, 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8, 1.0)

    if tick_labels is None:
        labs = []
        for i, f in enumerate(tick_fracs):
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
        L = length_km * 1000.0
        x1 = x0 + L
        ax.plot([x0, x1], [y0, y0], color=color, lw=linewidth)

        for frac, lab in zip(tick_fracs, tick_labels):
            xx = x0 + L * float(frac)
            ax.plot([xx, xx], [y0 - tick_h, y0 + tick_h], color=color, lw=1)
            ax.text(xx, y0 - text_off, lab, ha="center", va="top", fontsize=fontsize)
    else:
        geod = Geod(ellps="WGS84")
        lon0, lat0 = float(x0), float(y0)
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


def load_population_and_remap(pop_path: str, map_csv: str, n_target: int) -> np.ndarray:
    """
    Read population array POP (original indices), map to shp index space according to mapping.csv,
    return new population array new_POP consistent with gdf row count.

    Note: To be compatible with the original script's mapping logic, we use its column names and direction.
    If your mapping.csv means "shp_index -> original_index",
    please swap the usage of original_idx / filtered_idx.
    """
    POP = np.load(pop_path)

    mapping = pd.read_csv(map_csv)
    # Original script constructs original_idx -> filtered_idx dictionary
    original_to_filtered = {}
    for i in range(len(mapping)):
        original_idx = mapping.iloc[i]["original_idx"]
        filtered_idx = mapping.iloc[i]["filtered_idx"]
        original_to_filtered[original_idx] = filtered_idx

    # We need an array with target space (shp order) length
    new_POP = np.zeros(n_target, dtype=float)

    # Here we follow original script's "query original index by target index i" approach,
    # if mapping direction is opposite, swap dictionary keys/values or change to filtered_to_original
    for i in range(n_target):
        if i not in original_to_filtered:
            continue
        ori = original_to_filtered[i]
        if 0 <= ori < len(POP):
            new_POP[i] = POP[ori]

    return new_POP


# ---------------------------------------------------------------------
# Main plotting function: population distribution
# ---------------------------------------------------------------------
def plot_population_map(
    gdf,
    population,
    bins=(0, 2e3, 3e3, 5e3, 8e3, 1.2e4, 2e4),
    crs_proj="EPSG:3857",
    figsize=(12, 6),
    save_path=None,
    # === New: index label related ===
    label_indices=False,             # True to enable index labels
    label_by="position",             # "position" | "OBJECTID" | column name | callable(row)->str
    label_fontsize=8,
    label_color="k",
    label_fmt=None,                  # Like lambda x: f"{x}"
    # === New: border highlight related ===
    highlight_indices=None,          # List of indices to highlight (interpreted with label_by semantics, see below)
    highlight_by="position",         # "position" | "OBJECTID" | column name
    highlight_linewidth=2.0,
    highlight_color="k",
    highlight_zorder=3,
):
    assert len(population) == len(gdf), "population length must match number of communities"

    # —— Projection (maintain original logic) ——
    if crs_proj is not None and (gdf.crs is None or gdf.crs.to_string() != crs_proj):
        gdf_plot = gdf.to_crs(crs_proj)
    else:
        gdf_plot = gdf.copy()

    # —— Binning and color mapping (maintain original logic) ——
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap
    bins = np.asarray(bins, dtype=float)
    if np.any(np.diff(bins) <= 0):
        raise ValueError("bins must be strictly increasing.")

    n_bins = len(bins) - 1
    # cmap = plt.get_cmap("RdYlGn_r", n_bins)
    cmap = plt.get_cmap("YlOrRd", n_bins)
    listed_cmap = ListedColormap([cmap(i) for i in range(n_bins)])
    norm = BoundaryNorm(bins, ncolors=n_bins, clip=False)

    gdf_plot["population"] = population

    # —— Plot —— (polygon fill + internal boundaries + outer outline preserved)
    fig, ax = plt.subplots(figsize=figsize)

    gdf_plot.plot(
        ax=ax,
        column="population",
        cmap=listed_cmap,
        norm=norm,
        linewidth=0.3,
        edgecolor="0.7",
        zorder=1
    )

    # City outline bold
    plot_city_outline(ax, gdf_plot, heal_tol=None, color="k", linewidth=1, zorder=2)

    # ======== New: highlight selected region borders ========
    def _select_rows(by, values):
        """
        Convert user-provided highlight_indices to row selector (boolean vector) based on by semantics.
        - by == "position": values are treated as row number list 0..len(gdf_plot)-1
        - by == "OBJECTID" or string column name: values are treated as value list in that column
        """
        import numpy as np
        mask = np.zeros(len(gdf_plot), dtype=bool)
        if values is None or len(values) == 0:
            return mask
        if by == "position":
            idxs = [v for v in values if isinstance(v, (int, np.integer)) and 0 <= v < len(gdf_plot)]
            mask[idxs] = True
        else:
            col = by if by in gdf_plot.columns else None
            if col is None and by == "OBJECTID" and "OBJECTID" in gdf_plot.columns:
                col = "OBJECTID"
            if col is None:
                raise ValueError(f'highlight_by="{by}" does not exist in columns and is not "position"')
            mask = gdf_plot[col].isin(values).to_numpy()
        return mask

    if highlight_indices:
        sel_mask = _select_rows(highlight_by, highlight_indices)
        if sel_mask.any():
            gdf_plot.loc[sel_mask].boundary.plot(
                ax=ax, linewidth=highlight_linewidth, color=highlight_color, zorder=highlight_zorder,
                linestyle="--"
            )

    # ======== New: index labels ========
    def _get_label_text(row, by, fmt):
        if callable(by):
            val = by(row)
        elif by == "position":
            val = row.name  # Row number
        else:
            col = by if by in row.index else None
            if col is None and by == "OBJECTID" and "OBJECTID" in row.index:
                col = "OBJECTID"
            if col is None:
                # Fallback to row number
                val = row.name
            else:
                val = row[col]
        return fmt(val) if (fmt is not None) else str(val)

    if label_indices:
        # Use representative_point() instead of centroid to better ensure point falls inside polygon
        reps = gdf_plot.representative_point()
        for idx, (geom, row) in enumerate(zip(reps, gdf_plot.itertuples(index=False))):
            # Note: itertuples(index=False) does not contain row number, here we use gdf_plot.index[idx]
            irow = gdf_plot.iloc[idx]
            text = _get_label_text(irow, label_by, label_fmt)
            x, y = geom.x, geom.y
            ax.text(x, y, text, ha="center", va="center",
                    fontsize=label_fontsize, color=label_color, zorder=5)

    # —— Custom discrete legend (maintain style)
    from matplotlib.patches import Patch
    labels = []
    for i in range(n_bins):
        lo = int(bins[i]); hi = int(bins[i+1])
        lab = f"< {hi:,}" if i == 0 else f"{lo:,} - {hi:,}"
        labels.append(lab)
    handles = [Patch(facecolor=listed_cmap(i), edgecolor='none') for i in range(n_bins)]
    # leg = ax.legend(handles, labels, title="Population", loc="upper right", frameon=True)
    leg = ax.legend(handles, labels, title="Population (persons)", loc="upper right", frameon=True)
    try:
        leg._legend_box.align = "left"
    except Exception:
        pass

    # —— North arrow & scale bar (maintain consistency)
    add_north(ax)
    add_scalebar(ax, gdf_plot, length_km=20, where=(0.35, 0.03), fontsize=16)

    # —— Axis style
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()

    if save_path:
        # plt.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.show()



# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Map data
    data = load_map(SHP_FILE)

    # 2) Population array (mapped to data index space via mapping)
    new_POP = load_population_and_remap(POP_FILE, MAP_CSV, n_target=len(data))

    # 3) Plot: bins are fully customizable (consistent with original "color axis mapping")
    # Example: similar to bins given in original script, you can change as needed
    bins = (0, 1.5e3, 3e3, 5e3, 8e3, 1.2e4, 2e4)
    bins = [int(x) *10 for x in bins]
    print("bins: ",  bins)

    # 1) Label each region with row number (position)
    # plot_population_map(
    #     data, new_POP, bins=bins,
    #     label_indices=True, label_by="position", label_fontsize=7
    # )

    # 2) Label OBJECTID, and highlight borders of regions with OBJECTID in [3, 7, 42]
    plot_population_map(
        data, new_POP, bins=bins,
        # label_indices=True,
        # highlight_indices=[335],
        highlight_linewidth=2, highlight_color="black",
        save_path="population_map_en.png"
    )

    # plot_population_map(
    #     data,
    #     new_POP,
    #     bins=bins,
    #     crs_proj="EPSG:3857",
    #     figsize=(12, 6),
    #     save_path="population_map.png"
    # )

"""
用于在地图上显示流量关系（重构版：更易读、模块化，功能不变）
"""
# === 标准库 / 第三方依赖 ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from shapely.geometry import LineString
from shapely.ops import unary_union
from pyproj import CRS, Geod

# === Matplotlib 全局样式 ===
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 15,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
})

# === 路径配置（保持与原脚本一致） ===
SHP_FILE = "../data/sz/Shenzhen_geo_data/Shenzhen_Community.shp"
POP_FILE = "../data/sz/community_654/population.npy"
OD_FILE  = "../data/sz/community_654/flow.npy"
MAP_CSV  = "../data/sz/community_654/mapping.csv"


# ---------------------------------------------------------------------
# 工具函数（绘制部件）
# ---------------------------------------------------------------------
def plot_city_outline(ax, gdf: gpd.GeoDataFrame, heal_tol=None, **line_kw):
    """
    仅绘制城市外轮廓（外环），避免把内部相邻边界加粗。
    heal_tol: 可选，单位为当前投影的长度单位（米/度），用于修补微小缝隙：
              先 buffer(+tol) 再 buffer(-tol)。
    line_kw: 传给 plot 的样式参数，如 color、linewidth、zorder。
    """
    geom = gdf.geometry
    if heal_tol and heal_tol > 0:
        geom = geom.buffer(heal_tol).buffer(-heal_tol)

    union = unary_union(geom).buffer(0)  # 并集并修正几何有效性

    outlines = []
    if union.geom_type == "Polygon":
        outlines.append(LineString(union.exterior.coords))
    elif union.geom_type == "MultiPolygon":
        for p in union.geoms:
            outlines.append(LineString(p.exterior.coords))
    else:
        return  # 不是面就不画

    gpd.GeoSeries(outlines, crs=gdf.crs).plot(ax=ax, **line_kw)


def add_north(ax, x=0.05, y=0.9, size=0.06):
    """
    绘制指北箭头（朝上）：箭头从 (x,y) 指向 (x,y+size)，坐标系为 axes fraction。
    """
    ax.annotate(
        'N',
        xy=(x, y + size),           # 箭头尖端（更高的位置）
        xytext=(x, y),              # 文本位置（更低的位置）
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
    在当前坐标系下添加比例尺（投影坐标按米；经纬度用大地线换算）。
    注意：为保持与原脚本一致，这里会覆盖 tick_fracs 为 9 等分（0,1/8,...,1.0）。

    - length_km: 比例尺总长度（千米）
    - where: 比例尺左端点在坐标轴的相对位置 (axes fraction)
    - tick_fracs: 刻度相对位置（会被覆盖为 9 等分以保持原功能）
    - tick_labels: 刻度文本，长度需与 tick_fracs 一致；None 时自动生成
    - unit_on_last_only: 仅在最后一个刻度文本加单位
    """
    # —— 保持原功能：强制使用 9 等分刻度 ——（若不想强制，可移除此行）
    tick_fracs = (0, 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8, 1.0)

    if tick_labels is None:
        labs = []
        for i, f in enumerate(tick_fracs):
            # 在 1/8、3/8、5/8、7/8 位置放空标签（保持原显示风格）
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
            raise ValueError("tick_labels 的长度必须与 tick_fracs 一致。")

    minx, miny, maxx, maxy = gdf.total_bounds
    x0 = minx + where[0] * (maxx - minx)
    y0 = miny + where[1] * (maxy - miny)

    dy = (maxy - miny)
    tick_h = 0.004 * dy
    text_off = 0.01 * dy

    crs = CRS.from_user_input(gdf.crs) if gdf.crs else None

    if crs is not None and crs.is_projected:
        # —— 投影坐标：按米 ——
        L = length_km * 1000.0
        x1 = x0 + L
        ax.plot([x0, x1], [y0, y0], color=color, lw=linewidth)

        for frac, lab in zip(tick_fracs, tick_labels):
            xx = x0 + L * float(frac)
            ax.plot([xx, xx], [y0 - tick_h, y0 + tick_h], color=color, lw=1)
            ax.text(xx, y0 - text_off, lab, ha="center", va="top", fontsize=fontsize)
    else:
        # —— 地理坐标（经纬度）：用大地线把 km -> 经度差 ——
        geod = Geod(ellps="WGS84")
        lon0, lat0 = float(x0), float(y0)

        # 主线（保持水平：方位角 90°）
        lon1, lat1, _ = geod.fwd(lon0, lat0, 90, length_km * 1000.0)
        ax.plot([lon0, lon1], [lat0, lat0], color=color, lw=linewidth)

        for frac, lab in zip(tick_fracs, tick_labels):
            loni, lati, _ = geod.fwd(lon0, lat0, 90, length_km * 1000.0 * float(frac))
            ax.plot([loni, loni], [lat0 - tick_h, lat0 + tick_h], color=color, lw=1)
            ax.text(loni, lat0 - text_off, lab, ha="center", va="top", fontsize=fontsize)

    ax.text(x0, y0 + 3 * tick_h, "Scale", ha="left", va="bottom", fontsize=fontsize)


# ---------------------------------------------------------------------
# 数据准备函数
# ---------------------------------------------------------------------
def load_map(shp_path: str) -> gpd.GeoDataFrame:
    """读取社区多边形并按 OBJECTID 排序。"""
    gdf = gpd.read_file(shp_path)
    gdf.sort_values(by="OBJECTID", inplace=True)
    print(len(gdf))
    return gdf


def load_flow_and_remap(pop_path: str, od_path: str, map_csv: str, n_target: int) -> np.ndarray:
    """
    读取人口与流量矩阵（flow），按 mapping.csv 将原索引映射到目标（shp）索引空间，
    生成与 gdf 行数一致的新 OD（new_OD）。功能保持与原脚本一致。
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
# 主绘图函数
# ---------------------------------------------------------------------
def plot_flow_map(
    gdf,                    # 社区多边形 GeoDataFrame（已按 OBJECTID 排序）
    OD,                     # 流量矩阵（已做过 *POP 的加权，并映射至 gdf 索引空间）
    bins=(0, 1e5, 2.5e5, 4.5e5, 8.5e5, 1.75e6),
    topk=None,              # 仅绘制流量最大的前 topk 条边；None 表示不用
    min_flow=None,          # 仅绘制 >= min_flow 的边；None 表示不用
    crs_proj="EPSG:3857",   # 投影坐标系（用于质心与比例尺；若已是米制可传 None）
    figsize=(12, 6),
    save_path=None
):
    assert OD.shape[0] == len(gdf), "OD 尺寸需与社区数一致"

    # —— 投影与质心 ——（保持原逻辑）
    if crs_proj is not None and (gdf.crs is None or gdf.crs.to_string() != crs_proj):
        gdf_plot = gdf.to_crs(crs_proj)
    else:
        gdf_plot = gdf

    cent = gdf_plot.geometry.centroid
    xs = cent.x.to_numpy()
    ys = cent.y.to_numpy()

    # —— 无向流量（i↔j 合并），仅取上三角 ——（保持原逻辑）
    flow = OD + OD.T
    iu = np.triu_indices_from(flow, k=1)
    pairs = np.stack([iu[0], iu[1]], axis=1)
    flows = flow[iu]

    # —— 按阈值 / TopK 过滤 ——（保持原逻辑）
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
        print("没有满足条件的连线可绘制。")
        return
    print(f"flows max: {flows.max()}, min: {flows.min()}")

    # —— 分箱与颜色/线宽 ——（保持原逻辑）
    bins = np.asarray(bins, dtype=float)
    bin_ids = np.digitize(flows, bins, right=True) - 1  # 0..len(bins)-2
    n_bins = len(bins) - 1
    cmap = plt.get_cmap("RdYlGn_r", n_bins)            # 低流量为绿，高流量为红
    colors = [cmap(i) for i in range(n_bins)]
    widths = np.linspace(0.7, 3.2, n_bins)             # 高流量更粗

    # —— 绘图准备 ——（边界、外轮廓）
    fig, ax = plt.subplots(figsize=figsize)
    # 社区边界
    gdf_plot.boundary.plot(ax=ax, linewidth=0.6, color="0.7", zorder=1)
    # 市域外轮廓（替代 dissolve().boundary，避免内部边界加粗）
    plot_city_outline(ax, gdf_plot, heal_tol=None, color="k", linewidth=1.2, zorder=2)

    # —— 批量绘制分箱线段集合（性能优于逐条 plot） ——（保持原逻辑）
    for b in range(n_bins):
        sel = (bin_ids == b)
        if not np.any(sel):
            continue
        segs = [[(xs[i], ys[i]), (xs[j], ys[j])] for i, j in pairs[sel]]
        lc = LineCollection(segs, colors=[colors[b]], linewidths=widths[b], alpha=0.9, zorder=3)
        ax.add_collection(lc)

    # —— 图例（分箱标签） ——（保持原逻辑）
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

    # —— 指北箭头 ——（保持原逻辑）
    add_north(ax)

    # —— 通用比例尺（任意 CRS 均可） ——（保持原调用）
    add_scalebar(ax, gdf_plot, length_km=20, where=(0.35, 0.03), fontsize=16)

    # —— 轴样式 ——（保持原逻辑）
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # 1) 地图数据
    data = load_map(SHP_FILE)

    # 2) 流动矩阵（按 mapping 映射到 data 的索引空间）
    new_OD = load_flow_and_remap(POP_FILE, OD_FILE, MAP_CSV, n_target=len(data))

    # 3) 绘图（参数与原脚本一致）
    for topk in [2000]:
        # topk = 800
        save_path = f"flow_map_{topk}_en.png"
        plot_flow_map(
            data,
            new_OD,
            bins=(0, 2e3, 3e3, 5e3, 8e3, 1.2e4, 2e4),
            # min_flow=1e5,          # 或者 topk=150
            topk=topk,
            crs_proj="EPSG:3857",
            figsize=(12, 6),
            save_path=save_path
        )

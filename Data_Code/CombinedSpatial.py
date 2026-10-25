"""
Combined trend maps for the Upper Blue Nile Basin.

Row 1 : station-level Sen's slopes (fill colour = slope, marker shape/edge = significance)
Row 2 : interpolated (spatially continuous) Sen's slopes clipped to the basin

All text styling (serif font, bold titles, font sizes) follows the interpolated-map script.

Requires: pandas, numpy, scipy, matplotlib, geopandas, shapely
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.colors import Normalize, TwoSlopeNorm, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import RBFInterpolator, griddata

# ----------------------------------------------------------------- paths ----
EVENTS = "C:/Users/YOGA/Desktop/Water/UBN_results!/tables/T4_trend_statistics_all"
COORDS = "C:/Users/YOGA/Desktop/Water/UBN_results!/tables/ElCoord.csv"   # columns: station, lon, lat
BASIN  = "D:/4thOBJ/UBN2.shp"                                           # Upper Blue Nile boundary
OUT    = "C:/Users/YOGA/Desktop/Water/CDHE_maps_combined.png"

# ------------------------------------------------------------- settings -----
# Row 1 (stations): (variable, panel title, unit, colormap)
VARS_ROW1 = [
    ("Tmin_mean", "Mean minimum temperature", "°C/decade",   "YlOrRd"),
    ("Tmax_mean", "Mean maximum temperature", "°C/decade",   "YlOrRd"),
    ("HW_days",   "Heatwave days",            "days/decade", "YlOrRd"),
    ("Wet_days",  "Wet days",                 "days/decade", "BrBG"),
]
# Row 2 (interpolated): (variable, panel title, unit, colormap)
VARS_ROW2 = [
    ("Tmin_mean", "Tmin trend",     "°C/decade",   "YlOrRd"),
    ("Tmax_mean", "Tmax trend",     "°C/decade",   "YlOrRd"),
    ("HW_days",   "Heatwave days",  "days/decade", "YlOrRd"),
    ("Wet_days",  "Wet days",       "days/decade", "RdYlBu"),
]
NCOLS = len(VARS_ROW1)

VALUE_COL = "slope_per_decade"
ALPHA_P   = 0.05
ALPHA_Q   = 0.10
USE_P_COL = "p_mmk"      # use "p" for the plain Mann-Kendall test
LABEL_STATIONS = False

METHOD    = "idw"        # interpolation: "idw" | "rbf" | "linear"
IDW_POWER = 2.0
N_LEVELS  = 10
GRID_N    = 400
DPI       = 300

# font sizes (same as the interpolated-map script)
FS_CBAR_TICKS = 20
FS_CBAR_LABEL = 22
FS_TITLE      = 22
FS_LEGEND     = 22
FS_STATION    = 12

# marker styles for row 1
STYLE = dict(
    ns  = dict(marker="v", edge="#7f7f7f", lw=1.2, s=130, z=3),   # not significant: down triangle
    sig = dict(marker="^", edge="#1f4eff", lw=2.2, s=170, z=4),   # significant: up triangle
    fdr = dict(marker="*", edge="#d6006e", lw=1.8, s=520, z=5),   # significant + FDR: star
)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.weight": "bold",          # all text bold (ticks, legend, annotations)
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})

# ------------------------------------------------------------------ data ----
ev_path = EVENTS if os.path.splitext(EVENTS)[1] else EVENTS + ".csv"
ev = pd.read_csv(ev_path)
co = pd.read_csv(COORDS)
ev["station"] = ev["station"].astype(str).str.strip()
co["station"] = co["station"].astype(str).str.strip()
ev["_key"], co["_key"] = ev["station"].str.lower(), co["station"].str.lower()

basin_rows = ev[ev["station"].str.upper() == "BASIN"].set_index("var")
st = ev[ev["station"].str.upper() != "BASIN"].merge(
    co[["_key", "lon", "lat"]].drop_duplicates("_key"), on="_key", how="left")
missing = sorted(st.loc[st["lon"].isna(), "station"].unique())
if missing:
    print("WARNING - no coordinates for:", missing)
st = st.dropna(subset=["lon", "lat"])

basin = gpd.read_file(BASIN)
if basin.crs is not None and basin.crs.to_epsg() != 4326:
    basin = basin.to_crs(epsg=4326)
geom = basin.geometry.union_all() if hasattr(basin.geometry, "union_all") \
    else basin.geometry.unary_union
minx, miny, maxx, maxy = geom.bounds
padx, pady = 0.04 * (maxx - minx), 0.04 * (maxy - miny)


def geom_to_path(g):
    """Shapely (Multi)Polygon -> matplotlib Path (holes included)."""
    polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
    verts, codes = [], []
    for p in polys:
        for ring in [p.exterior, *p.interiors]:
            xy = np.asarray(ring.coords)
            verts.append(xy)
            codes += [Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY]
    return Path(np.vstack(verts), codes)


basin_path = geom_to_path(geom)

gx = np.linspace(minx, maxx, GRID_N)
gy = np.linspace(miny, maxy, GRID_N)
GX, GY = np.meshgrid(gx, gy)


def interpolate(x, y, z):
    if METHOD == "idw":
        d = np.hypot(GX[..., None] - x, GY[..., None] - y)
        d = np.maximum(d, 1e-9)
        w = 1.0 / d ** IDW_POWER
        return (w * z).sum(-1) / w.sum(-1)
    if METHOD == "rbf":
        f = RBFInterpolator(np.c_[x, y], z, kernel="thin_plate_spline", smoothing=0.0)
        return f(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)
    if METHOD == "linear":
        zi = griddata((x, y), z, (GX, GY), method="linear")
        nn = griddata((x, y), z, (GX, GY), method="nearest")
        return np.where(np.isnan(zi), nn, zi)
    raise ValueError("METHOD must be 'idw', 'rbf' or 'linear'")


# ------------------------------------------------------------------ plot ----
fig, axes = plt.subplots(2, NCOLS, figsize=(6.2 * NCOLS, 13.0), squeeze=False)

# ---------------- row 1: station-level maps ----------------
for i, (var, title, unit, cmap) in enumerate(VARS_ROW1):
    ax = axes[0, i]
    d = st[st["var"] == var].copy()
    if d.empty:
        ax.set_visible(False)
        print(f"WARNING - variable '{var}' not found")
        continue

    vals = d[VALUE_COL].to_numpy()
    vmax = np.nanmax(np.abs(vals))
    if np.nanmin(vals) >= 0:
        norm = Normalize(vmin=0, vmax=vmax)
    else:
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    basin.plot(ax=ax, facecolor="#f2f2f2", edgecolor="black", linewidth=1.0, zorder=1)

    sig = d[USE_P_COL] < ALPHA_P
    fdr = d["q_fdr"] < ALPHA_Q
    for mask, sty in [(~sig, STYLE["ns"]), (sig & ~fdr, STYLE["sig"]), (sig & fdr, STYLE["fdr"])]:
        g = d[mask]
        if len(g):
            ax.scatter(g["lon"], g["lat"], c=g[VALUE_COL], cmap=cmap, norm=norm,
                       marker=sty["marker"], s=sty["s"], edgecolors=sty["edge"],
                       linewidths=sty["lw"], zorder=sty["z"])

    if LABEL_STATIONS:
        for _, r in d.iterrows():
            ax.annotate(r["station"], (r["lon"], r["lat"]), fontsize=FS_STATION, fontweight="bold",
                        xytext=(3, 3), textcoords="offset points", zorder=6)

    sub = ""
    if var in basin_rows.index:
        b = basin_rows.loc[var]
        sub = f"Basin: {b[VALUE_COL]:+.2f} {unit}"
    ax.set_title(f"({chr(97 + i)}) {title}\n{sub}", loc="left",
                 fontsize=FS_TITLE, fontweight="bold")

    ax.set_xlim(minx - padx, maxx + padx)
    ax.set_ylim(miny - pady, maxy + pady)
    ax.set_aspect("equal")
    ax.axis("off")

    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      shrink=0.85, pad=0.02, aspect=22)
    cb.ax.tick_params(labelsize=FS_CBAR_TICKS)
    for t in cb.ax.get_yticklabels():
        t.set_fontweight("bold")
    cb.ax.yaxis.set_major_locator(MaxNLocator(5))
    cb.set_label(unit, rotation=270, labelpad=24, fontsize=FS_CBAR_LABEL, fontweight="bold")

# ---------------- row 2: interpolated maps ----------------
for i, (var, title, unit, cmap) in enumerate(VARS_ROW2):
    ax = axes[1, i]
    d = st[st["var"] == var]
    if d.empty:
        ax.set_visible(False)
        continue

    x, y, z = d["lon"].to_numpy(), d["lat"].to_numpy(), d[VALUE_COL].to_numpy()
    Z = interpolate(x, y, z)

    lo, hi = np.nanmin(Z), np.nanmax(Z)
    if lo < 0 < hi:
        half = MaxNLocator(nbins=max(N_LEVELS // 2, 1)).tick_values(0, max(abs(lo), abs(hi)))
        levels = np.unique(np.r_[-half, half])
    else:
        levels = MaxNLocator(nbins=N_LEVELS).tick_values(lo, hi)
    norm = BoundaryNorm(levels, ncolors=plt.get_cmap(cmap).N, extend="neither")

    cs = ax.contourf(GX, GY, Z, levels=levels, cmap=cmap, norm=norm)

    patch = PathPatch(basin_path, transform=ax.transData, fc="none", ec="none")
    ax.add_patch(patch)
    if hasattr(cs, "set_clip_path"):
        cs.set_clip_path(patch)
    else:                                              # matplotlib < 3.8
        for c in cs.collections:
            c.set_clip_path(patch)
    ax.add_patch(PathPatch(basin_path, fc="none", ec="black", lw=0.9, zorder=5))

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"({chr(97 + NCOLS + i)}) {title}", loc="left",
                 fontsize=FS_TITLE, fontweight="bold")

    cb = fig.colorbar(cs, ax=ax, ticks=levels, shrink=0.85, pad=0.02, aspect=22)
    cb.ax.tick_params(labelsize=FS_CBAR_TICKS)
    for t in cb.ax.get_yticklabels():
        t.set_fontweight("bold")
    cb.set_label(unit, rotation=270, labelpad=24, fontsize=FS_CBAR_LABEL, fontweight="bold")
    cb.ax.set_yticklabels([f"{t + 0.0:.3g}" for t in levels], fontweight="bold")

# ---------------- shared legend (row 1 symbols) ----------------
def _h(sty, label, ms):
    return Line2D([], [], marker=sty["marker"], ls="", mfc="white",
                  mec=sty["edge"], mew=sty["lw"], ms=ms, label=label)

handles = [
    _h(STYLE["ns"],  f"Not significant ({USE_P_COL} ≥ {ALPHA_P})", 14),
    _h(STYLE["sig"], f"Significant ({USE_P_COL} < {ALPHA_P})",     15),
    _h(STYLE["fdr"], f"Significant and FDR q < {ALPHA_Q}",          24),
]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           prop={"size": FS_LEGEND, "weight": "bold"}, handletextpad=0.5, columnspacing=2.0)
fig.tight_layout(rect=[0, 0.05, 1, 1])

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=DPI, bbox_inches="tight", facecolor="white")
print("Saved:", OUT)
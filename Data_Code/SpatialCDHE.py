import numpy as np, pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from shapely import contains_xy
from scipy.interpolate import RBFInterpolator
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------- INPUTS (edit paths / column names) ----------------
EVENTS = "C:/Users/YOGA/Desktop/Water/UBN_results/tables/T5a_CDHE_events.csv"
COORDS = "C:/Users/YOGA/Desktop/Water/UBN_results/tables/ElCoord.csv"   # needs columns: station, lon, lat
BASIN  = "D:/4thOBJ/UBN2.shp"                                           # Upper Blue Nile boundary
OUT    = "C:/Users/YOGA/Desktop/Water/CDHE_maps_1row.png"

METHOD     = "idw"  # "idw" (classic) or "rbf" (smoother, no bullseyes)
IDW_POWER  = 2      # 1.5-3 typical; higher = more local
RBF_SMOOTH = 0.5    # only for METHOD = "rbf"
GRID_N     = 400    # grid resolution
N_CLASSES  = 10     # number of discrete colour classes

# ---- Typography (journal style) ----
TITLE_SIZE = 36     # panel titles (a), (b), ...
CBAR_TICK  = 32     # colourbar tick labels
CBAR_LABEL = 32     # colourbar unit labels
# --------------------------------------------------------------------

# All text in Times New Roman (falls back to similar serif fonts if not installed)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 14,
    "axes.titleweight": "bold",
})

# 1) Station-level metrics
ev = pd.read_csv(EVENTS)
st = ev.groupby("station").agg(
    frequency=("year", "size"),          # number of events
    duration=("duration", "sum"),        # total event-months
    severity=("severity", "sum"),        # cumulative severity
).reset_index()
st["intensity"] = st["severity"] / st["duration"]   # severity per event-month

xy = pd.read_csv(COORDS)
st = st.merge(xy[["station", "lon", "lat"]], on="station", how="left")
missing = st[st.lon.isna()].station.tolist()
if missing:
    print("No coordinates for:", missing)
st = st.dropna(subset=["lon", "lat"])

basin = gpd.read_file(BASIN).to_crs(4326)
minx, miny, maxx, maxy = basin.total_bounds

# 2) Grid + basin mask
gx = np.linspace(minx, maxx, GRID_N)
gy = np.linspace(miny, maxy, GRID_N)
GX, GY = np.meshgrid(gx, gy)
geom = basin.geometry.union_all() if hasattr(basin.geometry, "union_all") else basin.unary_union
inside = contains_xy(geom, GX, GY)

def idw(x, y, v, GX, GY, p=2):
    d = np.hypot(GX[..., None] - x, GY[..., None] - y)
    d = np.maximum(d, 1e-9)
    w = 1.0 / d**p
    return (w * v).sum(-1) / w.sum(-1)

def rbf(x, y, v, GX, GY, smooth=0.5):
    f = RBFInterpolator(np.c_[x, y], v, kernel="thin_plate_spline", smoothing=smooth)
    return f(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)

def interpolate(col):
    x, y, v = st.lon.values, st.lat.values, st[col].values
    Z = idw(x, y, v, GX, GY, IDW_POWER) if METHOD == "idw" else rbf(x, y, v, GX, GY, RBF_SMOOTH)
    Z = np.clip(Z, v.min(), v.max())
    return np.where(inside, Z, np.nan)

# 3) Panel settings: title, column, colormap, colourbar label
panels = [
    ("(a) Frequency", "frequency", "RdYlBu_r", "events"),
    ("(b) Duration",  "duration",  "RdYlBu_r", "months"),
    ("(c) Severity",  "severity",  "RdYlBu_r", "index"),
    ("(d) Intensity", "intensity", "RdYlBu_r", "index/month"),
]

fig, axes = plt.subplots(1, 4, figsize=(32, 16))
for ax, (title, col, cmap, unit) in zip(axes.ravel(), panels):
    Z = interpolate(col)

    vmin, vmax = st[col].min(), st[col].max()
    levels = np.linspace(vmin, vmax, N_CLASSES + 1)
    norm = BoundaryNorm(levels, plt.get_cmap(cmap).N)

    basin.boundary.plot(ax=ax, color="k", lw=1.4)
    pm = ax.pcolormesh(GX, GY, Z, cmap=cmap, norm=norm, shading="auto")
    # (station points intentionally not drawn)

    ax.set_aspect("equal")
    divider = make_axes_locatable(ax)                       # colourbar height = map height
    cax = divider.append_axes("right", size="5%", pad=0.12)
    cb = fig.colorbar(pm, cax=cax)
    cb.ax.tick_params(labelsize=CBAR_TICK, length=9, width=2)
    cb.set_ticks(levels[::2])
    cb.ax.set_yticklabels([f"{t:.2f}" if col == "intensity" else f"{t:.0f}"
                           for t in levels[::2]])
    cb.set_label(unit, fontsize=CBAR_LABEL, fontweight="bold")
    for t in cb.ax.get_yticklabels():
        t.set_fontweight("bold")
    cb.outline.set_linewidth(1.5)
    ax.set_title(title, loc="left", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_axis_off()

fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.03, wspace=0.18)
plt.savefig(OUT, dpi=600, bbox_inches="tight")
print("saved", OUT)
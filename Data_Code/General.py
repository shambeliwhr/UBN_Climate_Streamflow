#!/usr/bin/env python3
"""
Upper Blue Nile Basin - compound drought-heat extremes, elevation(thermal)-dependence and explainable machine learning
=====================================================================================================================
ONE COMBINED SCRIPT (Parts 1-4). Reproduces every table (tables/*.csv), figure (figures/*.png) and number in the report.

Inputs  : one daily station CSV per station in
              C:/Users/YOGA/Desktop/Water/NASA_POWER_Selected2   (set by DATA in PART 1, or env var UBN_DATA), columns
          Date, Rain_mm, Tmax_C, Tmin_C, RH_%, Wind_m_s, Solar*   (-999 = missing)
          Optional: nino34.csv and dmi.csv (columns Date,Value) in the same folder -> teleconnection correlations.
          Optional: basin boundary (.shp or .geojson) in the same folder, or env var UBN_BOUNDARY -> clips the interpolated maps of Fig 18 (needs geopandas).
Outputs : C:/Users/YOGA/Desktop/Water/UBN_results  (tables/, figures/)  - set by OUT in PART 1, or env var UBN_OUT
Run     : python ubn_analysis_combined.py            (about 4-6 minutes on one CPU core)
Needs   : numpy pandas scipy scikit-learn matplotlib seaborn   (xgboost / lightgbm / catboost are used automatically if installed; geopandas optional)

PART 1  data loading, QC, duplicate-station check, ETCCDI-type indices, SPI/SPEI, compound drought-heat events (CDHE) and severity (CDHI)
PART 2  MK / Hamed-Rao / Sen + FDR trends, homogeneity & change points, copulas, thermal-zone dependence, EOF/clusters, wavelets, sensitivity
PART 3  ML: classification (occurrence) + regression (severity), time-blocked validation, tuning, robustness, lead time, spatial CV,
        Shapley attribution (permutation-sampling estimator), ALE, Friedman-H interactions, threshold detection, zone-specific models
PART 4  all figures (Fig 18 a-d are now interpolated surfaces clipped to the basin)
ELEV    Station longitude / latitude / elevation come from ElCoord.csv in the same data folder (columns: station, longitude, latitude, elevation).
        Elevation defines the agro-ecological zones (Kolla < 1500 m, Weyna Dega 1500-2300 m, Dega > 2300 m); latitude drives PET (Hargreaves).
"""
# %% ===================== PART 1: DATA, QC, INDICES, DROUGHT, COMPOUND EVENTS =====================
import os, sys, glob, warnings, pickle, json, itertools
import numpy as np, pandas as pd
from scipy import stats, special, optimize
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
# ---------------------------------------------------------------- PATHS (edit here if your folder changes)
# Folder holding the daily station CSVs (the folder name 'NASA_POWER_Selected2' is historical: the files are station observations) (Adet.csv, Ambo.csv, ..., Shmbu.csv). Override with the environment variable UBN_DATA.
DATA = os.environ.get('UBN_DATA', r'C:/Users/YOGA/Desktop/Water/NASA_POWER_Selected2')
SANDBOX = False
if not os.path.isdir(DATA) and os.path.isdir('/mnt/user-data/uploads'):      # fallback used only inside the Claude sandbox
    DATA, SANDBOX = '/mnt/user-data/uploads', True
# Results are written to a folder "UBN_results" next to the data folder (override with UBN_OUT).
OUT = os.environ.get('UBN_OUT', '/mnt/user-data/outputs/UBN_results!!' if SANDBOX else os.path.join(os.path.dirname(os.path.normpath(DATA)), 'UBN_results!!'))
FIG, TAB = os.path.join(OUT, 'figures'), os.path.join(OUT, 'tables')
CACHE = os.environ.get('UBN_CACHE', os.path.join(OUT, '_cache'))
for p in (FIG, TAB, CACHE): os.makedirs(p, exist_ok=True)
print('Reading station data from :', DATA); print('Writing results to           :', OUT)
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
BASE = (1981, 2010)      # reference (baseline) period for percentiles / SPI / SPEI fitting
LAT = 10.5               # basin-mean latitude used ONLY for extraterrestrial radiation in Hargreaves PET
DROUGHT, HEAT_Q, SEED = -1.0, 0.90, 42
rng = np.random.default_rng(SEED)
plt.rcParams.update({'figure.dpi':110, 'savefig.dpi':200, 'axes.spines.top':False, 'axes.spines.right':False,
    'font.family':'serif', 'font.serif':['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'Times', 'DejaVu Serif'],   # Times New Roman (metric-compatible fallbacks on Linux)
    'mathtext.fontset':'stix', 'axes.unicode_minus':False,
    'font.weight':'bold', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold',
    'font.size':13, 'axes.titlesize':14, 'axes.titlepad':10, 'axes.labelsize':13.5, 'xtick.labelsize':12, 'ytick.labelsize':12, 'legend.fontsize':11.5, 'legend.title_fontsize':12, 'figure.titlesize':15})
AUTO_LABELS = [   # (panel-title prefix, x label, y label) - applied only where a label is missing
    ('Moisture balance', None, 'Station'), ('Daily Tmax correlation', 'Station file', 'Station file'), ('SNHT against', None, 'Station'),
    ('Timing of detected shifts', None, 'Series'), ('Compound frequency vs. independence', 'Station (ordered high → low elevation)', None),
    ('1981–2003 (blue)', 'Hydroclimatic state', None), ('One episode dominates', 'Hydroclimatic state', None),
    ('Compound months / yr', 'Year', 'Months per year'), ('Hot months / yr', 'Year', 'Months per year'), ('Drought months / yr', 'Year', 'Months per year'),
    ('Tmax (°C)', 'Year', 'Tmax (°C)'), ('Rainfall (mm)', 'Year', 'Rainfall (mm/yr)'), ('Global spectrum', 'Global wavelet power', 'Period (years)'),
    ('EOF1 of SPEI-3', 'Station', 'EOF1 loading'), ('Hierarchical clusters', 'Station', 'Ward linkage distance'), ('PC1 of SPEI-3', 'Year', 'PC1 score'),
    ('Spatial extent', 'Year', None), ('Test skill', 'Model', 'Score (independent test)'), ('Feature-set robustness', None, 'Feature-set experiment'),
    ('Global importance', None, 'Predictor'), ('Beeswarm', None, 'Predictor'), ('Pairwise interaction', 'Predictor', 'Predictor'),
    ('Compound drought–heat months per year', 'Year', 'Months / yr')]
FIGSZ = {}                                                    # figure name -> (width, height) in inches; widths of 8.5-10 in keep text readable when scaled to a 6.5-in page column
MIN_FONT = 10.5
def _fit_text(fig_):
    import textwrap
    from matplotlib.text import Text
    _skip = {id(t_) for a_ in fig_.axes if len(a_.texts) > 60 for t_ in a_.texts}          # dense cell annotations of heatmaps keep their own size
    for t_ in fig_.findobj(Text):
        if id(t_) not in _skip and t_.get_text().strip() and t_.get_fontsize() < MIN_FONT: t_.set_fontsize(MIN_FONT)
    for a_ in fig_.axes:
        if a_.get_label() == '<colorbar>' or not a_.axison: continue
        ttl = a_.get_title()
        if ttl and '\n' not in ttl:
            w_in = a_.get_position().width*fig_.get_size_inches()[0]; n = max(18, int(w_in*72/(a_.title.get_fontsize()*0.55)))
            if len(ttl) > n: a_.set_title(textwrap.fill(ttl.replace('\n', ' '), n))
def pf_(p): return '<0.001' if p < 0.001 else f'{p:.3f}' if p < 0.01 else f'{p:.2f}'
def savefig(name):
    fig_ = plt.gcf()
    if name in FIGSZ: fig_.set_size_inches(*FIGSZ[name])
    fig_.canvas.draw()                                        # draw once so that tick labels exist
    for _a in fig_.axes:
        if _a.get_label() == '<colorbar>' or not _a.axison: continue
        for key, xl, yl in AUTO_LABELS:
            if _a.get_title().startswith(key):
                if xl and not _a.get_xlabel(): _a.set_xlabel(xl)
                if yl and not _a.get_ylabel(): _a.set_ylabel(yl)
        if name.startswith('Fig08') and not _a.get_title() and not _a.get_ylabel(): _a.set_ylabel('Variable')
        if name.startswith('Fig05') and not _a.get_title() and not _a.get_xlabel() and not _a.get_ylabel(): _a.set_xlabel('Year'); _a.set_ylabel('Station')
        if not _a.get_xlabel():                                    # any remaining axis whose tick labels are calendar years
            _tx = [l.get_text() for l in _a.get_xticklabels() if l.get_text()]
            if _tx and all(t_.strip().isdigit() and 1900 <= int(t_) <= 2100 for t_ in _tx): _a.set_xlabel('Year')
        _m = [n for n, t in (('x', _a.get_xlabel()), ('y', _a.get_ylabel())) if not t and any(l.get_text() for l in (_a.get_xticklabels() if n == 'x' else _a.get_yticklabels()))]
        if _m: print(f'  [label check] {name}: axis without title ->', _m, '| panel title:', _a.get_title()[:50])
    _fit_text(fig_); plt.tight_layout(); plt.savefig(f'{FIG}/{name}.png', bbox_inches='tight'); plt.close()
def save_state(S): pickle.dump(S, open(os.path.join(CACHE, 'state.pkl'),'wb'))
S = {}

# ---------------------------------------------------------------- 1. load
import re
def _norm(n): return re.sub(r'[^a-z0-9]', '', str(n).lower())
raw, skipped = {}, []
for f in sorted(glob.glob(os.path.join(DATA, '*.csv'))):
    k = os.path.basename(f)[:-4]
    if k.lower() in ('nino34', 'dmi', 'nao', 'elcoord'): continue          # index files and the coordinate table are not stations
    x = pd.read_csv(f)
    if 'Date' not in x.columns or 'Rain_mm' not in x.columns: skipped.append(k); continue
    try:    x['Date'] = pd.to_datetime(x['Date'], format='%m/%d/%Y')
    except Exception: x['Date'] = pd.to_datetime(x['Date'], format='mixed')
    x = x.set_index('Date').sort_index()
    x.columns = ['Solar' if c.startswith('Solar') else c for c in x.columns]
    raw[k] = x.replace(-999, np.nan)
if skipped: print('Skipped (not station data files):', skipped)

# ---------------------------------------------------------------- 1b. station coordinates and elevation (ElCoord.csv in the data folder)
COORD_FILE = os.path.join(DATA, 'ElCoord.csv')
if not os.path.exists(COORD_FILE): raise FileNotFoundError(f'ElCoord.csv (station, longitude, latitude, elevation) not found in {DATA}')
_c = pd.read_csv(COORD_FILE, dtype=str, encoding='utf-8-sig'); _c.columns = [c.strip().lower() for c in _c.columns]
_col = lambda pre: next(c for c in _c.columns if c.startswith(pre))                   # tolerant to typos such as 'elevatation'
_num = lambda c: pd.to_numeric(_c[_col(c)].str.replace(',', '').str.strip(), errors='coerce')
COORD_ALL = pd.DataFrame({'name': _c[_col('station')].str.strip(), 'lon': _num('lon'), 'lat': _num('lat'), 'elev': _num('elev')}); COORD_ALL['key'] = COORD_ALL.name.map(_norm)
def _match(name):
    n = _norm(name); hit = COORD_ALL[COORD_ALL.key == n]
    if len(hit) == 0 and len(n) >= 4: hit = COORD_ALL[[(k.startswith(n) or n.startswith(k)) for k in COORD_ALL.key]]      # e.g. DegoT -> DegoTsion
    if len(hit) == 0:                                                                                                      # e.g. Mekanslm -> Mekaneselm (spelling differences)
        import difflib
        cand = difflib.get_close_matches(n, list(COORD_ALL.key), n=1, cutoff=0.72)
        if cand: hit = COORD_ALL[COORD_ALL.key == cand[0]]; print(f'  [name match] {name} -> {hit.iloc[0]["name"]} (approximate match; please check)')
    return hit.iloc[0] if len(hit) else None
_rows, _missing = {}, []
for k in raw:
    m = _match(k)
    if m is None or not np.isfinite(m.elev): _missing.append(k)
    else: _rows[k] = dict(lon=m.lon, lat=m.lat, elev=m.elev, coord_name=m['name'])
if _missing: raise ValueError(f'No coordinates/elevation in ElCoord.csv for: {_missing}. Check the station names in ElCoord.csv.')
COORD = pd.DataFrame(_rows).T.astype({'lon': float, 'lat': float, 'elev': float})
print(f'Coordinates matched for {len(COORD)} data files; ElCoord rows without a data file:', sorted(set(COORD_ALL.name) - set(COORD.coord_name)))
full_idx = pd.date_range('1981-01-01', '2025-12-31', freq='D')

# ---------------------------------------------------------------- 2. quality control
def longest_run_identical(v):
    v = np.asarray(v); best = cur = 1
    for i in range(1, len(v)):
        cur = cur+1 if v[i]==v[i-1] else 1; best = max(best, cur)
    return best
qc = []
for k, x in raw.items():
    qc.append(dict(station=k, n_days=len(x), complete_calendar=bool(x.index.equals(full_idx)), duplicated_dates=int(x.index.duplicated().sum()),
        missing_rain=int(x.Rain_mm.isna().sum()), missing_tmax=int(x.Tmax_C.isna().sum()), missing_tmin=int(x.Tmin_C.isna().sum()),
        solar_missing_pct=round(100*x.Solar.isna().mean(),1), rain_neg=int((x.Rain_mm<0).sum()), rain_gt200=int((x.Rain_mm>200).sum()),
        tmax_lt_tmin=int((x.Tmax_C<x.Tmin_C).sum()), tmax_gt45=int((x.Tmax_C>45).sum()), tmin_lt_neg5=int((x.Tmin_C<-5).sum()),
        rh_out_of_range=int(((x['RH_%']<0)|(x['RH_%']>100)).sum()), rain_max_mm=x.Rain_mm.max(),
        wetday_pct_ge1mm=round(100*(x.Rain_mm>=1).mean(),1), drizzle_pct_0_1mm=round(100*((x.Rain_mm>0)&(x.Rain_mm<1)).mean(),1),
        longest_identical_tmax_run=longest_run_identical(x.Tmax_C.values)))
qc = pd.DataFrame(qc).set_index('station'); qc.to_csv(f'{TAB}/T1_quality_control.csv')

# duplicate-series detection
names = list(raw); dup_pairs = []
for a, b in itertools.combinations(names, 2):
    xa, xb = raw[a], raw[b]
    if (np.isclose(xa.Rain_mm.values, xb.Rain_mm.values).mean() > .99) and (np.isclose(xa.Tmax_C.values, xb.Tmax_C.values).mean() > .99):
        dup_pairs.append((a, b))
parent = {n: n for n in names}
def _find(a):
    while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
    return a
for a_, b_ in dup_pairs: parent[_find(b_)] = _find(a_)
_g = {}
for n in names: _g.setdefault(_find(n), []).append(n)
dup_groups = [sorted(g) for g in _g.values() if len(g) > 1]              # stations whose records are identical (copied or derived from one source series)
rep_of = {}
for g in dup_groups:
    r = 'BahirDar' if 'BahirDar' in g else g[0]
    for n in g: rep_of[n] = r
drop = [n for n in rep_of if rep_of[n] != n]
stations = [s_ for s_ in names if s_ not in drop]
pass
def _km(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (COORD.lat[a], COORD.lon[a], COORD.lat[b], COORD.lon[b])); h = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2*6371*np.arcsin(np.sqrt(h))
_tab = pd.DataFrame([dict(group=i+1, station=n, representative=rep_of[n], kept=(n == rep_of[n]), lon=COORD.lon[n], lat=COORD.lat[n], elev_m=COORD.elev[n], max_distance_in_group_km=max(_km(a_, b_) for a_ in g for b_ in g)) for i, g in enumerate(dup_groups) for n in g])
_tab.to_csv(f'{TAB}/T1b_identical_series_groups.csv', index=False)
COORD_ORIG = COORD.copy(); group_members = {}
for g in dup_groups:                                  # one series represents all members of a group: use the group-mean position and elevation
    r_ = rep_of[g[0]]; group_members[r_] = g; COORD.loc[r_, ['lon', 'lat', 'elev']] = COORD_ORIG.loc[g, ['lon', 'lat', 'elev']].mean().values
S['COORD_ORIG'], S['group_members'] = COORD_ORIG, group_members
S['dup_pairs'], S['dropped'], S['stations'] = dup_pairs, drop, stations
S['dup_groups'], S['COORD'], S['COORD_ALL'], S['rep_of'] = dup_groups, COORD, COORD_ALL, rep_of
print('Identical-series groups (duplicated records):', dup_groups, '-> analysed independent series:', len(stations))

# ---------------------------------------------------------------- 3. daily derived variables
def ra_mm(doy, lat=LAT):
    phi = np.deg2rad(lat); dr = 1+0.033*np.cos(2*np.pi*doy/365); dl = 0.409*np.sin(2*np.pi*doy/365-1.39)
    ws = np.arccos(-np.tan(phi)*np.tan(dl))
    Ra = 24*60/np.pi*0.0820*dr*(ws*np.sin(phi)*np.sin(dl)+np.cos(phi)*np.cos(dl)*np.sin(ws))
    return 0.408*Ra                                  # mm/day equivalent
daily = {}
for k in stations:
    x = raw[k].copy()
    x['Tmean'] = (x.Tmax_C+x.Tmin_C)/2; x['DTR'] = x.Tmax_C-x.Tmin_C
    x['PET'] = 0.0023*ra_mm(x.index.dayofyear.values, COORD.lat[k])*(x.Tmean+17.8)*np.sqrt(x.DTR.clip(lower=0))   # Hargreaves-Samani
    daily[k] = x
def doy_percentile(s, q, base=BASE, win=2):
    b = s[(s.index.year>=base[0])&(s.index.year<=base[1])]; doy = b.index.dayofyear.values; v = b.values
    out = np.full(367, np.nan)
    for d in range(1, 367):
        df_ = np.abs(doy-d); df_ = np.minimum(df_, 366-df_); out[d] = np.nanpercentile(v[df_<=win], q*100)
    return out
def runs(mask):
    """return list of (start_idx, length) of True runs"""
    m = np.asarray(mask, int); d = np.diff(np.r_[0, m, 0]); st = np.where(d==1)[0]; en = np.where(d==-1)[0]
    return list(zip(st, en-st))

# ---------------------------------------------------------------- 4. annual ETCCDI-type indices
years = np.arange(1981, 2026); ann = {}
def put(var, k, ser): ann.setdefault(var, {}).setdefault(k, []).append(ser)
for k in stations:
    x = daily[k]; P, Tx, Tn = x.Rain_mm, x.Tmax_C, x.Tmin_C
    thx90 = doy_percentile(Tx, .9); thn90 = doy_percentile(Tn, .9); thx95 = doy_percentile(Tx, .95)
    hot = pd.Series(Tx.values > thx90[Tx.index.dayofyear], index=x.index)
    hot95 = pd.Series(Tx.values > thx95[Tx.index.dayofyear], index=x.index)
    warmn = pd.Series(Tn.values > thn90[Tn.index.dayofyear], index=x.index)
    wet = P >= 1; p95 = np.percentile(P[(P>=1)&(P.index.year>=BASE[0])&(P.index.year<=BASE[1])], 95)
    for y, g in x.groupby(x.index.year):
        gh, gw, g95 = hot[g.index], warmn[g.index], hot95[g.index]
        hw = [l for s_, l in runs(gh.values) if l >= 3]; ws = [l for s_, l in runs(gh.values) if l >= 6]
        gp = g.Rain_mm; gwet = gp >= 1; wr = runs(gwet.values); dr = runs(~gwet.values)
        mm = gp.groupby(gp.index.month).sum()
        put('PRCP_total', k, (y, gp.sum())); put('Wet_days', k, (y, gwet.sum()))
        put('R10mm', k, (y, (gp>=10).sum())); put('R20mm', k, (y, (gp>=20).sum()))
        put('Rx1day', k, (y, gp.max())); put('Rx5day', k, (y, gp.rolling(5).sum().max()))
        put('SDII', k, (y, gp[gwet].sum()/max(gwet.sum(),1)))
        put('CDD', k, (y, max([l for _, l in dr], default=0))); put('CWD', k, (y, max([l for _, l in wr], default=0)))
        put('R95pTOT_pct', k, (y, 100*gp[gp>p95].sum()/max(gp[gwet].sum(),1e-9)))
        put('PCI', k, (y, 100*(mm**2).sum()/max(mm.sum()**2,1e-9)))
        put('Kiremt_share_pct', k, (y, 100*mm.reindex(range(6,10)).sum()/max(mm.sum(),1e-9)))
        put('Tmax_mean', k, (y, g.Tmax_C.mean())); put('Tmin_mean', k, (y, g.Tmin_C.mean())); put('Tmean', k, (y, g.Tmean.mean()))
        put('DTR', k, (y, g.DTR.mean())); put('TXx', k, (y, g.Tmax_C.max())); put('TNn', k, (y, g.Tmin_C.min()))
        put('TX90p', k, (y, 100*gh.mean())); put('TN90p', k, (y, 100*gw.mean())); put('TX95p', k, (y, 100*g95.mean()))
        put('HW_days', k, (y, sum(hw))); put('HW_events', k, (y, len(hw))); put('WSDI', k, (y, sum(ws)))
        put('PET_total', k, (y, g.PET.sum()))
ann = {v: pd.DataFrame({k: pd.Series(dict(s)) for k, s in d.items()}) for v, d in ann.items()}
S['ann'] = ann

# ---------------------------------------------------------------- 5. monthly matrices
def mon(col, how):
    return pd.DataFrame({k: getattr(daily[k][col].resample('MS'), how)() for k in stations})
Pm, Txm, Tnm, Tmm = mon('Rain_mm','sum'), mon('Tmax_C','mean'), mon('Tmin_C','mean'), mon('Tmean','mean')
RHm, Wm, PETm, DTRm = mon('RH_%','mean'), mon('Wind_m_s','mean'), mon('PET','sum'), mon('DTR','mean')
S.update(Pm=Pm, Txm=Txm, Tnm=Tnm, Tmm=Tmm, RHm=RHm, Wm=Wm, PETm=PETm, DTRm=DTRm)

# ---------------------------------------------------------------- 6. SPI (gamma) and SPEI (log-logistic, PWM) on baseline
def _cal_apply(A, fitfun, cdf_fun, base=BASE):
    out = pd.Series(np.nan, index=A.index)
    for m in range(1, 13):
        a = A[A.index.month==m]; b = a[(a.index.year>=base[0])&(a.index.year<=base[1])].dropna()
        params = fitfun(b.values); p = cdf_fun(a.values, params)
        out[a.index] = stats.norm.ppf(np.clip(p, 1e-4, 1-1e-4))
    return out
def spi_fit(b):
    q = (b<=0).mean(); bp = b[b>0]; a, loc, sc = stats.gamma.fit(bp, floc=0); return q, a, sc
def spi_cdf(x, p):
    q, a, sc = p; return q+(1-q)*stats.gamma.cdf(np.where(x<0, 0, x), a, loc=0, scale=sc)
def ll_fit(b):
    x = np.sort(b); n = len(x); F = (np.arange(1, n+1)-0.35)/n
    w0, w1, w2 = [np.mean((1-F)**s*x) for s in (0, 1, 2)]
    beta = (2*w1-w0)/(6*w1-w0-6*w2)
    if not np.isfinite(beta) or beta <= 1.05: return ('norm', x.mean(), x.std(ddof=1))
    g = special.gamma(1+1/beta)*special.gamma(1-1/beta); alpha = (w0-2*w1)*beta/g; gam = w0-alpha*g
    return ('ll', alpha, beta, gam)
def ll_cdf(x, p):
    if p[0]=='norm': return stats.norm.cdf(x, p[1], p[2])
    _, al, be, ga = p; z = np.where(x-ga<=1e-9, 1e-9, x-ga); return 1/(1+(al/z)**be)
def SPI(P, k):  return pd.DataFrame({s: _cal_apply(P[s].rolling(k, min_periods=k).sum(), spi_fit, spi_cdf) for s in P})
def SPEI(P, PET, k):
    D = P-PET; return pd.DataFrame({s: _cal_apply(D[s].rolling(k, min_periods=k).sum(), ll_fit, ll_cdf) for s in P})
spi1, spi3, spi6, spi12 = [SPI(Pm, k) for k in (1, 3, 6, 12)]
spei1, spei3, spei6, spei12 = [SPEI(Pm, PETm, k) for k in (1, 3, 6, 12)]
S.update(spi1=spi1, spi3=spi3, spi6=spi6, spi12=spi12, spei1=spei1, spei3=spei3, spei6=spei6, spei12=spei12)

# ---------------------------------------------------------------- 7. monthly heat, compound drought-heat events (CDHE), severity index (CDHI)
def month_stats(M, base=BASE):
    b = M[(M.index.year>=base[0])&(M.index.year<=base[1])]
    mu = b.groupby(b.index.month).mean(); sd = b.groupby(b.index.month).std()
    q90 = b.groupby(b.index.month).quantile(.90); q95 = b.groupby(b.index.month).quantile(.95)
    mo = M.index.month
    z = (M-mu.loc[mo].values)/sd.loc[mo].values; an = M-mu.loc[mo].values
    h90 = M >= q90.loc[mo].values; h95 = M >= q95.loc[mo].values
    return an, z, h90, h95
Tx_an, Tx_z, hot90, hot95 = month_stats(Txm)
Tn_an, Tn_z, _, _ = month_stats(Tnm)
DTR_an, _, _, _ = month_stats(DTRm)
def cdhe(dry_idx, hot, thr=DROUGHT): return (dry_idx <= thr) & hot
CDHE = cdhe(spi3, hot90); CDHE_spei = cdhe(spei3, hot90); CDHE_p95 = cdhe(spi3, hot95)
CDHE_spi6 = cdhe(spi6, hot90); CDHE_dry15 = cdhe(spi3, hot90, -1.5)
CDHI = 0.5*(-spi3) + 0.5*Tx_z           # continuous compound severity (equal weights)
CDHI_spei = 0.5*(-spei3) + 0.5*Tx_z
S.update(Tx_an=Tx_an, Tx_z=Tx_z, Tn_an=Tn_an, DTR_an=DTR_an, hot90=hot90, hot95=hot95, CDHE=CDHE, CDHE_spei=CDHE_spei,
         CDHE_p95=CDHE_p95, CDHE_spi6=CDHE_spi6, CDHE_dry15=CDHE_dry15, CDHI=CDHI, CDHI_spei=CDHI_spei)

# climatology table
clim = pd.DataFrame({'Rain_mm_yr': ann['PRCP_total'].mean(), 'Tmax': ann['Tmax_mean'].mean(), 'Tmin': ann['Tmin_mean'].mean(),
    'Tmean': ann['Tmean'].mean(), 'DTR': ann['DTR'].mean(), 'RH': pd.Series({k: daily[k]['RH_%'].mean() for k in stations}),
    'PET_mm_yr': ann['PET_total'].mean(), 'Kiremt_share_pct': ann['Kiremt_share_pct'].mean(), 'PCI': ann['PCI'].mean(),
    'Rain_CV_pct': 100*ann['PRCP_total'].std()/ann['PRCP_total'].mean()})
clim['Aridity_P_over_PET'] = clim.Rain_mm_yr/clim.PET_mm_yr
for c_ in ('lon', 'lat', 'elev'): clim[c_.capitalize() if c_ != 'elev' else 'Elev'] = COORD[c_].reindex(clim.index)
ZONES = ['Kolla (<1500 m)', 'Weyna Dega (1500–2300 m)', 'Dega (>2300 m)']
clim['Zone'] = np.where(clim.Elev < 1500, ZONES[0], np.where(clim.Elev <= 2300, ZONES[1], ZONES[2]))
clim.round(2).to_csv(f'{TAB}/T2_station_climatology.csv'); S['clim'] = clim
print(clim.round(1).to_string()); print('Stations per zone:', clim.Zone.value_counts().to_dict())
stz = clim.sort_values('Elev', ascending=False).index.tolist()      # high -> low elevation ordering
ZC = {ZONES[0]: '#b2182b', ZONES[1]: '#999999', ZONES[2]: '#2166ac'}
zone = clim.Zone
def _pretty(n):
    fix = {'BahirDar':'Bahir Dar', 'D_Birhan':'Debre Birhan', 'D_Markos':'Debre Markos', 'D_Tabor':'Debre Tabor', 'D_Work':'Debre Work', 'Shmbu':'Shambu',
           'ArbeGebya':'Arbe Gebya', 'WotetAbay':'Wotet Abay', 'DegoT':'DegoTsion'}
    return fix.get(n, n[:1].upper()+n[1:])
LBL = {n: _pretty(n) for n in COORD.index}
lab = lambda s: LBL.get(s, s)
NS = len(stz)                                           # number of analysed stations (never hard-coded)
S.update(stz=stz, zone=zone, ZC=ZC, LBL=LBL, NS=NS, ZONES=ZONES)
# CDHE quick stats
tot = pd.DataFrame({'drought_pct': 100*(spi3<=DROUGHT).mean(), 'hot_pct': 100*hot90.mean(), 'CDHE_pct': 100*CDHE.mean(),
                    'expected_indep_pct': 100*(spi3<=DROUGHT).mean()*hot90.mean()})
tot['obs_over_exp'] = tot.CDHE_pct/tot.expected_indep_pct
print(tot.round(2).to_string()); S['tot']=tot
# (state is saved at the end of parts 2 and 3)


# %% ===================== PART 2: TRENDS, CHANGE-POINTS, COPULAS, ZONES, EOF, WAVELETS, SENSITIVITY =====================
from scipy.cluster import hierarchy as sch
R = {}                                    # numeric results for the report


# ---------------------------------------------------------------- annual/seasonal series incl. CDHE counts
def season_sum(M, months, how='sum', shift_jan=False):
    m = M[M.index.month.isin(months)].copy(); yr = m.index.year.values.copy()
    if shift_jan: yr = np.where(m.index.month.values==1, yr-1, yr)
    g = m.groupby(yr); out = g.sum(min_count=1) if how=='sum' else g.mean()
    if shift_jan: out = out.loc[out.index<=2024]
    return out
SEAS = {'Belg':([2,3,4,5],False), 'Kiremt':([6,7,8,9],False), 'Bega':([10,11,12,1],True)}
for sn, (mm, sh) in SEAS.items():
    ann[f'Rain_{sn}'] = season_sum(Pm, mm, 'sum', sh); ann[f'Tmax_{sn}'] = season_sum(Txm, mm, 'mean', sh)
    ann[f'CDHE_{sn}'] = season_sum(CDHE.astype(float), mm, 'sum', sh)
ann['CDHE_months'] = CDHE.astype(float).groupby(CDHE.index.year).sum()
ann['CDHE_months_SPEI'] = CDHE_spei.astype(float).groupby(CDHE.index.year).sum()
ann['Drought_months_SPI3'] = (spi3<=DROUGHT).astype(float).groupby(spi3.index.year).sum()
ann['Drought_months_SPEI3'] = (spei3<=DROUGHT).astype(float).groupby(spi3.index.year).sum()
ann['Hot_months_P90'] = hot90.astype(float).groupby(hot90.index.year).sum()
ann['SPEI12_Dec'] = spei12[spei12.index.month==12].set_index(spei12.index[spei12.index.month==12].year)
ann['CDHI_mean'] = CDHI.groupby(CDHI.index.year).mean()
# CDHE events (runs of consecutive CDHE months)
ev_rows = []
for s in stations:
    for st, ln in runs(CDHE[s].values):
        idx = CDHE.index[st:st+ln]; ev_rows.append(dict(station=s, start=idx[0], end=idx[-1], duration=ln, severity=float(CDHI[s].iloc[st:st+ln].sum()),
                                                         peak=float(CDHI[s].iloc[st:st+ln].max()), year=idx[0].year))
EV = pd.DataFrame(ev_rows); EV.to_csv(f'{TAB}/T5a_CDHE_events.csv', index=False)
ann['CDHE_events'] = EV.groupby(['year','station']).size().unstack().reindex(index=years, columns=stations).fillna(0)
S['ann'], S['EV'] = ann, EV

# ---------------------------------------------------------------- MK / Hamed-Rao modified MK / Sen slope
def mk(x, alpha=0.05):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    if n < 10 or np.nanstd(x) == 0: return dict(n=n, tau=np.nan, slope=np.nan, lo=np.nan, hi=np.nan, p=np.nan, p_mmk=np.nan)
    sg = np.sign(x[None,:]-x[:,None]); Sx = sg[np.triu_indices(n, 1)].sum()
    _, cnt = np.unique(x, return_counts=True); var = (n*(n-1)*(2*n+5) - (cnt*(cnt-1)*(2*cnt+5)).sum())/18
    zf = lambda s_, v: 0 if s_==0 else (s_-np.sign(s_))/np.sqrt(v)
    z = zf(Sx, var); p = 2*(1-stats.norm.cdf(abs(z)))
    i, j = np.triu_indices(n, 1); sl = (x[j]-x[i])/(j-i); sen = np.median(sl)
    # Hamed-Rao (1998) variance correction using detrended ranks
    d = x - sen*np.arange(n); rk = stats.rankdata(d); rk = rk-rk.mean(); acf = np.array([np.sum(rk[:n-l]*rk[l:])/np.sum(rk**2) for l in range(1, n-1)])
    sig = np.abs(acf) > 1.96/np.sqrt(n); lags = np.arange(1, n-1)
    cf = 1 + 2/(n*(n-1)*(n-2))*np.sum((n-lags[sig])*(n-lags[sig]-1)*(n-lags[sig]-2)*acf[sig]) if sig.any() else 1.0
    cf = max(cf, 1e-3); zm = zf(Sx, var*cf); pm = 2*(1-stats.norm.cdf(abs(zm)))
    N = len(sl); C = stats.norm.ppf(1-alpha/2)*np.sqrt(var); ss = np.sort(sl)
    lo = ss[int(np.floor((N-C)/2))] if (N-C)/2 >= 0 else ss[0]; hi = ss[min(int(np.ceil((N+C)/2)), N-1)]
    return dict(n=n, tau=Sx/(n*(n-1)/2), slope=sen, lo=lo, hi=hi, p=p, p_mmk=pm)
def fdr_bh(p):
    p = np.asarray(p, float); ok = ~np.isnan(p); q = np.full(len(p), np.nan); pv = p[ok]; o = np.argsort(pv); n = len(pv)
    r = pv[o]*n/np.arange(1, n+1); r = np.minimum.accumulate(r[::-1])[::-1]; tmp = np.empty(n); tmp[o] = np.minimum(r, 1); q[ok] = tmp; return q
TVARS = ['PRCP_total','Wet_days','R10mm','R20mm','Rx1day','Rx5day','SDII','CDD','CWD','R95pTOT_pct','PCI','Kiremt_share_pct','Rain_Belg','Rain_Kiremt','Rain_Bega',
         'Tmax_mean','Tmin_mean','Tmean','DTR','TXx','TNn','TX90p','TN90p','HW_days','WSDI','Tmax_Belg','Tmax_Kiremt','Tmax_Bega','PET_total',
         'Drought_months_SPI3','Drought_months_SPEI3','Hot_months_P90','CDHE_months','CDHE_months_SPEI','CDHE_events','CDHE_Belg','CDHE_Kiremt','CDHE_Bega','CDHI_mean']
rows = []
for v in TVARS:
    for s in stations + ['BASIN']:
        ser = ann[v].mean(axis=1) if s == 'BASIN' else ann[v][s]
        r = mk(ser.values); r.update(var=v, station=s); rows.append(r)
TR = pd.DataFrame(rows)
mask = TR.station != 'BASIN'
TR.loc[mask, 'q_fdr'] = fdr_bh(TR.loc[mask, 'p_mmk'].values); TR['slope_per_decade'] = TR.slope*10
TR['lo_dec'], TR['hi_dec'] = TR.lo*10, TR.hi*10
TR.to_csv(f'{TAB}/T4_trend_statistics_all.csv', index=False); S['TR'] = TR
def sig_summary(v):
    t = TR[(TR['var']==v)&mask]; return dict(n_pos_sig=int(((t.q_fdr<.05)&(t.slope>0)).sum()), n_neg_sig=int(((t.q_fdr<.05)&(t.slope<0)).sum()),
        n_pos=int((t.slope>0).sum()), median_slope_dec=float(t.slope_per_decade.median()))
R['trend_summary'] = {v: sig_summary(v) for v in TVARS}
R['basin_trends'] = TR[TR.station=='BASIN'].set_index('var')[['slope_per_decade','lo_dec','hi_dec','p','p_mmk','tau']].round(4).to_dict('index')

# ---------------------------------------------------------------- homogeneity + change points
def pettitt(x):
    x = np.asarray(x, float); n = len(x); U = np.array([np.sign(x[:t+1,None]-x[None,t+1:]).sum() for t in range(n-1)])
    K = np.abs(U).max(); t = np.abs(U).argmax(); return t+1, K, min(1, 2*np.exp(-6*K**2/(n**3+n**2)))
def snht(x):
    z = (x-x.mean())/x.std(ddof=1); n = len(z); T = [k*z[:k].mean()**2+(n-k)*z[k:].mean()**2 for k in range(1, n)]; return int(np.argmax(T)+1), float(np.max(T))
_sim = np.array([snht(rng.standard_normal(45))[1] for _ in range(2000)]); SNHT_CRIT = np.percentile(_sim, 95)
hom = []
for v in ['Tmean','PRCP_total']:
    A = ann[v]; Z = (A-A.mean())/A.std()
    for s in stations:
        ref = Z.drop(columns=s).mean(axis=1); rel = (Z[s]-ref)
        for kind, ser in (('raw', Z[s]), ('relative', rel)):
            t, K, p = pettitt(ser.values); ts, T = snht(ser.values)
            hom.append(dict(var=v, station=s, series=kind, pettitt_year=int(years[t-1]), pettitt_p=p, snht_year=int(years[ts-1]), snht_T=T, snht_sig=T>SNHT_CRIT))
HOM = pd.DataFrame(hom); HOM.to_csv(f'{TAB}/T3_homogeneity.csv', index=False); S['HOM'] = HOM
R['snht_crit'] = float(SNHT_CRIT)
R['hom_summary'] = HOM.groupby(['var','series']).agg(n_pettitt_sig=('pettitt_p', lambda x: int((x<.05).sum())), n_snht_sig=('snht_sig','sum'), med_year=('pettitt_year','median')).reset_index().to_dict('records')
cp = []
for v in ['CDHE_months','Hot_months_P90','Drought_months_SPI3','Tmax_mean','PRCP_total','SPEI12_Dec']:
    for s in ['BASIN']+stations:
        ser = ann[v].mean(axis=1) if s=='BASIN' else ann[v][s]; ser = ser.dropna(); t, K, p = pettitt(ser.values)
        yr = ser.index[t-1]; cp.append(dict(var=v, station=s, year=int(yr), p=p, pre=ser.iloc[:t].mean(), post=ser.iloc[t:].mean()))
CP = pd.DataFrame(cp); CP.to_csv(f'{TAB}/T4b_changepoints.csv', index=False); S['CP'] = CP

# ---------------------------------------------------------------- copulas
def _pobs(x): return stats.rankdata(x)/(len(x)+1)
def ll_gauss(th, u, v):
    a, b = stats.norm.ppf(u), stats.norm.ppf(v); return np.sum(-0.5*np.log(1-th**2)-(th**2*(a**2+b**2)-2*th*a*b)/(2*(1-th**2)))
def ll_clay(th, u, v): return np.sum(np.log1p(th)-(1+th)*np.log(u*v)-(2+1/th)*np.log(u**-th+v**-th-1))
def ll_gumb(th, u, v):   # clean log-density of the Gumbel-Hougaard copula
    a, b = -np.log(u), -np.log(v); A = a**th+b**th; C = np.exp(-A**(1/th))
    return np.sum(np.log(C)-np.log(u*v)+(th-1)*np.log(a*b)+(2/th-2)*np.log(A)+np.log1p((th-1)*A**(-1/th)))
def ll_frank(th, u, v):
    e = np.exp(-th); return np.sum(np.log(th*(1-e))-th*(u+v)-2*np.log((1-e)-(1-np.exp(-th*u))*(1-np.exp(-th*v))))
def cdf_c(fam, th, u, v):
    if fam=='Gaussian': return stats.multivariate_normal.cdf([stats.norm.ppf(u), stats.norm.ppf(v)], mean=[0,0], cov=[[1,th],[th,1]])
    if fam=='Clayton':  return (u**-th+v**-th-1)**(-1/th)
    if fam=='Gumbel':   return np.exp(-((-np.log(u))**th+(-np.log(v))**th)**(1/th))
    if fam=='Frank':    return -np.log(1+(np.exp(-th*u)-1)*(np.exp(-th*v)-1)/(np.exp(-th)-1))/th
FAM = {'Gaussian':(ll_gauss,(-.95,.95)), 'Clayton':(ll_clay,(0.01,15)), 'Gumbel':(ll_gumb,(1.001,15)), 'Frank':(ll_frank,(0.01,30))}
def fit_cop(u, v, fams=FAM):
    out = {}
    for f, (ll, bd) in fams.items():
        try:
            r = optimize.minimize_scalar(lambda t: -ll(t, u, v), bounds=bd, method='bounded'); out[f] = dict(theta=r.x, ll=-r.fun, aic=2*r.fun+2, bic=2*r.fun+np.log(len(u)))
        except Exception: pass
    return out
def tail_dep(f, th):
    return {'Gumbel': (0, 2-2**(1/th)), 'Clayton': (2**(-1/th), 0)}.get(f, (0, 0))
cop_rows = []; COPFIT = {}
for s in stations:
    d = pd.DataFrame({'D': -spi3[s], 'H': Tx_z[s]}).dropna(); u, v = _pobs(d.D.values), _pobs(d.H.values)
    pd_, ph_ = (spi3[s]<=DROUGHT).mean(), hot90[s].mean()            # marginal exceedance probabilities
    fits = fit_cop(u, v); best = min(fits, key=lambda f: fits[f]['aic']); th = fits[best]['theta']
    u0, v0 = 1-pd_, 1-ph_; joint = 1-u0-v0+cdf_c(best, th, u0, v0)     # P(D>=d, H>=h): dry AND hot
    kt = stats.kendalltau(d.D, d.H)[0]
    # block bootstrap over years (12-month blocks) for CI of joint probability
    bj = []; n = len(u); nb = n//12
    for _ in range(150):
        b = rng.integers(0, nb, nb); ii = (b[:,None]*12+np.arange(12)).ravel(); ii = ii[ii<n]
        uu, vv = _pobs(d.D.values[ii]), _pobs(d.H.values[ii])
        r = optimize.minimize_scalar(lambda t: -FAM[best][0](t, uu, vv), bounds=FAM[best][1], method='bounded')
        bj.append(1-u0-v0+cdf_c(best, r.x, u0, v0))
    lamL, lamU = tail_dep(best, th)
    cop_rows.append(dict(station=s, kendall_tau=kt, best=best, theta=th, **{f'AIC_{f}': fits[f]['aic'] for f in fits}, p_drought=pd_, p_hot=ph_, joint_p_copula=joint,
        joint_p_indep=pd_*ph_, joint_p_empirical=CDHE[s].mean(), ratio_cop_over_indep=joint/(pd_*ph_), ci_lo=np.percentile(bj, 2.5), ci_hi=np.percentile(bj, 97.5),
        return_period_months=1/joint, lambda_upper=lamU, lambda_lower=lamL))
COP = pd.DataFrame(cop_rows).set_index('station'); COP.round(4).to_csv(f'{TAB}/T6_copula_results.csv'); S['COP'] = COP
# seasonal Kendall tau between dry severity and heat
seas_tau = {}
for sn, (mm, _) in SEAS.items():
    seas_tau[sn] = {s: stats.kendalltau(-spi3[s][spi3.index.month.isin(mm)].dropna(), Tx_z[s][spi3.index.month.isin(mm)].loc[spi3[s][spi3.index.month.isin(mm)].dropna().index])[0] for s in stations}
SEASTAU = pd.DataFrame(seas_tau); SEASTAU.round(3).to_csv(f'{TAB}/T6b_seasonal_kendall_tau.csv'); S['SEASTAU'] = SEASTAU
# obs CDHE frequency by season
seas_freq = pd.DataFrame({sn: CDHE[CDHE.index.month.isin(mm)].mean()*100 for sn, (mm, _) in SEAS.items()}); S['seas_freq'] = seas_freq

# ---------------------------------------------------------------- early vs late period; event characteristics
def per(df, a, b): return df[(df.index.year>=a)&(df.index.year<=b)]
E1, E2 = (1981, 2003), (2004, 2025)
cmp = pd.DataFrame({'CDHE_pct_early': 100*per(CDHE,*E1).mean(), 'CDHE_pct_late': 100*per(CDHE,*E2).mean(),
                    'drought_pct_early': 100*(per(spi3,*E1)<=DROUGHT).mean(), 'drought_pct_late': 100*(per(spi3,*E2)<=DROUGHT).mean(),
                    'hot_pct_early': 100*per(hot90,*E1).mean(), 'hot_pct_late': 100*per(hot90,*E2).mean(),
                    'Tmax_early': per(Txm,*E1).mean(), 'Tmax_late': per(Txm,*E2).mean(), 'Rain_early': per(Pm,*E1).sum()/23, 'Rain_late': per(Pm,*E2).sum()/22})
cmp['dTmax'] = cmp.Tmax_late-cmp.Tmax_early; cmp['CDHE_ratio'] = cmp.CDHE_pct_late/cmp.CDHE_pct_early; cmp['dRain_pct'] = 100*(cmp.Rain_late/cmp.Rain_early-1)
cmp.round(3).to_csv(f'{TAB}/T5b_early_vs_late.csv'); S['cmp'] = cmp
ev_stat = EV.assign(period=np.where(EV.year<=2003, 'early', 'late')).groupby(['station','period']).agg(n=('duration','size'), mean_dur=('duration','mean'), max_dur=('duration','max'), mean_sev=('severity','mean'), max_peak=('peak','max')).unstack()
ev_stat.round(2).to_csv(f'{TAB}/T5c_event_stats_by_period.csv'); S['ev_stat'] = ev_stat
extent = 100*CDHE.mean(axis=1); S['extent'] = extent
R['wilcoxon_CDHE_late_vs_early'] = float(stats.wilcoxon(cmp.CDHE_pct_late, cmp.CDHE_pct_early).pvalue)
R['wilcoxon_hot'] = float(stats.wilcoxon(cmp.hot_pct_late, cmp.hot_pct_early).pvalue)
R['wilcoxon_drought'] = float(stats.wilcoxon(cmp.drought_pct_late, cmp.drought_pct_early).pvalue)

# ---------------------------------------------------------------- elevation dependence (true station elevation from ElCoord.csv)
proxy = clim.Elev
_l = stats.linregress(clim.Elev, clim.Tmean); _lx = stats.linregress(clim.Elev, clim.Tmax); _ln = stats.linregress(clim.Elev, clim.Tmin)
R['lapse'] = dict(Tmean_C_per_km=float(_l.slope*1000), r=float(_l.rvalue), p=float(_l.pvalue), Tmax_C_per_km=float(_lx.slope*1000), Tmin_C_per_km=float(_ln.slope*1000), n=int(len(clim)))
R['rain_vs_elev'] = dict(zip(('rho', 'p'), map(float, stats.spearmanr(clim.Elev, clim.Rain_mm_yr))))
R['elev_range'] = [float(clim.Elev.min()), float(clim.Elev.max())]
def spear(y): r, p = stats.spearmanr(proxy, y); return r, p
ed = pd.DataFrame({'CDHE_trend_per_dec': TR[(TR['var']=='CDHE_months')&mask].set_index('station').slope_per_decade,
                   'Tmax_trend_per_dec': TR[(TR['var']=='Tmax_mean')&mask].set_index('station').slope_per_decade,
                   'Tmin_trend_per_dec': TR[(TR['var']=='Tmin_mean')&mask].set_index('station').slope_per_decade,
                   'Rain_trend_pct_dec': (TR[(TR['var']=='PRCP_total')&mask].set_index('station').slope_per_decade/clim.Rain_mm_yr*100),
                   'TX90p_trend_per_dec': TR[(TR['var']=='TX90p')&mask].set_index('station').slope_per_decade,
                   'Drought_trend_per_dec': TR[(TR['var']=='Drought_months_SPI3')&mask].set_index('station').slope_per_decade,
                   'CDHE_pct_total': tot.CDHE_pct, 'Kendall_tau_DH': COP.kendall_tau, 'Joint_ratio': COP.ratio_cop_over_indep, 'Elev': clim.Elev})
ed['Zone'] = zone; ed.round(3).to_csv(f'{TAB}/T7_elevation_station_metrics.csv'); S['ed'] = ed
R['spearman_proxy'] = {c: dict(zip(('rho','p'), map(float, spear(ed[c])))) for c in ed.columns if c not in ('Zone','Elev')}
R['zone_means'] = ed.groupby('Zone').mean(numeric_only=True).round(3).to_dict('index')
R['kruskal_zone'] = {c: float(stats.kruskal(*[g[c].values for _, g in ed.groupby('Zone')]).pvalue) for c in ['Tmax_trend_per_dec','TX90p_trend_per_dec','Kendall_tau_DH','CDHE_pct_total','Joint_ratio','Rain_trend_pct_dec']}
# quantile regression of station-year CDHE months on proxy (sklearn) - upper-tail sensitivity to thermal position
from sklearn.linear_model import QuantileRegressor
yy = ann['CDHE_months'].stack().reset_index(); yy.columns = ['year','station','cd']; yy['T'] = yy.station.map(clim.Elev)/100; yy['t'] = yy.year-1981
qr = {}
for tau in (.5, .75, .9):
    m = QuantileRegressor(quantile=tau, alpha=0, solver='highs').fit(yy[['T','t']].values, yy.cd.values); qr[tau] = dict(coef_per100m=float(m.coef_[0]), coef_year=float(m.coef_[1]))
R['quantreg'] = qr

# ---------------------------------------------------------------- EOF / clustering (spatial structure and propagation)
Z3 = spei3.dropna(); Zs = (Z3-Z3.mean())/Z3.std(); U_, s_, Vt = np.linalg.svd(Zs.values, full_matrices=False)
ev_ = s_**2/np.sum(s_**2); PCs = pd.DataFrame(U_[:, :3]*s_[:3], index=Zs.index, columns=['PC1','PC2','PC3']); load = pd.DataFrame(Vt[:3].T*np.sqrt(np.maximum(1,1)), index=Zs.columns, columns=['EOF1','EOF2','EOF3'])
R['eof_var'] = (100*ev_[:3]).round(1).tolist(); S.update(PCs=PCs, load=load)
R['eof_load_vs_proxy'] = {c: dict(zip(('rho','p'), map(float, stats.spearmanr(clim.Elev.loc[load.index], load[c])))) for c in load}
mf = pd.DataFrame({s: Pm[s].groupby(Pm.index.month).mean()/Pm[s].groupby(Pm.index.month).mean().sum() for s in stations}).T
feat = pd.concat([mf, clim[['Tmax','Tmin','Rain_mm_yr']]], axis=1); featz = (feat-feat.mean())/feat.std()
LK = sch.linkage(featz.values, 'ward'); cl = pd.Series(sch.fcluster(LK, 3, 'maxclust'), index=stations); S.update(LK=LK, cl=cl)
R['clusters'] = {int(c): cl[cl==c].index.tolist() for c in sorted(cl.unique())}
R['corr_spei3_mean'] = float(np.tril(spei3.corr().values, -1)[np.tril_indices(len(stations), -1)].mean())
# spatial extent of compound events / drought / heat
S['ext'] = pd.DataFrame({'CDHE': 100*CDHE.mean(axis=1), 'Drought': 100*(spi3<=DROUGHT).mean(axis=1), 'Heat': 100*hot90.mean(axis=1)})
R['max_extent'] = S['ext'].max().round(1).to_dict(); R['max_extent_dates'] = {k: str(S['ext'][k].idxmax().date()) for k in S['ext']}
R['months_extent_ge50'] = int((S['ext'].CDHE>=50).sum()); R['months_extent_ge25'] = int((S['ext'].CDHE>=25).sum())
R['top_extent_months'] = [(str(d.date()), round(v, 1)) for d, v in S['ext'].CDHE.sort_values(ascending=False).head(8).items()]

# ---------------------------------------------------------------- wavelets (Morlet CWT, AR1 red-noise significance)
def cwt_morlet(x, dt=1/12, dj=0.25, w0=6.0, maxper=16):
    x = np.asarray(x, float); x = (x-x.mean())/x.std(); n = len(x); s0 = 2*dt; J = int(np.log2(maxper/1.033/s0)/dj)
    N = 2**int(np.ceil(np.log2(n))+1); xp = np.r_[x, np.zeros(N-n)]; xf = np.fft.fft(xp); k = 2*np.pi*np.fft.fftfreq(N, dt); sc = s0*2**(dj*np.arange(J+1))
    W = np.zeros((J+1, n), complex)
    for j, s_ in enumerate(sc):
        psi = np.pi**-.25*(k>0)*np.exp(-(s_*k-w0)**2/2)*np.sqrt(2*np.pi*s_/dt); W[j] = np.fft.ifft(xf*psi)[:n]
    per = 1.033*sc; power = np.abs(W)**2; a = np.corrcoef(x[:-1], x[1:])[0, 1]
    Pk = (1-a**2)/(1+a**2-2*a*np.cos(2*np.pi*dt/per)); sig = power/(Pk[:,None]*stats.chi2.ppf(.95, 2)/2)
    coi = np.minimum(np.arange(n), n-1-np.arange(n))*dt*1.033*np.sqrt(2)/np.sqrt(2)*1.0
    return per, power, sig, coi, W
WV = {}
for nm, ser in {'SPEI-3 (basin mean)': spei3.mean(axis=1).bfill().ffill(), 'Tmax anomaly z (basin mean)': Tx_z.mean(axis=1), 'CDHE extent (% stations)': extent}.items():
    pr_, pw, sg, coi, W = cwt_morlet(ser.values); gws = pw.mean(axis=1); WV[nm] = (pr_, pw, sg, coi, gws)
    R.setdefault('wavelet_peak_period_yr', {})[nm] = float(pr_[np.argmax(gws)])
for nm, (pr_, pw, sg, coi, gws) in WV.items():
    b = (pr_>=1.5)&(pr_<=8); R.setdefault('wavelet_band_1p5_8yr', {})[nm] = dict(peak_period=float(pr_[b][np.argmax(gws[b])]), frac_signif=float((sg[b]>1).mean()))
# ---------------------------------------------------------------- dominance of the 2001-2005 episode
epm = (CDHE.index.year>=2001)&(CDHE.index.year<=2005)
R['episode'] = dict(share_CDHE_in_2001_2005_pct=float(100*CDHE.values[epm].sum()/CDHE.values.sum()), share_of_time_pct=float(100*epm.mean()),
    CDHE_pct_inside=float(100*CDHE.values[epm].mean()), CDHE_pct_outside=float(100*CDHE.values[~epm].mean()),
    hot_pct_inside=float(100*hot90.values[epm].mean()), hot_pct_outside=float(100*hot90.values[~epm].mean()),
    drought_pct_inside=float(100*(spi3.values[epm]<=DROUGHT).mean()), drought_pct_outside=float(100*(spi3.values[~epm]<=DROUGHT).mean()),
    rain_inside=float(ann['PRCP_total'].loc[2001:2005].mean().mean()), rain_outside=float(ann['PRCP_total'].drop(range(2001,2006)).mean().mean()),
    tmax_inside=float(ann['Tmax_mean'].loc[2001:2005].mean().mean()), tmax_outside=float(ann['Tmax_mean'].drop(range(2001,2006)).mean().mean()))
ex = ann['CDHE_months'].drop(range(2001, 2006)).mean(axis=1); r_ = mk(ex.values); R['episode']['CDHE_trend_excl_dec'] = dict(slope=r_['slope']*10, p=r_['p_mmk'])
exh = ann['Hot_months_P90'].drop(range(2001, 2006)).mean(axis=1); r_ = mk(exh.values); R['episode']['Hot_trend_excl_dec'] = dict(slope=r_['slope']*10, p=r_['p_mmk'])
exd = ann['Drought_months_SPI3'].drop(range(2001, 2006)).mean(axis=1); r_ = mk(exd.values); R['episode']['Drought_trend_excl_dec'] = dict(slope=r_['slope']*10, p=r_['p_mmk'])
ext_ = ann['Tmax_mean'].drop(range(2001, 2006)).mean(axis=1); r_ = mk(ext_.values); R['episode']['Tmax_trend_excl_dec'] = dict(slope=r_['slope']*10, p=r_['p_mmk'])
R['episode']['tau_median_all'] = float(COP.kendall_tau.median())
R['episode']['tau_median_excl'] = float(np.median([stats.kendalltau(-spi3[s][~epm].dropna(), Tx_z[s][~epm].loc[spi3[s][~epm].dropna().index])[0] for s in stations]))
# optional teleconnections (only if index files are supplied: nino34.csv / dmi.csv with columns Date,Value)
TELE = {}
for nm in ('nino34', 'dmi'):
    fp = os.path.join(DATA, f'{nm}.csv')
    if os.path.exists(fp):
        ix = pd.read_csv(fp, parse_dates=['Date']).set_index('Date').Value.resample('MS').mean()
        for lag in range(0, 7): TELE[(nm, lag)] = float(stats.spearmanr(ix.shift(lag).reindex(extent.index), extent, nan_policy='omit')[0])
R['teleconnection_run'] = bool(TELE)

# ---------------------------------------------------------------- sensitivity of the event definition
sens = {}
for nm, M in {'SPI3 & TX>P90 (baseline)': CDHE, 'SPEI3 & TX>P90': CDHE_spei, 'SPI3 & TX>P95': CDHE_p95, 'SPI6 & TX>P90': CDHE_spi6, 'SPI3<=-1.5 & TX>P90': CDHE_dry15}.items():
    a = M.astype(float).groupby(M.index.year).sum(); r = mk(a.mean(axis=1).values)
    tr = [mk(a[s].values) for s in stations]
    sens[nm] = dict(mean_pct_months=100*M.values.mean(), basin_slope_months_per_decade=r['slope']*10, basin_p=r['p_mmk'], n_stations_sig_pos=sum((t['p_mmk']<.05) and t['slope']>0 for t in tr),
                    n_stations_pos=sum(t['slope']>0 for t in tr), early_late_ratio=100*per(M,*E2).values.mean()/max(100*per(M,*E1).values.mean(), 1e-9))
SENS = pd.DataFrame(sens).T; SENS.round(3).to_csv(f'{TAB}/T9_event_definition_sensitivity.csv'); S['SENS'] = SENS
S['WV'] = WV; S['R'] = R; S['stz'] = stz; save_state(S)


# %% ===================== PART 2b: SIGNIFICANCE OF DRYNESS-HEAT DEPENDENCE (year-block permutation) AND EXCESS BY PERIOD =====================
# H0: drought and heat are independent. Whole calendar years of the hot-month series are permuted (same permutation for all stations, which keeps
# the spatial coherence, seasonality and within-year persistence of each series) and the compound frequency is recomputed.
_perm_rng = np.random.default_rng(SEED + 7)
_dry = (spi3 <= DROUGHT); _hot = hot90.copy()
_yrs = np.arange(1982, 2026)                                                        # complete calendar years with defined SPI-3
_Dm = _dry.loc['1982-01-01':].values.reshape(len(_yrs), 12, -1); _Hm = _hot.loc['1982-01-01':].values.reshape(len(_yrs), 12, -1)
def _perm_test(keep, nperm=5000):
    D, H = _Dm[keep], _Hm[keep]; obs = (D & H).mean(axis=(0, 1))                   # per-station observed frequency
    null = np.empty((nperm, D.shape[2]))
    for i in range(nperm): null[i] = (D & H[_perm_rng.permutation(len(D))]).mean(axis=(0, 1))
    pb = (1 + np.sum(null.mean(1) >= obs.mean())) / (nperm + 1); ps = (1 + np.sum(null >= obs[None, :], axis=0)) / (nperm + 1)
    return obs, null, pb, ps
_all = np.ones(len(_yrs), bool); _excl = ~((_yrs >= 2001) & (_yrs <= 2005))
perm_rows = []
for nm_, keep in (('All years 1982-2025', _all), ('Excluding 2001-2005', _excl)):
    obs, null, pb, ps = _perm_test(keep)
    perm_rows.append(dict(sample=nm_, n_years=int(keep.sum()), observed_pct=100*obs.mean(), null_mean_pct=100*null.mean(), null_p95_pct=100*np.percentile(null.mean(1), 95), null_max_pct=100*null.mean(1).max(),
                          excess_factor=obs.mean()/null.mean(), p_basin=pb, stations_p_lt_0_05=int((ps < .05).sum()), stations_p_lt_0_01=int((ps < .01).sum())))
PERM = pd.DataFrame(perm_rows); PERM.round(4).to_csv(f'{TAB}/T10_dependence_permutation_test.csv', index=False)
def _pp(sel):                                                                      # period-wise frequencies (mean over stations)
    d = _dry[sel].mean(); h = _hot[sel].mean(); c = (_dry & _hot)[sel].mean()
    return dict(drought_pct=100*d.mean(), hot_pct=100*h.mean(), expected_indep_pct=100*(d*h).mean(), observed_pct=100*c.mean(), excess_factor=(c/(d*h)).mean(), p_drought_given_hot_pct=100*(c/h).mean(), p_hot_given_drought_pct=100*(c/d).mean(), hot_not_drought_pct=100*(h-c).mean())
_vy = _dry.index.year
EXC = pd.DataFrame({'1981-2003': _pp((_vy <= 2003)), '2004-2025': _pp((_vy >= 2004)), '2001-2005': _pp((_vy >= 2001) & (_vy <= 2005)), 'All other years': _pp(~((_vy >= 2001) & (_vy <= 2005))), '1981-2000': _pp(_vy <= 2000), '2006-2025': _pp(_vy >= 2006)}).T
EXC.round(3).to_csv(f'{TAB}/T10b_excess_by_period.csv'); R['perm'] = PERM.set_index('sample').to_dict('index'); R['excess_by_period'] = EXC.to_dict('index')
print(PERM.round(4).to_string()); print(EXC.round(2).to_string())

_e, _l = (_vy <= 2000), (_vy >= 2006)                                              # post hoc contrast that excludes the 2001-2005 episode
R['wilcoxon_pre_post_episode'] = {k: float(stats.wilcoxon(v[_l].mean(), v[_e].mean()).pvalue) for k, v in (('compound', _dry & _hot), ('hot', _hot), ('drought', _dry))}
R['n_stations_compound_higher_post'] = int(((_dry & _hot)[_l].mean() > (_dry & _hot)[_e].mean()).sum())
R['median_compound_pct_pre_post'] = [float(100*(_dry & _hot)[_e].mean().median()), float(100*(_dry & _hot)[_l].mean().median())]
print(R['wilcoxon_pre_post_episode'], R['n_stations_compound_higher_post'], R['median_compound_pct_pre_post'])


# %% ===================== PART 2c: ELEVATION DEPENDENCE - EXTENDED TESTS (family-wise correction, spatial confounding, robustness) =====================
_trm = TR[TR.station != 'BASIN']
def _ts(v): return _trm[_trm['var'] == v].set_index('station').slope_per_decade.reindex(stations)
E2 = pd.DataFrame({'Tmax trend (°C/decade)': _ts('Tmax_mean'), 'Tmin trend (°C/decade)': _ts('Tmin_mean'), 'Tmean trend (°C/decade)': _ts('Tmean'), 'DTR trend (°C/decade)': _ts('DTR'),
    'TX90p trend (% of days/decade)': _ts('TX90p'), 'TN90p trend (% of days/decade)': _ts('TN90p'), 'Heatwave-days trend (days/decade)': _ts('HW_days'), 'PET trend (mm/decade)': _ts('PET_total'),
    'Rainfall trend (%/decade)': 100*_ts('PRCP_total')/clim.Rain_mm_yr.reindex(stations), 'Wet-day trend (days/decade)': _ts('Wet_days'), 'Hot-month trend (months/decade)': _ts('Hot_months_P90'),
    'Compound frequency (% of months)': tot.CDHE_pct.reindex(stations), 'Kendall tau (dryness, heat)': COP.kendall_tau.reindex(stations), 'Excess factor (copula/independence)': COP.ratio_cop_over_indep.reindex(stations)})
S['E2'] = E2
_el, _la, _lo = [clim[c].reindex(stations).values.astype(float) for c in ('Elev', 'Lat', 'Lon')]
_grp = np.array([s in group_members for s in stations])                       # series that represent a group of identical station files
_rg = np.random.default_rng(SEED + 11); rows_ = []
def _ols_t(y, X):
    X1 = np.c_[np.ones(len(y)), X]; b, *_ = np.linalg.lstsq(X1, y, rcond=None); res = y - X1@b; df = len(y) - X1.shape[1]; s2 = res@res/df
    se = np.sqrt(np.diag(s2*np.linalg.inv(X1.T@X1))); return b, b/se, 2*(1 - stats.t.cdf(np.abs(b/se), df))
for c in E2:
    y = E2[c].values.astype(float); ok = np.isfinite(y)
    rho, p = stats.spearmanr(_el[ok], y[ok])
    bs = []
    for _ in range(2000):
        i = _rg.integers(0, ok.sum(), ok.sum()); bs.append(stats.spearmanr(_el[ok][i], y[ok][i])[0])
    ry, re_, rla, rlo = [stats.rankdata(v[ok]) for v in (y, _el, _la, _lo)]
    b_, t_, p_ = _ols_t((ry-ry.mean())/ry.std(), np.c_[(re_-re_.mean())/re_.std(), (rla-rla.mean())/rla.std(), (rlo-rlo.mean())/rlo.std()])
    b2, t2, p2 = _ols_t(y[ok], np.c_[_el[ok]/100, _la[ok], _lo[ok]])
    k2 = ok & ~_grp; rho2, pp2 = stats.spearmanr(_el[k2], y[k2])
    rows_.append(dict(metric=c, n=int(ok.sum()), rho=rho, ci_lo=np.nanpercentile(bs, 2.5), ci_hi=np.nanpercentile(bs, 97.5), p=p, partial_rank_beta_elev=b_[1], p_partial_lat_lon=p_[1],
                      slope_per_100m=b2[1], p_slope=p2[1], n_without_groups=int(k2.sum()), rho_without_groups=rho2, p_without_groups=pp2))
EL = pd.DataFrame(rows_).set_index('metric'); EL['q_fdr'] = fdr_bh(EL.p.values); EL['p_bonferroni'] = np.minimum(1, EL.p*len(EL))
EL.round(4).to_csv(f'{TAB}/T7b_elevation_extended_tests.csv'); R['elev_ext'] = EL.to_dict('index')
_kw = {c: float(stats.kruskal(*[E2[c][clim.Zone.reindex(stations) == z].dropna().values for z in ZONES]).pvalue) for c in E2}; R['elev_kruskal_all'] = _kw
_zm = E2.groupby(clim.Zone.reindex(stations)).mean(); _zm.round(3).to_csv(f'{TAB}/T7c_zone_means_extended.csv'); R['zone_means_ext'] = _zm.to_dict('index')
print(EL[['rho', 'ci_lo', 'ci_hi', 'p', 'q_fdr', 'p_partial_lat_lon', 'slope_per_100m', 'rho_without_groups', 'p_without_groups']].round(3).to_string()); print({k: round(v, 3) for k, v in _kw.items()})


# %% ===================== PART 3: MACHINE LEARNING (classification + severity), TIME-BLOCKED VALIDATION, EXPLAINABILITY =====================
import time
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss, f1_score, precision_score, recall_score,
                             balanced_accuracy_score, mean_squared_error, mean_absolute_error, r2_score, roc_curve, precision_recall_curve)
from sklearn.inspection import permutation_importance
HAVE = {m: (__import__('importlib').util.find_spec(m) is not None) for m in ('xgboost','lightgbm','catboost','shap')}
print('optional libs available:', HAVE)
RM = {}                                             # ML results for the report
TRAIN_END, VAL_END = 2004, 2014                      # train 1981-2004 | validation 2005-2014 | independent test 2015-2025

# ---------------------------------------------------------------- feature engineering (lagged precursors only -> avoids target circularity)
RH_an, _, _, _ = month_stats(RHm); W_an, _, _, _ = month_stats(Wm); PET_an, _, _, _ = month_stats(PETm)
def build(s):
    d = pd.DataFrame(index=Pm.index)
    for l in (1, 2, 3): d[f'P_l{l}'] = Pm[s].shift(l); d[f'Tx_an_l{l}'] = Tx_an[s].shift(l)
    d['Tn_an_l1'] = Tn_an[s].shift(1); d['DTR_an_l1'] = DTR_an[s].shift(1); d['RH_an_l1'] = RH_an[s].shift(1); d['Wind_an_l1'] = W_an[s].shift(1); d['PET_an_l1'] = PET_an[s].shift(1)
    d['SPI3_l1'] = spi3[s].shift(1); d['SPEI3_l1'] = spei3[s].shift(1); d['Tx_an_12m_l1'] = Tx_an[s].rolling(12).mean().shift(1)
    d['Elev'] = clim.Elev[s]; d['Lat'] = clim.Lat[s]; d['Lon'] = clim.Lon[s]; d['MAP'] = clim.Rain_mm_yr[s]
    d['sin_month'] = np.sin(2*np.pi*d.index.month/12); d['cos_month'] = np.cos(2*np.pi*d.index.month/12)
    d['y'] = CDHE[s].astype(int); d['cdhi'] = CDHI[s]; d['cdhi_l1'] = CDHI[s].shift(1); d['station'] = s; d['year'] = d.index.year; d['month'] = d.index.month
    return d
DF = pd.concat([build(s) for s in stations]).dropna(subset=['Tx_an_12m_l1', 'SPI3_l1', 'SPEI3_l1', 'cdhi']).reset_index().rename(columns={'index':'date'})
FEATS = ['P_l1','P_l2','P_l3','Tx_an_l1','Tx_an_l2','Tx_an_l3','Tn_an_l1','DTR_an_l1','RH_an_l1','Wind_an_l1','PET_an_l1','SPI3_l1','SPEI3_l1','Tx_an_12m_l1','Elev','Lat','Lon','MAP','sin_month','cos_month']
NICE = {'P_l1':'Rain (t-1)','P_l2':'Rain (t-2)','P_l3':'Rain (t-3)','Tx_an_l1':'Tmax anom (t-1)','Tx_an_l2':'Tmax anom (t-2)','Tx_an_l3':'Tmax anom (t-3)','Tn_an_l1':'Tmin anom (t-1)',
        'DTR_an_l1':'DTR anom (t-1)','RH_an_l1':'RH anom (t-1)','Wind_an_l1':'Wind anom (t-1)','PET_an_l1':'PET anom (t-1)','SPI3_l1':'SPI-3 (t-1)','SPEI3_l1':'SPEI-3 (t-1)',
        'Tx_an_12m_l1':'Tmax anom, prior 12 mo','Elev':'Elevation','Lat':'Latitude','Lon':'Longitude','MAP':'Mean annual rain','sin_month':'Season (sin)','cos_month':'Season (cos)'}
tr_m, va_m, te_m = DF.year<=TRAIN_END, (DF.year>TRAIN_END)&(DF.year<=VAL_END), DF.year>VAL_END
Xtr, Xva, Xte = DF.loc[tr_m, FEATS], DF.loc[va_m, FEATS], DF.loc[te_m, FEATS]; ytr, yva, yte = DF.loc[tr_m,'y'].values, DF.loc[va_m,'y'].values, DF.loc[te_m,'y'].values
RM['split'] = {k: dict(n=int(m.sum()), events=int(DF.loc[m,'y'].sum()), prevalence_pct=float(100*DF.loc[m,'y'].mean())) for k, m in (('train',tr_m),('val',va_m),('test',te_m))}
print(RM['split'])
sw = lambda y: np.where(y==1, (y==0).sum()/max((y==1).sum(),1), 1.0)     # balanced sample weights

# ---------------------------------------------------------------- models and hyper-parameter search (random search, validation PR-AUC)
def mk_models():
    M = {'Logistic regression': (lambda **p: make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000, **p)), dict(C=[.01,.03,.1,.3,1,3,10])),
         'Random forest': (lambda **p: RandomForestClassifier(class_weight='balanced_subsample', n_jobs=1, random_state=SEED, **p),
                           dict(n_estimators=[200,400], max_depth=[4,6,8,12], min_samples_leaf=[5,10,20,40], max_features=['sqrt',0.5])),
         'Hist. gradient boosting (LightGBM-type)': (lambda **p: HistGradientBoostingClassifier(class_weight='balanced', random_state=SEED, early_stopping=False, **p),
                           dict(learning_rate=[.02,.05,.1], max_iter=[100,200,300], max_depth=[2,3,4,6], min_samples_leaf=[10,20,40], l2_regularization=[0,1,10])),
         'Gradient boosting (XGBoost-type)': (lambda **p: GradientBoostingClassifier(subsample=0.7, random_state=SEED, **p),
                           dict(n_estimators=[100,200,300], learning_rate=[.03,.05,.1], max_depth=[2,3,4], min_samples_leaf=[10,20,40]))}
    if HAVE['xgboost']:
        import xgboost as xgb; M['XGBoost'] = (lambda **p: xgb.XGBClassifier(subsample=.7, colsample_bytree=.8, eval_metric='logloss', random_state=SEED, n_jobs=1, **p),
                           dict(n_estimators=[100,200,300], learning_rate=[.03,.05,.1], max_depth=[2,3,4,6], min_child_weight=[1,5,10], reg_lambda=[1,5,10]))
    if HAVE['lightgbm']:
        import lightgbm as lgb; M['LightGBM'] = (lambda **p: lgb.LGBMClassifier(class_weight='balanced', subsample=.7, subsample_freq=1, verbose=-1, random_state=SEED, n_jobs=1, **p),
                           dict(n_estimators=[100,200,300], learning_rate=[.03,.05,.1], num_leaves=[4,8,16], min_child_samples=[10,20,40], reg_lambda=[0,1,10]))
    if HAVE['catboost']:
        import catboost as cb; M['CatBoost'] = (lambda **p: cb.CatBoostClassifier(auto_class_weights='Balanced', verbose=0, random_seed=SEED, thread_count=1, **p),
                           dict(iterations=[200,400], depth=[3,4,6], learning_rate=[.03,.05,.1], l2_leaf_reg=[1,3,10]))
    return M
def fitm(name, ctor, p, X, y):
    m = ctor(**p)
    if name.startswith('Gradient boosting') or name=='XGBoost': m.fit(X, y, sample_weight=sw(y))
    else: m.fit(X, y)
    return m
def prob(m, X): return m.predict_proba(X)[:, 1]
def best_f1_thr(y, p):
    pr, rc, th = precision_recall_curve(y, p); f = 2*pr*rc/np.maximum(pr+rc, 1e-9); i = np.nanargmax(f[:-1]); return float(th[i])
MODELS, tuned, fitted, search_log = mk_models(), {}, {}, []
t0 = time.time()
for name, (ctor, grid) in MODELS.items():
    best = (-1, None)
    for p in ParameterSampler(grid, n_iter=min(12, int(np.prod([len(v) for v in grid.values()]))), random_state=SEED):
        m = fitm(name, ctor, p, Xtr, ytr); sc = average_precision_score(yva, prob(m, Xva)); search_log.append(dict(model=name, val_prauc=sc, **p))
        if sc > best[0]: best = (sc, p)
    tuned[name] = best[1]; fitted[name] = fitm(name, ctor, best[1], Xtr, ytr); print(f'{name:45s} val PR-AUC {best[0]:.3f}  {best[1]}  [{time.time()-t0:.0f}s]')
pd.DataFrame(search_log).to_csv(f'{TAB}/T8c_hyperparameter_search_log.csv', index=False)
RM['tuned'] = {k: {a: (float(b) if isinstance(b, (int, float, np.floating)) else b) for a, b in v.items()} for k, v in tuned.items()}

def cls_metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    return dict(ROC_AUC=roc_auc_score(y, p), PR_AUC=average_precision_score(y, p), Brier=brier_score_loss(y, np.clip(p, 0, 1)), F1=f1_score(y, yh), Precision=precision_score(y, yh, zero_division=0),
                Recall=recall_score(y, yh), Bal_acc=balanced_accuracy_score(y, yh), prevalence=y.mean())
rows, curves = [], {}
for name, m in fitted.items():
    pv, pt = prob(m, Xva), prob(m, Xte); thr = best_f1_thr(yva, pv)
    for split, y, p in (('validation', yva, pv), ('test', yte, pt)): rows.append(dict(model=name, split=split, threshold=thr, **cls_metrics(y, p, thr)))
    curves[name] = dict(roc=roc_curve(yte, pt), pr=precision_recall_curve(yte, pt), p=pt)
# baselines: persistence of last month's compound severity, and SPEI-3 memory only
for nm, col in (('Baseline: persistence (CDHI t-1)', 'cdhi_l1'), ('Baseline: SPEI-3 (t-1) alone', 'SPEI3_l1')):
    sgn = 1 if col == 'cdhi_l1' else -1
    for split, m in (('validation', va_m), ('test', te_m)):
        y = DF.loc[m, 'y'].values; p = sgn*DF.loc[m, col].values; rows.append(dict(model=nm, split=split, ROC_AUC=roc_auc_score(y, p), PR_AUC=average_precision_score(y, p), prevalence=y.mean()))
    curves[nm] = dict(roc=roc_curve(yte, sgn*DF.loc[te_m, col].values), pr=precision_recall_curve(yte, sgn*DF.loc[te_m, col].values))
CLS = pd.DataFrame(rows); CLS.round(4).to_csv(f'{TAB}/T8_classification_performance.csv', index=False); S['CLS'] = CLS
cand = {k: v for k, v in fitted.items() if k != 'Logistic regression'}
best_name = max(cand, key=lambda k: CLS[(CLS.model==k)&(CLS.split=='validation')].PR_AUC.values[0]); RM['best_model'] = best_name; bm = fitted[best_name]
print('Best (validation PR-AUC):', best_name); print(CLS[CLS.split=='test'].round(3).to_string())

# block bootstrap (resampling test years) for the best model and baselines
pt_best = curves[best_name]['p']; te_years = DF.loc[te_m, 'year'].values; uy = np.unique(te_years); ci = {'ROC_AUC': [], 'PR_AUC': []}
for _ in range(300):
    ys = rng.choice(uy, len(uy)); idx = np.concatenate([np.where(te_years==y_)[0] for y_ in ys])
    if yte[idx].sum() > 0: ci['ROC_AUC'].append(roc_auc_score(yte[idx], pt_best[idx])); ci['PR_AUC'].append(average_precision_score(yte[idx], pt_best[idx]))
RM['best_ci'] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in ci.items()}
# calibration
pv_, ys_ = pt_best, yte; bins = np.quantile(pv_, np.linspace(0, 1, 9)); bi = np.clip(np.digitize(pv_, bins[1:-1]), 0, 7)
calib = pd.DataFrame({'pred': [pv_[bi==b].mean() for b in range(8)], 'obs': [ys_[bi==b].mean() for b in range(8)]}); S['calib'] = calib

from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds='clip').fit(prob(bm, Xva), yva); pt_cal = iso.predict(prob(bm, Xte))
RM['brier_test_raw'] = float(brier_score_loss(yte, pt_best)); RM['brier_test_isotonic'] = float(brier_score_loss(yte, pt_cal)); RM['brier_test_climatology_val_prev'] = float(brier_score_loss(yte, np.full(len(yte), yva.mean())))
bi2 = np.clip(np.digitize(pt_cal, np.unique(np.quantile(pt_cal, np.linspace(0, 1, 6)))[1:-1]), 0, 4); S['calib_iso'] = pd.DataFrame({'pred': [pt_cal[bi2==b].mean() for b in np.unique(bi2)], 'obs': [yte[bi2==b].mean() for b in np.unique(bi2)]})
# ---------------------------------------------------------------- Experiment B: severity (CDHI) regression
RG = {'Ridge (linear baseline)': (lambda **p: make_pipeline(StandardScaler(), Ridge(**p)), dict(alpha=[.1,1,10,100])),
      'Random forest': (lambda **p: RandomForestRegressor(n_jobs=1, random_state=SEED, **p), dict(n_estimators=[200,400], max_depth=[4,6,8,12], min_samples_leaf=[5,10,20], max_features=['sqrt',0.5])),
      'Hist. gradient boosting': (lambda **p: HistGradientBoostingRegressor(random_state=SEED, early_stopping=False, **p), dict(learning_rate=[.02,.05,.1], max_iter=[100,200,300], max_depth=[2,3,4,6], min_samples_leaf=[10,20,40], l2_regularization=[0,1,10])),
      'Gradient boosting': (lambda **p: GradientBoostingRegressor(subsample=.7, random_state=SEED, **p), dict(n_estimators=[100,200,300], learning_rate=[.03,.05,.1], max_depth=[2,3,4], min_samples_leaf=[10,20,40]))}
ctr, cva, cte = DF.loc[tr_m,'cdhi'].values, DF.loc[va_m,'cdhi'].values, DF.loc[te_m,'cdhi'].values; regs, rrows = {}, []
def nse(y, p): return 1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)
for name, (ctor, grid) in RG.items():
    best = (1e9, None)
    for p in ParameterSampler(grid, n_iter=min(8, int(np.prod([len(v) for v in grid.values()]))), random_state=SEED):
        m = ctor(**p).fit(Xtr, ctr); r = mean_squared_error(cva, m.predict(Xva))**.5
        if r < best[0]: best = (r, p)
    m = ctor(**best[1]).fit(Xtr, ctr); regs[name] = m
    for split, X, y in (('validation', Xva, cva), ('test', Xte, cte)):
        p = m.predict(X); rrows.append(dict(model=name, split=split, RMSE=mean_squared_error(y, p)**.5, MAE=mean_absolute_error(y, p), R2=r2_score(y, p), NSE=nse(y, p), Bias=float(np.mean(p-y))))
for split, m in (('validation', va_m), ('test', te_m)):
    y = DF.loc[m,'cdhi'].values; p = DF.loc[m,'cdhi_l1'].values; rrows.append(dict(model='Baseline: persistence (CDHI t-1)', split=split, RMSE=mean_squared_error(y, p)**.5, MAE=mean_absolute_error(y, p), R2=r2_score(y, p), NSE=nse(y, p), Bias=float(np.mean(p-y))))
REG = pd.DataFrame(rrows); REG.round(4).to_csv(f'{TAB}/T8b_severity_regression_performance.csv', index=False); S['REG'] = REG
best_reg = min(regs, key=lambda k: REG[(REG.model==k)&(REG.split=='validation')].RMSE.values[0]) if False else min([k for k in regs if not k.startswith('Ridge')], key=lambda k: REG[(REG.model==k)&(REG.split=='validation')].RMSE.values[0])
RM['best_reg'] = best_reg; print(REG[REG.split=='test'].round(3).to_string())
S['reg_pred_te'] = pd.DataFrame({'obs': cte, 'pred': regs[best_reg].predict(Xte), 'year': DF.loc[te_m,'year'].values})

# ---------------------------------------------------------------- robustness of feature sets (Experiments 1-5)
FS = {'E1 Raw lags: rain + Tmax (t-1..t-3)': ['P_l1','P_l2','P_l3','Tx_an_l1','Tx_an_l2','Tx_an_l3'],
      'E2 E1 + SPI-3/SPEI-3 memory': ['P_l1','P_l2','P_l3','Tx_an_l1','Tx_an_l2','Tx_an_l3','SPI3_l1','SPEI3_l1'],
      'E3 Meteorological only (no static/season)': [f for f in FEATS if f not in ('Elev','Lat','Lon','MAP','sin_month','cos_month')],
      'E4 Meteorological + elevation/location + rain climatology': [f for f in FEATS if f not in ('sin_month','cos_month')],
      'E5 Full predictor set': FEATS,
      'E6 Full set without SPI/SPEI (no index-memory overlap)': [f for f in FEATS if f not in ('SPI3_l1','SPEI3_l1')]}
rob = []; hp = tuned['Hist. gradient boosting (LightGBM-type)']; ctor_h = MODELS['Hist. gradient boosting (LightGBM-type)'][0]
for nm, fs in FS.items():
    m = ctor_h(**hp).fit(DF.loc[tr_m, fs], ytr); pt = prob(m, DF.loc[te_m, fs]); pv = prob(m, DF.loc[va_m, fs])
    rob.append(dict(experiment=nm, n_features=len(fs), val_ROC_AUC=roc_auc_score(yva, pv), val_PR_AUC=average_precision_score(yva, pv), test_ROC_AUC=roc_auc_score(yte, pt), test_PR_AUC=average_precision_score(yte, pt)))
ROB = pd.DataFrame(rob); ROB.round(4).to_csv(f'{TAB}/T8d_feature_set_robustness.csv', index=False); S['ROB'] = ROB; print(ROB.round(3).to_string())

# lead-time experiment (predictors from month t-L only)
lead = []
for L in (1, 2, 3, 4, 6, 9, 12):
    rows_ = []
    for s in stations:
        d = pd.DataFrame(index=Pm.index); d['spi3'] = spi3[s].shift(L); d['spei3'] = spei3[s].shift(L); d['tx'] = Tx_an[s].shift(L); d['tx12'] = Tx_an[s].rolling(12).mean().shift(L)
        d['p'] = Pm[s].shift(L); d['T'] = clim.Elev[s]; d['sm'] = np.sin(2*np.pi*d.index.month/12); d['cm'] = np.cos(2*np.pi*d.index.month/12); d['y'] = CDHE[s].astype(int); d['year'] = d.index.year; rows_.append(d)
    D = pd.concat(rows_).dropna(); fs = ['spi3','spei3','tx','tx12','p','T','sm','cm']; a, b = D.year<=TRAIN_END, D.year>VAL_END
    m = ctor_h(**hp).fit(D.loc[a, fs], D.loc[a, 'y']); pt = prob(m, D.loc[b, fs]); lead.append(dict(lead_months=L, test_ROC_AUC=roc_auc_score(D.loc[b, 'y'], pt), test_PR_AUC=average_precision_score(D.loc[b, 'y'], pt)))
LEAD = pd.DataFrame(lead); LEAD.round(4).to_csv(f'{TAB}/T8e_lead_time_skill.csv', index=False); S['LEAD'] = LEAD; print(LEAD.round(3).to_string())

# ---------------------------------------------------------------- spatial transfer: leave-one-station-out (train on other stations <=2004, test held-out station 2005-2025)
loso = []
for s in stations:
    a = tr_m & (DF.station != s); b = (DF.station == s) & (DF.year > TRAIN_END)
    m = ctor_h(**hp).fit(DF.loc[a, FEATS], DF.loc[a, 'y']); p = prob(m, DF.loc[b, FEATS]); y = DF.loc[b, 'y'].values
    loso.append(dict(station=s, ROC_AUC=roc_auc_score(y, p), PR_AUC=average_precision_score(y, p), prevalence=y.mean()))
LOSO = pd.DataFrame(loso).set_index('station'); LOSO.round(4).to_csv(f'{TAB}/T8f_leave_one_station_out.csv'); S['LOSO'] = LOSO; print(LOSO.round(3).describe().loc[['mean','min','max']])

# ---------------------------------------------------------------- explainability: Shapley values (permutation-sampling estimator; identical target to SHAP), ALE, H-statistic
def shap_mc(predict, X, bg, nperm=24, seed=SEED):
    g = np.random.default_rng(seed); n, d = X.shape; phi = np.zeros((n, d)); base = predict(bg).mean()
    for i in range(n):
        perms = [g.permutation(d) for _ in range(nperm)]; zs = bg[g.integers(0, len(bg), nperm)]; rows_ = []
        for p, z in zip(perms, zs):
            cur = z.copy(); rows_.append(cur.copy())
            for j in p: cur[j] = X[i, j]; rows_.append(cur.copy())
        pr_ = predict(np.array(rows_)).reshape(nperm, d+1)
        for k, p in enumerate(perms): phi[i, p] += np.diff(pr_[k])
    return phi/nperm, base
XA = DF.loc[va_m | te_m, FEATS].reset_index(drop=True); ysel = DF.loc[va_m | te_m, 'y'].values
ix = np.r_[rng.choice(np.where(ysel==1)[0], min(150, int(ysel.sum())), replace=False), rng.choice(np.where(ysel==0)[0], 350, replace=False)]      # event-enriched explanation sample
bg = Xtr.values[rng.choice(len(Xtr), 120, replace=False)]; Xe = XA.values[ix]
f = lambda A: bm.predict_proba(pd.DataFrame(A, columns=FEATS))[:, 1]
t0 = time.time(); PHI, BASE = shap_mc(f, Xe, bg); print('Shapley time', time.time()-t0)
add_err = np.abs(PHI.sum(1)+BASE-f(Xe)).mean(); RM['shap_additivity_err'] = float(add_err)
imp = pd.Series(np.abs(PHI).mean(0), index=FEATS).sort_values(ascending=False); RM['shap_importance'] = imp.round(4).to_dict(); print(imp.round(4).head(10))
S.update(PHI=PHI, Xe=Xe, BASE=BASE, imp=imp)
# event-only SHAP
imp_ev = pd.Series(np.abs(PHI[:150]).mean(0), index=FEATS).sort_values(ascending=False); RM['shap_importance_events'] = imp_ev.round(4).to_dict()
top6 = imp.index[:6].tolist()
def pd_at(fn, X, cols):
    ci_ = [FEATS.index(c) for c in cols]; out = np.zeros(len(X))
    for i in range(len(X)):
        Z = X.copy(); Z[:, ci_] = X[i, ci_]; out[i] = fn(Z).mean()
    return out - out.mean()
Xh = XA.values[rng.choice(len(XA), 160, replace=False)]; pd1 = {c: pd_at(f, Xh, [c]) for c in top6}; H = pd.DataFrame(0.0, index=top6, columns=top6)
for a_, b_ in itertools.combinations(top6, 2):
    p2 = pd_at(f, Xh, [a_, b_]); H.loc[a_, b_] = H.loc[b_, a_] = np.sum((p2-pd1[a_]-pd1[b_])**2)/np.sum(p2**2)
S['H'] = H; RM['H_pairs'] = {f'{a_} x {b_}': float(H.loc[a_, b_]) for a_, b_ in itertools.combinations(top6, 2)}
def ale1d(fn, X, c, K=20):
    j = FEATS.index(c); q = np.unique(np.quantile(X[:, j], np.linspace(0, 1, K+1))); eff = []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (X[:, j] >= lo) & (X[:, j] <= hi)
        if m.sum() == 0: eff.append(0); continue
        A, B = X[m].copy(), X[m].copy(); A[:, j] = lo; B[:, j] = hi; eff.append((fn(B)-fn(A)).mean())
    ale = np.r_[0, np.cumsum(eff)]; ale -= np.average((ale[:-1]+ale[1:])/2, weights=np.maximum(np.histogram(X[:, j], q)[0], 1)); return q, ale
Xall = DF[FEATS].values; ALE = {c: ale1d(f, Xall[rng.choice(len(Xall), 3000, replace=False)], c) for c in top6}; S['ALE'] = ALE
# threshold detection from binned Shapley dependence: value where mean phi crosses zero, and steepest-rise location
thr = {}
for c in top6[:5]:
    j = FEATS.index(c); v = Xe[:, j]; ph = PHI[:, j]; qs = np.unique(np.quantile(v, np.linspace(0, 1, 13))); mid, mp = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (v >= lo) & (v <= hi); mid.append(v[m].mean()); mp.append(ph[m].mean())
    mid, mp = np.array(mid), np.array(mp); sl = np.gradient(mp, mid); dirn = np.sign(np.corrcoef(mid, mp)[0, 1]); sgn = np.where(np.diff(np.sign(mp)) != 0)[0]
    thr[c] = dict(direction='higher value -> higher risk' if dirn > 0 else 'higher value -> lower risk', zero_cross=float(np.interp(0, [mp[sgn[0]], mp[sgn[0]+1]][::1] if mp[sgn[0]] < mp[sgn[0]+1] else [mp[sgn[0]+1], mp[sgn[0]]], [mid[sgn[0]], mid[sgn[0]+1]] if mp[sgn[0]] < mp[sgn[0]+1] else [mid[sgn[0]+1], mid[sgn[0]]])) if len(sgn) else None,
                  steepest_at=float(mid[np.argmax(np.abs(sl))]), range_phi=float(mp.max()-mp.min()))
RM['thresholds'] = thr; print(thr)
# 2-D risk surface for the strongest interacting pair
pair = H.stack().sort_values(ascending=False).index[0]; RM['top_pair'] = list(pair)
ga, gb = [np.linspace(np.percentile(Xall[:, FEATS.index(c)], 2), np.percentile(Xall[:, FEATS.index(c)], 98), 25) for c in pair]; samp = Xall[rng.choice(len(Xall), 400, replace=False)]; surf = np.zeros((25, 25))
for i, a_ in enumerate(ga):
    for j2, b_ in enumerate(gb):
        Z = samp.copy(); Z[:, FEATS.index(pair[0])] = a_; Z[:, FEATS.index(pair[1])] = b_; surf[j2, i] = f(Z).mean()
S['surf'] = (pair, ga, gb, surf)

# ---------------------------------------------------------------- elevation(thermal)-zone-specific models + permutation importance
zone_rows, zimp = [], {}
zf = [c for c in FEATS if c not in ('Elev', 'Lat', 'Lon', 'MAP')]
for zn in ZONES:
    zs_ = clim.index[clim.Zone == zn]; a = tr_m & DF.station.isin(zs_); b = (~tr_m) & DF.station.isin(zs_); ya, yb = DF.loc[a, 'y'].values, DF.loc[b, 'y'].values
    if len(zs_) == 0 or ya.sum() < 5 or yb.sum() < 5 or len(np.unique(yb)) < 2:          # too few stations/events for a separate model
        zone_rows.append(dict(zone=zn, n_stations=len(zs_), n_events_train=int(ya.sum()), n_events_eval=int(yb.sum()), ROC_AUC=np.nan, PR_AUC=np.nan, prevalence=np.nan)); zimp[zn] = pd.Series(np.nan, index=zf); continue
    m = ctor_h(**hp).fit(DF.loc[a, zf], ya); p = prob(m, DF.loc[b, zf]); y = yb
    pi = permutation_importance(m, DF.loc[b, zf], y, scoring='average_precision', n_repeats=8, random_state=SEED)
    zimp[zn] = pd.Series(pi.importances_mean, index=zf); zone_rows.append(dict(zone=zn, n_stations=len(zs_), n_events_train=int(ya.sum()), n_events_eval=int(y.sum()), ROC_AUC=roc_auc_score(y, p), PR_AUC=average_precision_score(y, p), prevalence=y.mean()))
ZM = pd.DataFrame(zone_rows); ZM.round(4).to_csv(f'{TAB}/T8g_zone_specific_models.csv', index=False); S['ZM'] = ZM; ZI = pd.DataFrame(zimp); ZI.round(4).to_csv(f'{TAB}/T8h_zone_permutation_importance.csv'); S['ZI'] = ZI
print(ZM.round(3).to_string()); print(ZI.round(3).head(8))
# logistic-regression standardized coefficients (interpretable baseline)
lr = fitted['Logistic regression']; S['lr_coef'] = pd.Series(lr[-1].coef_[0], index=FEATS).sort_values()
S.update(FEATS=FEATS, NICE=NICE, RM=RM, curves=curves, DF=DF, best_name=best_name, fitted=fitted, regs=regs)
S['R'] = R; save_state({k: v for k, v in S.items() if k not in ('fitted', 'regs')})


# %% ===================== PART 4: FIGURES =====================
from scipy.cluster import hierarchy as sch
globals().update({k: v for k, v in S.items() if k not in globals()}); NS = len(stz)
FIGSZ.update({'Fig01_climatology': (10, max(8.8, 0.3*NS+2.4)), 'Fig02_QC_homogeneity': (10, 8.6), 'Fig03_trend_heatmap': (10.5, max(7.4, 0.25*NS+3.6)), 'Fig04_basin_timeseries': (9, 10.6), 'Fig05_CDHE_station_year_heatmap': (10, max(6.6, 0.2*NS+3.4)),
  'Fig06_frequency_periods_duration': (10, 8.4), 'Fig07_elevation_dependence': (9.5, 11), 'Fig08_change_points': (9.5, 11), 'Fig09_copula_dependence': (10, 8.8), 'Fig10_wavelets': (10, 10), 'Fig11_spatial_structure': (10, 8.6),
  'Fig12_event_definition_sensitivity': (8.5, 8.8), 'Fig13_ML_performance': (10, 9.2), 'Fig14_ML_diagnostics': (10, 9), 'Fig15_Shapley_explanation': (10, 9.4), 'Fig16_interactions_ALE': (10, 9.4), 'Fig17_zone_specific_drivers': (8.5, 9.6), 'Fig18_station_maps_elevation': (10, 13.6)})   # expose stored results (ext, extent, HOM, TR, ...) to the figure code
lab = lambda s: LBL.get(s, s); zc = lambda s: ZC[zone[s]]
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ext = S['ext']; YR = np.arange(1981, 2026); EPI = (2000.5, 2005.5)
def shade(ax): ax.axvspan(*EPI, color='#f4a582', alpha=.25, lw=0)
def letter(ax, t, x=-0.035): ax.text(x, 1.02, t, transform=ax.transAxes, fontweight='bold', fontsize=17, ha='right', va='bottom')
def repel_labels(ax, xs, ys, names, fs=10.5):
    """Annotate points with names, trying several offsets so that labels do not overlap each other."""
    ax.figure.canvas.draw(); rend = ax.figure.canvas.get_renderer(); placed = []
    offs = [(5, 5), (5, -15), (-6, 5), (-6, -15), (8, 18), (8, -26), (-8, 18), (-8, -26), (16, 0), (-16, 0), (10, 30), (-10, -34)]
    for x, y, n in sorted(zip(xs, ys, names), key=lambda t: -t[1]):
        ok = False
        for dx, dy in offs:
            t = ax.annotate(n, (x, y), xytext=(dx, dy), textcoords='offset points', fontsize=fs, ha='left' if dx >= 0 else 'right')
            bb = t.get_window_extent(rend)
            if not any(bb.overlaps(p) for p in placed): placed.append(bb); ok = True; break
            t.remove()
        if not ok: t = ax.annotate(n, (x, y), xytext=offs[0], textcoords='offset points', fontsize=fs); placed.append(t.get_window_extent(rend))
zone_handles = lambda: [Patch(facecolor=ZC[z], edgecolor='k', label=ZS[z]) for z in ZONES]
ZS = {ZONES[0]: 'Kolla', ZONES[1]: 'Weyna Dega', ZONES[2]: 'Dega'}

# ---- Fig 1 climatology
fig = plt.figure(figsize=(10, 8.8)); _gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1]); ax = [fig.add_subplot(_gs[:, 0]), fig.add_subplot(_gs[0, 1]), fig.add_subplot(_gs[1, 1])]
mr = pd.DataFrame({s: Pm[s].groupby(Pm.index.month).mean() for s in stz}).T; mr.columns = ['J','F','M','A','M','J','J','A','S','O','N','D']; mr.index = [lab(s) for s in mr.index]
sns.heatmap(mr, cmap='YlGnBu', ax=ax[0], cbar_kws={'label':'Mean monthly rainfall (mm)'}, annot=True, fmt='.0f', annot_kws={'size':9 if NS <= 18 else 7, 'weight':'bold'}, yticklabels=True); ax[0].set_title('Monthly rainfall regime'); ax[0].set_xlabel('Month'); ax[0].set_ylabel('Station (high → low elevation)'); letter(ax[0], 'a', x=-0.30)
for s in stations: ax[1].scatter(clim.Elev[s], clim.Rain_mm_yr[s], s=60, c=zc(s), edgecolor='k')
repel_labels(ax[1], [clim.Elev[s] for s in stations], [clim.Rain_mm_yr[s] for s in stations], [lab(s) for s in stations], fs=10.5 if NS <= 18 else 8.5)
ax[1].set_xlabel('Elevation (m a.s.l.)'); ax[1].set_ylabel('Mean annual rainfall (mm)'); ax[1].set_title('Elevation vs rainfall'); letter(ax[1], 'b')
for z, c in ZC.items(): ax[1].scatter([], [], c=c, edgecolor='k', label=ZS[z])
ax[1].legend(title='Agro-ecological zone', fontsize=11, loc='upper left')
o = clim.loc[stz]; ax[2].barh([lab(s) for s in o.index], o.Aridity_P_over_PET, color=[zc(s) for s in o.index]); ax[2].axvline(1, c='k', ls='--', lw=.8); ax[2].set_xlabel('Aridity ratio P / PET (Hargreaves)'); ax[2].set_title('Moisture balance'); letter(ax[2], 'c')
savefig('Fig01_climatology')

# ---- Fig 2 QC + homogeneity
fig = plt.figure(figsize=(10, 8.6)); _gs = fig.add_gridspec(2, 2); ax = [fig.add_subplot(_gs[0, 0]), fig.add_subplot(_gs[0, 1]), fig.add_subplot(_gs[1, :])]
raw_tx = pd.DataFrame({k: v.Tmax_C for k, v in raw.items()}); cm = raw_tx.corr(); sns.heatmap(cm, cmap='viridis', vmin=.8, vmax=1, ax=ax[0], cbar_kws={'label':'Pearson r (daily Tmax)'}, xticklabels=True, yticklabels=True); ax[0].tick_params(labelsize=10.5); ax[0].set_title('Daily Tmax correlation among files'); letter(ax[0], 'a')
h = HOM[(HOM.series=='relative')&(HOM['var']=='Tmean')].set_index('station').loc[stz]; ax[1].barh([lab(s) for s in h.index], h.snht_T, color=['#b2182b' if x else '#999' for x in h.snht_sig]); ax[1].axvline(R['snht_crit'], c='k', ls='--'); ax[1].set_xlabel('SNHT statistic (relative Tmean)'); ax[1].set_title('SNHT against a reference series'); letter(ax[1], 'b')
for i, (v, sr, c) in enumerate((('Tmean', 'raw', '#d6604d'), ('Tmean', 'relative', '#f4a582'), ('PRCP_total', 'raw', '#4393c3'), ('PRCP_total', 'relative', '#92c5de'))):
    d = HOM[(HOM['var']==v)&(HOM.series==sr)]; ax[2].scatter(d.pettitt_year, np.full(len(d), i)+rng.uniform(-.15, .15, len(d)), c=c, s=30, edgecolor='k', lw=.4)
ax[2].set_yticks(range(4)); ax[2].set_yticklabels(['Tmean (raw)', 'Tmean (relative)', 'Rain (raw)', 'Rain (relative)']); ax[2].set_xlabel('Pettitt break year (all stations)'); ax[2].set_title('Timing of detected shifts'); letter(ax[2], 'c'); shade(ax[2])
savefig('Fig02_QC_homogeneity')

# ---- Fig 3 trend heatmap
cols = ['PRCP_total','Wet_days','R10mm','R20mm','Rx1day','Rx5day','SDII','CDD','CWD','R95pTOT_pct','Tmax_mean','Tmin_mean','TXx','TX90p','TN90p','HW_days','WSDI','PET_total','Drought_months_SPI3','Hot_months_P90','CDHE_months']
T = TR[TR['var'].isin(cols)]; tau = T.pivot(index='station', columns='var', values='tau').loc[stz+['BASIN'], cols]; qq = T.assign(sig=np.where(T.station=='BASIN', T.p_mmk, T.q_fdr)).pivot(index='station', columns='var', values='sig').loc[stz+['BASIN'], cols]
fig, ax = plt.subplots(figsize=(16.8, 7.8)); annot = qq.apply(lambda c: c.map(lambda x: '*' if x < .05 else ''))
sns.heatmap(tau, cmap='RdBu_r', center=0, vmin=-.6, vmax=.6, annot=annot, fmt='', ax=ax, cbar_kws={'label':"Kendall's τ"}, annot_kws={'size':16, 'weight':'bold'})
ax.set_yticklabels([lab(s) for s in tau.index], rotation=0); ax.set_xlabel('* FDR-adjusted p < 0.05 (stations); modified-MK p < 0.05 (basin mean)'); ax.set_ylabel('Station'); ax.set_title('Trends in rainfall/temperature extremes and compound drought–heat metrics, 1981–2025 (Hamed–Rao modified MK)'); plt.xticks(rotation=60, ha='right'); ax.axhline(NS, c='k', lw=2)
savefig('Fig03_trend_heatmap')

# ---- Fig 4 basin time series
fig, ax = plt.subplots(4, 1, figsize=(14.4, 13.2), sharex=False)
a_ = ann['PRCP_total'].mean(axis=1); ax[0].bar(a_.index, a_-a_.mean(), color=np.where(a_>a_.mean(), '#4393c3', '#d6604d')); ax[0].plot(a_.index, (a_-a_.mean()).rolling(5, center=True).mean(), 'k'); ax[0].set_ylabel('Rainfall anomaly (mm/yr)'); ax[0].set_title(f'Basin-mean annual rainfall ({NS} stations)'); shade(ax[0]); letter(ax[0], 'a')
ax[1].plot(ann['Tmax_mean'].mean(axis=1).index, ann['Tmax_mean'].mean(axis=1)-ann['Tmax_mean'].mean(axis=1).loc[1981:2010].mean(), 'r', label='Tmax'); ax[1].plot(ann['Tmin_mean'].mean(axis=1).index, ann['Tmin_mean'].mean(axis=1)-ann['Tmin_mean'].mean(axis=1).loc[1981:2010].mean(), 'b', label='Tmin'); ax[1].axhline(0, c='k', lw=.5); ax[1].set_ylabel('Anomaly vs 1981–2010 (°C)'); ax[1].legend(); ax[1].set_title('Basin-mean temperature anomalies'); shade(ax[1]); letter(ax[1], 'b')
e = ext.groupby(ext.index.year).mean(); ax[2].plot(e.index, e.Drought, c='#d6604d', label='Drought (SPI-3 ≤ −1)'); ax[2].plot(e.index, e.Heat, c='#e08214', label='Hot month (Tmax ≥ P90)'); ax[2].fill_between(e.index, e.CDHE, color='k', alpha=.6, label='Compound drought–heat'); ax[2].set_ylabel('% station-months'); ax[2].legend(ncol=3, fontsize=12); ax[2].set_title('Frequency of drought, heat and compound events'); shade(ax[2]); letter(ax[2], 'c')
sp = spei3.mean(axis=1); ax[3].fill_between(sp.index, sp, 0, where=sp<0, color='#d6604d'); ax[3].fill_between(sp.index, sp, 0, where=sp>=0, color='#4393c3'); ax[3].plot(extent.index, -extent/25, c='k', lw=.8, label='Compound extent (−% stations / 25)'); ax[3].set_ylabel('SPEI-3 (basin mean)'); ax[3].legend(fontsize=12); ax[3].set_title('Basin-mean SPEI-3 and spatial extent of compound events'); letter(ax[3], 'd')
savefig('Fig04_basin_timeseries')

# ---- Fig 5 CDHE heatmap
M = ann['CDHE_months'].T.loc[stz]; fig, ax = plt.subplots(2, 2, figsize=(16.8, 8.4), gridspec_kw={'height_ratios':[1, 3.2], 'width_ratios':[6, 1]})
ax[0, 0].bar(YR, ann['CDHE_months'].mean(axis=1), color='k'); ax[0, 0].set_ylabel('Mean months / yr'); ax[0, 0].set_xlim(1980.5, 2025.5); ax[0, 0].set_title('Compound drought–heat months per year: basin mean and by station'); ax[0, 1].axis('off')
im = ax[1, 0].imshow(M.values, aspect='auto', cmap='YlOrRd', extent=[1980.5, 2025.5, len(stz), 0], vmin=0, vmax=8); ax[1, 0].set_yticks(np.arange(len(stz))+.5); ax[1, 0].set_yticklabels([lab(s) for s in stz], fontsize=12); plt.colorbar(im, ax=ax[1, 0], label='CDHE months in year', pad=.01, fraction=.03)
ax[1, 1].barh(np.arange(len(stz))+.5, M.sum(axis=1), color=[zc(s) for s in stz]); ax[1, 1].set_ylim(len(stz), 0); ax[1, 1].set_yticks([]); ax[1, 1].set_xlabel('Total months')
savefig('Fig05_CDHE_station_year_heatmap')

# ---- Fig 6 observed vs expected, periods, durations
fig, ax = plt.subplots(2, 2, figsize=(16.8, 10.8))
_ex, _ob = 100*COP.loc[stz].joint_p_indep, 100*COP.loc[stz].joint_p_empirical
for s_ in stz: ax[0, 0].scatter(_ex[s_], _ob[s_], c=zc(s_), edgecolor='k', s=55, zorder=3)
_xx = np.linspace(0, _ex.max()*1.15, 10)
for k_, ls_ in ((1, ':'), (3, '--'), (5, '-.')): ax[0, 0].plot(_xx, k_*_xx, 'k', ls=ls_, lw=1); ax[0, 0].text(_xx[-1], k_*_xx[-1], f'×{k_}', ha='left', va='center', fontsize=11)
ax[0, 0].set_xlim(0, _ex.max()*1.25); ax[0, 0].set_ylim(0, _ob.max()*1.15); ax[0, 0].set_xlabel('Expected if independent (% of months)'); ax[0, 0].set_ylabel('Observed compound (% of months)'); ax[0, 0].legend(handles=zone_handles(), title='Zone', loc='lower right', fontsize=10.5); ax[0, 0].set_title('Observed vs independence-expected'); letter(ax[0, 0], 'a')
lb = ['Drought', 'Hot month', 'Compound']; ea = [cmp.drought_pct_early, cmp.hot_pct_early, cmp.CDHE_pct_early]; la = [cmp.drought_pct_late, cmp.hot_pct_late, cmp.CDHE_pct_late]
for i in range(3): ax[0, 1].bar(i-.2, ea[i].mean(), .4, color='#92c5de'); ax[0, 1].bar(i+.2, la[i].mean(), .4, color='#d6604d'); ax[0, 1].scatter(np.full(NS, i-.2)+rng.uniform(-.07, .07, NS), ea[i], c='k', s=8); ax[0, 1].scatter(np.full(NS, i+.2)+rng.uniform(-.07, .07, NS), la[i], c='k', s=8)
ax[0, 1].set_xticks(range(3)); ax[0, 1].set_xticklabels(lb); ax[0, 1].set_ylabel('% of months'); ax[0, 1].set_title(f"1981–2003 (blue) vs 2004–2025 (red); dots = stations\nWilcoxon p: heat {R['wilcoxon_hot']:.4f}, drought {R['wilcoxon_drought']:.2f}, compound {R['wilcoxon_CDHE_late_vs_early']:.2f}", fontsize=13.5); letter(ax[0, 1], 'b')
for lbl, m, c in (('1981–2003', EV.year<=2003, '#92c5de'), ('2004–2025', EV.year>2003, '#d6604d')): ax[1, 0].hist(EV[m].duration, bins=np.arange(.5, 10.5), alpha=.7, color=c, label=f'{lbl} (n={m.sum()}, mean {EV[m].duration.mean():.1f} mo)')
ax[1, 0].set_xlabel('Event duration (consecutive months)'); ax[1, 0].set_ylabel('Number of station events'); ax[1, 0].legend(); ax[1, 0].set_title('Event duration'); letter(ax[1, 0], 'c')
ep = R['episode']; ax[1, 1].bar(['Compound', 'Hot month', 'Drought'], [ep['CDHE_pct_inside'], ep['hot_pct_inside'], ep['drought_pct_inside']], .4, align='edge', color='#b2182b', label='2001–2005'); ax[1, 1].bar(['Compound', 'Hot month', 'Drought'], [ep['CDHE_pct_outside'], ep['hot_pct_outside'], ep['drought_pct_outside']], -.4, align='edge', color='#4393c3', label='All other years')
ax[1, 1].legend(); ax[1, 1].set_ylabel('% of station-months'); ax[1, 1].set_title(f"One episode dominates: 2001–2005 = 11% of time but {ep['share_CDHE_in_2001_2005_pct']:.0f}% of compound months", fontsize=13.5); letter(ax[1, 1], 'd')
savefig('Fig06_frequency_periods_duration')

# ---- Fig 7 elevation dependence
mets = ['Tmin trend (°C/decade)', 'Tmax trend (°C/decade)', 'TX90p trend (% of days/decade)', 'Rainfall trend (%/decade)', 'Kendall tau (dryness, heat)', 'Compound frequency (% of months)']
fig, ax = plt.subplots(3, 2, figsize=(9.5, 11)); ax = ax.ravel()
for a_, c in zip(ax, mets):
    yv = E2[c].reindex(stations)
    for s in stations: a_.scatter(clim.Elev[s], yv[s], c=zc(s), edgecolor='k', s=45)
    b = np.polyfit(clim.Elev.reindex(stations), yv, 1); xx = np.linspace(clim.Elev.min(), clim.Elev.max(), 10); a_.plot(xx, np.polyval(b, xx), 'k--', lw=1)
    r_ = R['elev_ext'][c]; a_.set_title(f"ρ = {r_['rho']:.2f}, p {'< 0.001' if r_['p'] < 0.001 else '= %.2f' % r_['p']}, q = {r_['q_fdr']:.2f}", fontsize=13); a_.set_xlabel('Elevation (m a.s.l.)'); a_.set_ylabel(c)
for a_, t_ in zip(ax, 'abcdef'): letter(a_, t_)
ax[4].legend(handles=zone_handles() + [Line2D([0], [0], color='k', ls='--', label='Linear fit')], title='Agro-ecological zone', fontsize=10.5, loc='lower left')
savefig('Fig07_elevation_dependence')


# ---- Fig 8 change points
vs = [('CDHE_months', 'Compound months / yr'), ('Hot_months_P90', 'Hot months / yr'), ('Drought_months_SPI3', 'Drought months / yr'), ('Tmax_mean', 'Tmax (°C)'), ('PRCP_total', 'Rainfall (mm)')]
fig, ax = plt.subplots(3, 2, figsize=(9.5, 11)); axf = ax.ravel()
for a_, (v, nm) in zip(axf, vs):
    sr = ann[v].mean(axis=1).dropna(); c = CP[(CP['var']==v)&(CP.station=='BASIN')].iloc[0]; a_.plot(sr.index, sr, 'k-o', ms=3, lw=.8); a_.hlines(c.pre, sr.index[0], c.year, colors='#2166ac', lw=2.5); a_.hlines(c.post, c.year+1, sr.index[-1], colors='#b2182b', lw=2.5)
    a_.axvline(c.year+.5, c='gray', ls=':'); _ps = '<0.001' if c.p < 0.001 else '=%.3f' % c.p; a_.set_title(f'{nm}: Pettitt {int(c.year)} (p{_ps})\npre {c.pre:.2f} → post {c.post:.2f}', fontsize=13.5); shade(a_)
d = CP[CP.station!='BASIN']
for i, (v, nm) in enumerate(vs): sub = d[d['var']==v]; axf[5].scatter(sub.year, np.full(len(sub), i)+rng.uniform(-.15, .15, len(sub)), c=np.where(sub.p<.05, '#b2182b', '#aaa'), s=28, edgecolor='k', lw=.3)
axf[5].set_yticks(range(len(vs))); axf[5].set_yticklabels([n for _, n in vs], fontsize=12); axf[5].set_xlabel('Change-point year (red: p < 0.05)'); shade(axf[5])
savefig('Fig08_change_points')

# ---- Fig 9 copulas
fig, ax = plt.subplots(2, 2, figsize=(15.6, 10.8)); cc = {'Gaussian':'#4393c3', 'Gumbel':'#d6604d', 'Frank':'#5aae61', 'Clayton':'#8073ac'}
o = COP.loc[stz]; _e = clim.Elev.reindex(stz)
for s_ in stz: ax[0, 0].scatter(_e[s_], o.kendall_tau[s_], c=cc[o.best[s_]], edgecolor='k', s=55)
ax[0, 0].set_ylabel("Kendall τ (dryness, Tmax anomaly)")
ax[0, 0].legend(handles=[Patch(facecolor=c, label=k) for k, c in cc.items() if k in set(o.best)], loc='upper right', ncol=3, frameon=True, fontsize=10.5, title='Best-fit copula (AIC)'); ax[0, 0].set_ylim(o.kendall_tau.min()*0.8, o.kendall_tau.max()*1.2); ax[0, 0].set_xlabel('Elevation (m a.s.l.)'); ax[0, 0].set_title('Dependence between drought severity and heat'); letter(ax[0, 0], 'a')
ax[0, 1].errorbar(_e.values, 100*o.joint_p_copula, yerr=[100*(o.joint_p_copula-o.ci_lo), 100*(o.ci_hi-o.joint_p_copula)], fmt='o', c='#b2182b', label='Copula joint probability (95% bootstrap CI)'); ax[0, 1].plot(_e.values, 100*o.joint_p_indep, 's', c='#666', label='Independence'); ax[0, 1].set_ylabel('% of months'); ax[0, 1].set_xlabel('Elevation (m a.s.l.)'); ax[0, 1].legend(loc='upper right', frameon=True, fontsize=11); ax[0, 1].set_ylim(0, max(100*o.ci_hi.max(), 1)*1.38); ax[0, 1].set_title('Joint probability of SPI-3 ≤ −1 and Tmax ≥ P90'); letter(ax[0, 1], 'b')
bp = ax[1, 0].boxplot([SEASTAU.loc[stz, c].values for c in SEASTAU.columns], labels=list(SEASTAU.columns), patch_artist=True, medianprops=dict(color='k', lw=2))
for p_, c_ in zip(bp['boxes'], ['#92c5de', '#4393c3', '#2166ac']): p_.set_facecolor(c_); ax[1, 0].set_ylabel("Kendall τ (dryness, Tmax anomaly)"); ax[1, 0].set_xlabel('Season (all stations)'); ax[1, 0].set_title('Seasonal dependence (Belg FMAM, Kiremt JJAS, Bega ONDJ)'); letter(ax[1, 0], 'c')
s0 = 'BahirDar'; dd = pd.DataFrame({'D': -spi3[s0], 'H': Tx_z[s0], 'c': CDHE[s0]}).dropna(); ax[1, 1].scatter(dd.D, dd.H, c=np.where(dd.c, '#b2182b', '#bbb'), s=14, edgecolor='none'); ax[1, 1].axvline(1, c='k', ls='--', lw=.8); ax[1, 1].set_xlabel('Drought severity (−SPI-3)'); ax[1, 1].set_ylabel('Monthly Tmax anomaly (z-score)'); ax[1, 1].set_title(f"Bahir Dar: best copula = {COP.best[s0]}, τ = {COP.kendall_tau[s0]:.2f}", fontsize=13.5); ax[1, 1].legend(handles=[Line2D([0], [0], marker='o', ls='', color='#b2182b', label='Compound month'), Line2D([0], [0], marker='o', ls='', color='#bbbbbb', label='Other month'), Line2D([0], [0], color='k', ls='--', label='Drought threshold (SPI-3 = −1)')], loc='upper left', fontsize=10.5); letter(ax[1, 1], 'd')
savefig('Fig09_copula_dependence')

# ---- Fig 10 wavelets
fig, ax = plt.subplots(3, 2, figsize=(16.8, 13.2), gridspec_kw={'width_ratios':[4, 1]})
tt = 1981+np.arange(540)/12
for i, (nm, (pr_, pw, sg, coi, gws)) in enumerate(WV.items()):
    a_ = ax[i, 0]; a_.contourf(tt, np.log2(pr_), np.log2(np.clip(pw, 0.05, None)), 14, cmap='turbo'); a_.contour(tt, np.log2(pr_), sg, [1], colors='k', linewidths=1.2)
    a_.set_yticks(np.log2([.25, .5, 1, 2, 4, 8, 16])); a_.set_yticklabels(['0.25', '0.5', '1', '2', '4', '8', '16']); a_.invert_yaxis(); a_.set_ylabel('Period (yr)'); a_.set_title(nm+' – Morlet power (black: 95% vs AR1 red noise)', fontsize=13.5)
    ax[i, 1].plot(gws, np.log2(pr_), 'k'); ax[i, 1].invert_yaxis(); ax[i, 1].set_yticks(np.log2([.25, .5, 1, 2, 4, 8, 16])); ax[i, 1].set_yticklabels(['0.25', '0.5', '1', '2', '4', '8', '16']); ax[i, 1].set_title('Global spectrum', fontsize=13.5)
savefig('Fig10_wavelets')

# ---- Fig 11 EOF / clusters / extent
fig, ax = plt.subplots(2, 2, figsize=(16.8, 10.8)); ax = ax.ravel()
for s_ in stz: ax[0].scatter(clim.Elev[s_], load.loc[s_, 'EOF1']*np.sign(load.EOF1.mean()), c=zc(s_), edgecolor='k', s=55)
ax[0].set_xlabel('Elevation (m a.s.l.)'); ax[0].set_ylabel('EOF1 loading'); ax[0].set_title(f"EOF1 of SPEI-3 loadings ({R['eof_var'][0]}% variance): basin-wide coherent mode"); letter(ax[0], 'a')
sch.dendrogram(LK, labels=[lab(s).split(' (')[0] for s in stations], ax=ax[1], leaf_rotation=90, leaf_font_size=12, color_threshold=LK[-3, 2]); ax[1].set_title('Hierarchical clusters (rainfall regime + Tmax/Tmin/MAP)'); letter(ax[1], 'b')
ax[2].plot(PCs.index, PCs.PC1*np.sign(load.EOF1.mean()), 'k', lw=.8); ax[2].axvspan(pd.Timestamp('2001-01-01'), pd.Timestamp('2005-12-31'), color='#f4a582', alpha=.3); ax[2].set_title('PC1 of SPEI-3'); ax[2].axhline(0, c='k', lw=.4); letter(ax[2], 'c')
ax[3].fill_between(ext.index, ext.Drought, color='#d6604d', alpha=.5, label='Drought'); ax[3].fill_between(ext.index, ext.Heat, color='#e08214', alpha=.4, label='Heat'); ax[3].plot(ext.index, ext.CDHE, 'k', lw=.8, label='Compound'); ax[3].set_ylabel('% of stations affected'); ax[3].legend(); ax[3].set_title('Spatial extent (synchrony)'); letter(ax[3], 'd')
savefig('Fig11_spatial_structure')

# ---- Fig 12 sensitivity
fig, ax = plt.subplots(3, 1, figsize=(8.5, 8.8)); y = np.arange(len(SENS))
ax[0].barh(y, SENS.mean_pct_months, color='#666'); ax[0].set_yticks(y); ax[0].set_yticklabels(SENS.index, fontsize=12); ax[0].set_xlabel('Mean % of months in compound state'); ax[0].invert_yaxis()
ax[1].barh(y, SENS.basin_slope_months_per_decade, color=np.where(SENS.basin_p<.05, '#b2182b', '#aaa')); ax[1].set_yticks(y); ax[1].set_yticklabels(SENS.index, fontsize=12); ax[1].set_ylabel('Event definition'); ax[1].set_xlabel('Basin trend (months/yr per decade; red p<0.05)'); ax[1].invert_yaxis()
ax[2].barh(y, SENS.early_late_ratio, color='#4393c3'); ax[2].axvline(1, c='k', ls='--'); ax[2].set_yticks(y); ax[2].set_yticklabels(SENS.index, fontsize=12); ax[2].set_ylabel('Event definition'); ax[2].set_xlabel('Frequency ratio, 2004–2025 / 1981–2003'); ax[2].invert_yaxis()
for a_, t_, tt_ in zip(ax, 'abc', ['Compound frequency', 'Basin trend', 'Change between periods']): a_.set_title(tt_); letter(a_, t_)
ax[0].set_ylabel('Event definition'); ax[1].legend(handles=[Patch(facecolor='#b2182b', label='p < 0.05'), Patch(facecolor='#aaaaaa', label='not significant')], loc='lower right', fontsize=10.5)
savefig('Fig12_event_definition_sensitivity')

# ---- Fig 13 ML performance
cmap_m = {'Logistic regression':'#666666', 'Random forest':'#5aae61', 'Hist. gradient boosting (LightGBM-type)':'#d6604d', 'Gradient boosting (XGBoost-type)':'#8073ac',
          'XGBoost':'#762a83', 'LightGBM':'#b35806', 'CatBoost':'#01665e', 'Baseline: persistence (CDHI t-1)':'#e08214', 'Baseline: SPEI-3 (t-1) alone':'#4393c3'}
for _i, _nm in enumerate([n for n in curves if n not in cmap_m]): cmap_m[_nm] = plt.cm.tab10(_i % 10)       # any other model name gets a default colour
fig, ax = plt.subplots(2, 2, figsize=(15.6, 12))
for nm, cv in curves.items(): ax[0, 0].plot(cv['roc'][0], cv['roc'][1], c=cmap_m[nm], ls='--' if nm.startswith('Base') else '-', label=nm)
ax[0, 0].plot([0, 1], [0, 1], 'k:', lw=.7); ax[0, 0].set_xlabel('False-positive rate'); ax[0, 0].set_ylabel('True-positive rate'); ax[0, 0].legend(fontsize=10.5); ax[0, 0].set_title('ROC – independent test 2015–2025'); letter(ax[0, 0], 'a')
for nm, cv in curves.items(): ax[0, 1].plot(cv['pr'][1], cv['pr'][0], c=cmap_m[nm], ls='--' if nm.startswith('Base') else '-')
ax[0, 1].axhline(RM['split']['test']['prevalence_pct']/100, c='k', ls=':', label='No-skill (prevalence)'); ax[0, 1].set_xlabel('Recall'); ax[0, 1].set_ylabel('Precision'); ax[0, 1].legend(fontsize=10.5); ax[0, 1].set_title('Precision–recall – independent test'); letter(ax[0, 1], 'b')
t_ = CLS[CLS.split=='test'].set_index('model'); xx = np.arange(len(t_)); ax[1, 0].bar(xx-.2, t_.ROC_AUC, .4, label='ROC-AUC', color='#4393c3'); ax[1, 0].bar(xx+.2, t_.PR_AUC, .4, label='PR-AUC', color='#d6604d'); ax[1, 0].set_xticks(xx); ax[1, 0].set_xticklabels([n.replace('Baseline: ', 'BL: ').replace(' (LightGBM-type)', '\n(LGBM-type)').replace(' (XGBoost-type)', '\n(XGB-type)') for n in t_.index], rotation=35, ha='right', fontsize=10.5); ax[1, 0].set_ylim(0, 1.18); ax[1, 0].legend(ncol=2, loc='upper left'); ax[1, 0].set_title(f"Test skill (PR-AUC no-skill = {RM['split']['test']['prevalence_pct']/100:.3f})"); letter(ax[1, 0], 'c')
rp = S['reg_pred_te']; ax[1, 1].scatter(rp.obs, rp.pred, s=6, alpha=.4, c='k'); lim = [rp.obs.min(), rp.obs.max()]; ax[1, 1].plot(lim, lim, 'r--'); ax[1, 1].set_xlabel('Observed CDHI'); ax[1, 1].set_ylabel('Predicted CDHI'); rr = REG[(REG.split=='test')&(REG.model==RM['best_reg'])].iloc[0]; ax[1, 1].set_title(f"Severity regression ({RM['best_reg']}): test R² = {rr.R2:.2f}, RMSE = {rr.RMSE:.2f}", fontsize=13.5); letter(ax[1, 1], 'd')
savefig('Fig13_ML_performance')

# ---- Fig 14 ML diagnostics
fig, ax = plt.subplots(2, 2, figsize=(15.6, 10.8))
ax[0, 0].plot(calib.pred, calib.obs, 'o-', c='#d6604d', label='Raw (class-weighted) probabilities'); ci_ = S['calib_iso']; ax[0, 0].plot(ci_.pred, ci_.obs, 's-', c='#4393c3', label='Isotonic-recalibrated on validation'); ax[0, 0].plot([0, .6], [0, .6], 'k:'); ax[0, 0].set_xlabel('Predicted probability'); ax[0, 0].set_ylabel('Observed frequency (test)'); ax[0, 0].legend(fontsize=10.5, loc='upper left'); ax[0, 0].set_title(f"Reliability ({RM['best_model']}); Brier raw {RM['brier_test_raw']:.3f} → {RM['brier_test_isotonic']:.3f}", fontsize=13.5); letter(ax[0, 0], 'a')
r_ = ROB.set_index('experiment'); xx = np.arange(len(r_)); ax[0, 1].barh(xx-.2, r_.test_ROC_AUC, .4, color='#4393c3', label='ROC-AUC'); ax[0, 1].barh(xx+.2, r_.test_PR_AUC, .4, color='#d6604d', label='PR-AUC'); ax[0, 1].set_yticks(xx); import textwrap; ax[0, 1].set_yticklabels([textwrap.fill(e, 36) for e in r_.index], fontsize=10.5); ax[0, 1].invert_yaxis(); ax[0, 1].set_xlim(0, 1.3); ax[0, 1].set_xticks([0, .2, .4, .6, .8, 1.0]); ax[0, 1].set_xlabel('Score on independent test period'); ax[0, 1].legend(loc='center right', frameon=True); ax[0, 1].set_title('Feature-set robustness (test)'); letter(ax[0, 1], 'b')
ax[1, 0].plot(LEAD.lead_months, LEAD.test_ROC_AUC, 'o-', c='#4393c3', label='ROC-AUC'); ax[1, 0].plot(LEAD.lead_months, LEAD.test_PR_AUC, 's-', c='#d6604d', label='PR-AUC'); ax[1, 0].axhline(.5, c='k', ls=':'); ax[1, 0].set_xlabel('Lead time of predictors (months)'); ax[1, 0].set_ylabel('Test score'); ax[1, 0].set_title('Skill decay with lead time'); ax[1, 0].legend(loc='upper right'); letter(ax[1, 0], 'c')
l_ = LOSO.loc[stz]
for s_ in stz: ax[1, 1].scatter(clim.Elev[s_], l_.ROC_AUC[s_], c=zc(s_), edgecolor='k', s=55)
ax[1, 1].set_ylim(.5, 1.12); ax[1, 1].set_ylabel('ROC-AUC (held-out station, 2005–2025)'); ax[1, 1].set_xlabel('Elevation of held-out station (m a.s.l.)'); ax[1, 1].legend(handles=zone_handles(), title='Zone', loc='upper center', ncol=3, fontsize=10.5); ax[1, 1].set_title('Spatial transferability: leave-one-station-out'); letter(ax[1, 1], 'd')
savefig('Fig14_ML_diagnostics')

# ---- Fig 15 SHAP-type explanation
top = imp.index[:10].tolist(); fig, ax = plt.subplots(2, 2, figsize=(16.8, 12)); ax = ax.ravel()
ax[0].barh([NICE[c] for c in imp.index[:12]][::-1], imp.values[:12][::-1], color='#d6604d'); ax[0].set_xlabel('mean |Shapley value|  (probability scale)'); ax[0].set_title(f"Global importance – {RM['best_model']}"); letter(ax[0], 'a')
for k, c in enumerate(top[::-1]):
    j = FEATS.index(c); v = Xe[:, j]; z = (v-np.percentile(v, 2))/(np.percentile(v, 98)-np.percentile(v, 2)+1e-9); ax[1].scatter(PHI[:, j], np.full(len(v), k)+rng.uniform(-.25, .25, len(v)), c=np.clip(z, 0, 1), cmap='coolwarm', s=6)
ax[1].set_yticks(range(10)); ax[1].set_yticklabels([NICE[c] for c in top[::-1]], fontsize=12); ax[1].axvline(0, c='k', lw=.5); ax[1].set_xlabel('Shapley value (Δ probability)'); ax[1].set_title('Beeswarm (red = high feature value)'); letter(ax[1], 'b')
for a_, c, part, tag in ((ax[2], 'SPEI3_l1', 'Tx_an_12m_l1', 'c'), (ax[3], 'Tx_an_12m_l1', 'SPEI3_l1', 'd')):
    j, jp = FEATS.index(c), FEATS.index(part); sc = a_.scatter(Xe[:, j], PHI[:, j], c=Xe[:, jp], cmap='coolwarm', s=8); _cb = plt.colorbar(sc, ax=a_, pad=0.03); _cb.set_label(NICE[part], labelpad=12); qs = np.unique(np.quantile(Xe[:, j], np.linspace(0, 1, 14))); mid = [(Xe[(Xe[:, j]>=lo)&(Xe[:, j]<=hi), j]).mean() for lo, hi in zip(qs[:-1], qs[1:])]; mp = [PHI[(Xe[:, j]>=lo)&(Xe[:, j]<=hi), j].mean() for lo, hi in zip(qs[:-1], qs[1:])]
    a_.plot(mid, mp, 'k-', lw=2); a_.axhline(0, c='k', lw=.5); a_.set_xlabel(NICE[c]); a_.set_ylabel('Shapley value'); a_.set_title('Dependence: '+NICE[c]); letter(a_, tag)
savefig('Fig15_Shapley_explanation')

# ---- Fig 16 interactions + ALE
fig, ax = plt.subplots(2, 2, figsize=(15.6, 12)); sns.heatmap(H.rename(index=NICE, columns=NICE), annot=True, fmt='.3f', cmap='Reds', ax=ax[0, 0], cbar_kws={'label':"Friedman's H²"}); ax[0, 0].set_title('Pairwise interaction strength (top-6 drivers)'); letter(ax[0, 0], 'a')
pair, ga, gb, surf = S['surf']; im = ax[0, 1].contourf(ga, gb, surf, 14, cmap='YlOrRd'); plt.colorbar(im, ax=ax[0, 1], label='Mean predicted P(compound)'); ax[0, 1].set_xlabel(NICE[pair[0]]); ax[0, 1].set_ylabel(NICE[pair[1]]); ax[0, 1].set_title('Risk surface for strongest interaction'); letter(ax[0, 1], 'b')
for a_, c, tag in ((ax[1, 0], 'SPEI3_l1', 'c'), (ax[1, 1], 'Tx_an_12m_l1', 'd')):
    q, al = ALE[c]; a_.step(q, al, where='post', c='#d6604d', lw=2) if False else a_.plot((q[:-1]+q[1:])/2, (al[:-1]+al[1:])/2, 'o-', c='#d6604d'); a_.axhline(0, c='k', lw=.5); a_.set_xlabel(NICE[c]); a_.set_ylabel('ALE (Δ probability)'); a_.set_title('Accumulated local effect: '+NICE[c]); letter(a_, tag)
savefig('Fig16_interactions_ALE')

# ---- Fig 17 zone-specific models
fig, ax = plt.subplots(2, 1, figsize=(8.5, 9.6)); zi = ZI.loc[ZI.abs().max(axis=1).sort_values(ascending=False).index[:8]]; zi = zi.rename(index=NICE); zi.columns = [ZS[c] for c in zi.columns]; zi.plot.barh(ax=ax[0], color=['#b2182b', '#999999', '#2166ac']); ax[0].invert_yaxis(); ax[0].set_xlabel('Permutation importance (ΔPR-AUC)'); ax[0].set_title('Driver importance by elevation zone (2005–2025)'); ax[0].set_ylabel('Predictor'); ax[0].legend(title='Zone', loc='lower right', fontsize=10.5); letter(ax[0], 'a')
zm = ZM.set_index('zone'); xx = np.arange(3); ax[1].bar(xx-.2, zm.ROC_AUC, .4, label='ROC-AUC', color='#4393c3'); ax[1].bar(xx+.2, zm.PR_AUC, .4, label='PR-AUC', color='#d6604d'); ax[1].set_xticks(xx); ax[1].set_xticklabels([ZS[z] for z in zm.index]); ax[1].set_ylabel('Score (evaluation period)'); ax[1].set_xlabel('Agro-ecological zone'); ax[1].set_ylim(0, 1.12); ax[1].legend(loc='upper right', ncol=2); ax[1].set_title('Zone-specific model skill (train ≤2004, evaluate 2005–2025)'); letter(ax[1], 'b')
savefig('Fig17_zone_specific_drivers')

# ---- Fig 18 station maps: interpolated surfaces clipped to the basin (coordinates from ElCoord.csv)
from matplotlib.path import Path as MPath
fig, ax = plt.subplots(3, 2, figsize=(10, 13.6)); axf = ax.ravel()
kept = set(stations); allc = COORD_ALL.dropna(subset=['lon', 'lat', 'elev'])

# --- basin boundary: set UBN_BOUNDARY to your shapefile/GeoJSON, or put a .shp / .geojson in the data folder
BASIN = os.environ.get('UBN_BOUNDARY', r'D:/4thOBJ/UBN2.shp')        # basin shapefile used to clip the maps and draw the outline
def load_boundary():
    if not os.path.exists(BASIN): raise FileNotFoundError(f'Basin shapefile not found: {BASIN} (edit BASIN above or set UBN_BOUNDARY)')
    import geopandas as gpd                                           # pip install geopandas
    gdf = gpd.read_file(BASIN)
    if gdf.crs is None: gdf = gdf.set_crs(4326)                       # assume lon/lat if the .prj file is missing
    gdf = gdf.to_crs(4326)                                            # maps are drawn in lon/lat to match the station coordinates
    g = gdf.geometry.union_all() if hasattr(gdf.geometry, 'union_all') else gdf.geometry.unary_union
    polys = []
    for p in (g.geoms if hasattr(g, 'geoms') else [g]):
        if p.geom_type == 'Polygon': polys.append(np.asarray(p.exterior.coords)[:, :2])
    if not polys: raise ValueError(f'No polygon geometry found in {BASIN}')
    print('  [Fig18] basin boundary read from', BASIN, '| polygons:', len(polys), '| bounds (lon/lat):', np.round(g.bounds, 2))
    return polys
BND = load_boundary(); _allxy = np.vstack(BND)
GX = np.linspace(_allxy[:, 0].min(), _allxy[:, 0].max(), 320); GY = np.linspace(_allxy[:, 1].min(), _allxy[:, 1].max(), 320); GXX, GYY = np.meshgrid(GX, GY)
INSIDE = np.zeros(GXX.shape, bool)
for p_ in BND: INSIDE |= MPath(p_).contains_points(np.c_[GXX.ravel(), GYY.ravel()]).reshape(GXX.shape)

def idw(x, y, v, power=2.0):                                         # inverse-distance weighting on the grid
    x, y, v = [np.asarray(a, float) for a in (x, y, v)]; ok = np.isfinite(v); x, y, v = x[ok], y[ok], v[ok]
    d2 = (GXX[..., None]-x)**2 + (GYY[..., None]-y)**2 + 1e-12; w = d2**(-power/2); return (w*v).sum(-1)/w.sum(-1)

def imap(a_, x, y, v, cmap, lbl, title, tag, center=None, nlev=10, mark=None):
    v = np.asarray(v, float); Z = np.where(INSIDE, idw(x, y, v), np.nan)
    lo, hi = (-np.nanmax(abs(v)), np.nanmax(abs(v))) if center is not None else (np.nanmin(v), np.nanmax(v))
    lev = np.linspace(lo, hi, nlev+1); cs = a_.contourf(GXX, GYY, Z, levels=lev, cmap=cmap, extend='both', zorder=1)
    for p_ in BND: a_.plot(p_[:, 0], p_[:, 1], 'k-', lw=1, zorder=2)
   # a_.scatter(x, y, s=22, c='k', edgecolor='white', lw=.6, zorder=4)
    if mark is not None: a_.scatter(mark[0], mark[1], marker='x', s=45, c='k', zorder=5)
    cb = plt.colorbar(cs, ax=a_, label=lbl, ticks=lev[::2] if nlev > 8 else lev, pad=0.03); cb.ax.tick_params(labelsize=11)
    a_.set_title(title); a_.set_aspect('equal'); letter(a_, tag)

_drp = COORD.loc[[n for n in COORD.index if n not in kept]]
imap(axf[0], allc.lon.values, allc.lat.values, allc.elev.values, 'terrain', 'Elevation (m a.s.l.)', 'Stations and elevation', 'a', mark=(_drp.lon.values, _drp.lat.values))
imap(axf[1], clim.Lon[stations].values, clim.Lat[stations].values, clim.Rain_mm_yr[stations].values, 'YlGnBu', 'Mean annual rainfall (mm)', 'Rainfall climatology', 'b')
imap(axf[2], clim.Lon[stations].values, clim.Lat[stations].values, ed.CDHE_pct_total.reindex(stations).values, 'RdYlBu_r', 'Compound months (%)', 'Compound drought–heat frequency', 'c')
imap(axf[3], clim.Lon[stations].values, clim.Lat[stations].values, ed.Tmax_trend_per_dec.reindex(stations).values, 'RdBu_r', 'Tmax trend (°C/decade)', 'Tmax trend', 'd', center=0)
for a_ in axf[:4]: a_.set_xlabel('Longitude (°E)'); a_.set_ylabel('Latitude (°N)')
for s in stations: axf[4].scatter(clim.Elev[s], clim.Tmean[s], c=zc(s), edgecolor='k', s=70)
xx = np.linspace(clim.Elev.min(), clim.Elev.max(), 10); bb = np.polyfit(clim.Elev, clim.Tmean, 1); axf[4].plot(xx, np.polyval(bb, xx), 'k--')
axf[4].set_xlabel('Elevation (m a.s.l.)'); axf[4].set_ylabel('Mean temperature (°C)'); axf[4].set_title(f"Lapse rate {R['lapse']['Tmean_C_per_km']:.1f} °C/km\n(r = {R['lapse']['r']:.2f})"); letter(axf[4], 'e')
axf[4].legend(handles=[Patch(facecolor=c, edgecolor='k', label=ZS[z]) for z, c in ZC.items()], loc='upper right', fontsize=11)
for s in stations: axf[5].scatter(clim.Elev[s], ed.Kendall_tau_DH[s], c=zc(s), edgecolor='k', s=70)
axf[5].set_xlabel('Elevation (m a.s.l.)'); axf[5].set_ylabel('Kendall τ (dryness, Tmax anomaly)'); axf[5].set_title(f"Coupling vs elevation\n(ρ = {R['spearman_proxy']['Kendall_tau_DH']['rho']:.2f}, p = {R['spearman_proxy']['Kendall_tau_DH']['p']:.2f})"); letter(axf[5], 'f')
savefig('Fig18_station_maps_elevation')
print('figures done:', sorted(os.listdir(FIG)))

# %% ===================== PART 5: FIGURES MADE FOR THE MANUSCRIPT (workflow diagram, before/during/after-episode comparison) =====================
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
PAPER_FIG = f'{OUT}/paper_figs'; os.makedirs(PAPER_FIG, exist_ok=True)
_rc_backup = plt.rcParams.copy()
# ---- workflow diagram

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],'font.weight':'bold'})
fig, ax = plt.subplots(figsize=(8.6, 9.6)); ax.set_xlim(0, 8.6); ax.set_ylim(0, 9.6); ax.axis('off')
def box(x, y, w, h, t, fc, fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03,rounding_size=0.1', fc=fc, ec='#333', lw=1.3)); ax.text(x+w/2, y+h/2, t, ha='center', va='center', fontsize=fs, linespacing=1.25)
def arr(x1, y1, x2, y2): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=15, lw=1.4, color='#333'))
C1, C2, C3, C4 = '#dbe9f6', '#fde0d2', '#e5f2dc', '#ece3f4'
box(1.2, 8.5, 6.2, 0.95, 'Daily station data, 1981–2025\n(rain, Tmax, Tmin, RH, wind)\n+ station table (lon, lat, elevation)', C1)
box(1.9, 7.2, 4.8, 0.85, 'Quality control, duplicate check,\nhomogeneity tests (Pettitt, SNHT)', C1); arr(4.3, 8.5, 4.3, 8.05)
box(0.1, 5.4, 2.65, 1.35, 'Rainfall indices\nR10mm, R20mm, Rx1day,\nRx5day, SDII, CDD, CWD,\nR95p, PCI', C2, 10.5)
box(2.97, 5.4, 2.65, 1.35, 'Temperature indices\nTX90p, TN90p, TXx,\nheatwave days, WSDI,\nDTR', C2, 10.5)
box(5.85, 5.4, 2.65, 1.35, 'Drought indices\nSPI-1/3/6/12,\nSPEI-1/3/6/12\n(Hargreaves PET)', C2, 10.5)
for x in (1.42, 4.3, 7.18): arr(4.3, 7.2, x, 6.75)
box(0.7, 3.75, 7.2, 1.1, 'Compound drought–heat events (CDHE)\nand severity index (CDHI):\nSPI-3 ≤ −1 and monthly Tmax ≥ P90', C3)
arr(1.42, 5.4, 2.2, 4.85); arr(4.3, 5.4, 4.3, 4.85); arr(7.18, 5.4, 6.4, 4.85)
box(0.1, 1.95, 2.65, 1.45, 'Statistics\nMK/Hamed–Rao, Sen, FDR,\nPettitt, copulas,\npermutation tests', C3, 10.5)
box(2.97, 1.95, 2.65, 1.45, 'Elevation and structure\nzones, lapse rate,\nSpearman, EOF,\nclustering, wavelets', C3, 10.5)
box(5.85, 1.95, 2.65, 1.45, 'Machine learning\nLR, RF, boosting;\ntime-blocked split\n1982–2004 | 2005–2014 |\n2015–2025', C4, 10.2)
for x in (1.42, 4.3, 7.18): arr(4.3, 3.75, x, 3.4)
box(0.1, 0.15, 4.6, 1.2, 'Synthesis: frequency,\ndependence, change,\nelevation dependence,\nprecursors and thresholds', '#fff4c2', 10.5)
box(5.3, 0.15, 3.2, 1.2, 'Explainability\nShapley values, ALE,\nH² interactions,\nzone-specific models', C4, 10.2)
arr(1.42, 1.95, 1.9, 1.35); arr(4.3, 1.95, 3.6, 1.35); arr(7.18, 1.95, 7.0, 1.35); arr(5.3, 0.75, 4.7, 0.75)
plt.savefig(f'{PAPER_FIG}/workflow.png', dpi=200, bbox_inches='tight'); plt.close()

# ---- period comparison (uses tables/T10b_excess_by_period.csv written in PART 2b)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','font.size':12,'axes.titlesize':12.5,'axes.labelsize':12.5,'xtick.labelsize':11.5,'ytick.labelsize':11.5,'legend.fontsize':10.5,'axes.spines.top':False,'axes.spines.right':False})
E = pd.read_csv(f'{TAB}/T10b_excess_by_period.csv', index_col=0)
per = ['1981-2000', '2001-2005', '2006-2025']; lab = ['1981–2000', '2001–2005\n(episode)', '2006–2025']
fig, ax = plt.subplots(1, 2, figsize=(9.6, 5.4)); x = np.arange(3); w = .36
ob, ex = E.loc[per, 'observed_pct'].values, E.loc[per, 'expected_indep_pct'].values
ax[0].bar(x-w/2, ex, w, color='#999999', label='Expected if independent'); ax[0].bar(x+w/2, ob, w, color='#b2182b', label='Observed compound')
for i, p in enumerate(per): ax[0].text(x[i]+w/2, ob[i]+.8, f"×{E.loc[p, 'excess_factor']:.1f}", ha='center', fontsize=11)
ax[0].set_xticks(x); ax[0].set_xticklabels(lab); ax[0].set_ylabel('Compound months (% of station-months)'); ax[0].set_xlabel('Period'); ax[0].set_ylim(0, 38); ax[0].legend(loc='upper left'); ax[0].set_title('Observed vs independence-expected frequency')
hn, c = E.loc[per, 'hot_not_drought_pct'].values, E.loc[per, 'observed_pct'].values
ax[1].bar(x, c, .55, color='#b2182b', label='Hot month with drought (compound)'); ax[1].bar(x, hn, .55, bottom=c, color='#f4a582', label='Hot month without drought')
for i, p in enumerate(per): ax[1].text(x[i], c[i]+hn[i]+.8, f"P(D|H) = {E.loc[p, 'p_drought_given_hot_pct']:.0f}%", ha='center', fontsize=10.5)
ax[1].set_xticks(x); ax[1].set_xticklabels(lab); ax[1].set_ylabel('Hot months (% of station-months)'); ax[1].set_xlabel('Period'); ax[1].set_ylim(0, 58); ax[1].legend(loc='upper left'); ax[1].set_title('Composition of hot months')
for a_, t in zip(ax, 'ab'): a_.text(-0.02, 1.03, t, transform=a_.transAxes, fontsize=17, fontweight='bold', ha='right', va='bottom')
plt.tight_layout(); plt.savefig(f'{PAPER_FIG}/periods.png', dpi=200, bbox_inches='tight'); plt.close()

plt.rcParams.update(_rc_backup)
print('manuscript figures written to', PAPER_FIG)
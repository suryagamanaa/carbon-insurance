"""
================================================================================
KAFALACARBON - PARAMETRIC TAKAFUL FOR CARBON CREDIT INSURANCE
================================================================================
"KafalaCarbon: A Parametric Takaful Framework
for Wildfire-Induced Carbon Credit Reversal Risk in Indonesia"

Referensi Paper:
- Compound Negative Binomial-Lognormal aggregate loss model
- Grid-based pricing 0.1° × 0.1°
- λ_normal = 1,177.2 events/yr; λ_ENSO = 3,110.2 events/yr
- E[X] = IDR 666,742; CV = 5.02
- L_basis = P75 = IDR 2,375,789; δ = 0.50; θ = 0.35; w = 0.25
- φ = 0.70 (basis risk); η = 0.25 (net revenue factor)
- Premium range: 0.54% – 1.20% of net carbon revenue
================================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, os, glob, json
from datetime import datetime
from shapely.geometry import Polygon
from pyproj import Transformer

warnings.filterwarnings("ignore")

# ============================================================
# KONFIGURASI PATH
# ============================================================
BASE = r"C:\Users\Surya\OneDrive\Dokumen\HESTI TRIU"

VIIRS_PAT = os.path.join(BASE, "viirs-snpp_202*_Indonesia.csv")
SUB_CARB_CSV = os.path.join(BASE, "IDN (1) - Subnational 1 carbon data.csv")
SUB_PRIM_CSV = os.path.join(BASE, "IDN (1) - Subnational 1 primary loss.csv")
SUB_TREE_CSV = os.path.join(BASE, "IDN (1) - Subnational 1 tree cover loss.csv")
SUB_EMIS_CSV = os.path.join(BASE, "IDN (1) - Subnational 1 emissions.csv")

# ============================================================
# PARAMETER MODEL (SESUAI PAPER)
# ============================================================

# --- Parameter deteksi kebakaran (Sesuai paper §2.2) ---
TIME_WINDOW = 48     # jam (paper: 48 jam)
EPS_KM = 5           # radius DBSCAN km (paper: 5 km)
MIN_SAMPLES = 3      # minimum hotspot per cluster (paper: 3 deteksi)
THR = 30             # threshold tree cover density GFW
BUFFER_M = 500       # buffer convex hull (paper §2.3: 500 m)
MIN_HA_PER_HOTSPOT = 0.5  # paper §2.3: minimum 0.5 ha/deteksi

# --- Parameter aktuaria (Sesuai paper §3.2 & §3.3) ---
THETA = 0.35         # safety loading (paper §3.2)
W_FEE = 0.25         # wakalah fee (paper §3.2)
DELTA = 0.50         # spatial diversification exponent (paper §3.2 & §3.3)
PHI = 0.70           # basis risk factor (paper §2.4 & §3.2)
ETA = 0.25           # net revenue factor (paper §2.6 & §3.2)

# --- Parameter grid (Sesuai paper §2.6) ---
GRID_SIZE_DEG = 0.1  # grid 0.1° × 0.1°
GRID_AREA_HA = (GRID_SIZE_DEG * 111.0) ** 2  # ≈ 123.2 km² ≈ 12,321 ha

# --- Bounding box Kalimantan Timur (Sesuai paper §2.1) ---
KALTIM_LAT_MIN, KALTIM_LAT_MAX = -2.5, 2.5
KALTIM_LON_MIN, KALTIM_LON_MAX = 113.5, 119.0

# --- Harga karbon IDXCarbon (Sesuai paper §2.1: IDR 20,000–62,533/tCO2) ---
price_map = {
    2020: 30000, 2021: 35000, 2022: 50000,
    2023: 62533, 2024: 55985, 2025: 58800, 2026: 62000, 2027: 65000
}

# --- Segmentasi klien (Sesuai paper §3.3, Table 1) ---
TARGET_CLIENTS = {
    "Small Cooperative":        {"area": 500,    "is_entry_point": False},
    "Medium Cooperative":       {"area": 1000,   "is_entry_point": False},
    "Large Cooperative (entry)":{"area": 2500,   "is_entry_point": True},
    "Medium Cooperative Plus":  {"area": 5000,   "is_entry_point": False},
    "Small Company":            {"area": 10000,  "is_entry_point": False},
    "Corporate REDD+":          {"area": 25000,  "is_entry_point": False},
    "Large Corporate":          {"area": 50000,  "is_entry_point": False},
}

# ============================================================
# BURNED AREA ESTIMATOR (Sesuai paper §2.3)
# ============================================================
class BurnedAreaEstimator:
    """
    Estimasi luas kebakaran dengan convex hull + buffer 500m + minimum 0.5 ha/deteksi,
    dikalibrasi terhadap data GFW (paper §2.3).
    """
    
    def __init__(self, calibration_factor=1.0):
        self.calibration_factor = calibration_factor
        self.fallback_used_count = 0
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
        self.buffer_m = BUFFER_M
        self.ha_per_hotspot_base = MIN_HA_PER_HOTSPOT
    
    def _to_meters(self, lons, lats):
        xs, ys = [], []
        for lon, lat in zip(lons, lats):
            x, y = self.transformer.transform(lon, lat)
            xs.append(x)
            ys.append(y)
        return np.column_stack([xs, ys])
    
    def estimate(self, hotspots):
        n = len(hotspots)
        if n == 0:
            return 0.0
        if n == 1:
            return self.ha_per_hotspot_base * self.calibration_factor
        
        try:
            coords_m = self._to_meters(hotspots['longitude'].values, 
                                        hotspots['latitude'].values)
            hull = ConvexHull(coords_m)
            polygon = Polygon([coords_m[i] for i in hull.vertices])
            polygon = polygon.buffer(self.buffer_m)
            area_ha = polygon.area / 10000.0
            min_area = n * self.ha_per_hotspot_base
            area_ha = max(area_ha, min_area)
            return area_ha * self.calibration_factor
        except Exception:
            # Fallback: minimal 0.5 ha per hotspot (sesuai paper, bukan 0.3)
            self.fallback_used_count += 1
            return n * self.ha_per_hotspot_base * self.calibration_factor


# ============================================================
# SEVERITY MODEL (Sesuai paper §2.4)
# ============================================================
class SeverityModel:
    """
    Fitting distribusi severity: Lognormal, Gamma, Weibull.
    Dipilih berdasarkan AIC terendah (paper §2.4).
    """
    
    def __init__(self, loss_vals):
        self.loss_vals = loss_vals[loss_vals > 0]
        self.best_dist = None
        self.best_name = None
        self.best_params = None
        self.fallback_mode = False
        self.fit()
    
    def fit(self):
        distributions = {
            'Lognormal': stats.lognorm,
            'Gamma': stats.gamma,
            'Weibull': stats.weibull_min
        }
        
        best_aic = np.inf
        aic_results = {}
        
        print(f"\n   Fitting severity model ({len(self.loss_vals):,} samples):")
        
        for name, dist in distributions.items():
            try:
                params = dist.fit(self.loss_vals, floc=0)
                loglik = np.sum(dist.logpdf(self.loss_vals, *params))
                k = len(params)
                aic = 2*k - 2*loglik
                aic_results[name] = aic
                
                flag = ""
                if aic < best_aic:
                    best_aic = aic
                    self.best_dist = dist
                    self.best_name = name
                    self.best_params = params
                    flag = " ← BEST"
                
                print(f"     {name:<12} AIC={aic:>15,.1f}{flag}")
            except Exception as e:
                print(f"     {name:<12} FAILED: {str(e)[:50]}")
        
        if self.best_dist is None:
            self.fallback_mode = True
            self.best_name = "Empirical (Bootstrap)"
            print(f"   ⚠ FALLBACK: Empirical distribution")
        else:
            # Hitung ΔAIC vs Weibull (paper: ΔAIC = 9,220)
            if 'Weibull' in aic_results:
                delta_aic = aic_results['Weibull'] - best_aic
                print(f"   ΔAIC (best vs Weibull): {delta_aic:,.1f}")
    
    def sample(self, n_samples):
        if n_samples == 0:
            return np.array([])
        if self.fallback_mode:
            return np.random.choice(self.loss_vals, n_samples)
        
        try:
            if self.best_name == "Lognormal":
                s, loc, scale = self.best_params
                mu = np.log(scale)
                sigma = s
                return np.random.lognormal(mu, sigma, n_samples)
            else:
                return self.best_dist.rvs(*self.best_params, size=n_samples)
        except Exception:
            return np.random.choice(self.loss_vals, n_samples)
    
    @property
    def mean(self):
        if self.fallback_mode or len(self.loss_vals) == 0:
            return np.mean(self.loss_vals) if len(self.loss_vals) > 0 else 0
        if self.best_name == "Lognormal":
            s, loc, scale = self.best_params
            return np.exp(np.log(scale) + s**2/2)
        return self.best_dist.mean(*self.best_params)
    
    @property
    def cv(self):
        """Coefficient of variation (paper: CV = 5.02)"""
        if self.fallback_mode or len(self.loss_vals) == 0:
            return np.std(self.loss_vals) / np.mean(self.loss_vals) if np.mean(self.loss_vals) > 0 else 0
        return np.std(self.loss_vals) / self.mean if self.mean > 0 else 0


# ============================================================
# STEP 0: LOAD DATA
# ============================================================
print("=" * 80)
print(" KAFALACARBON - PARAMETRIC TAKAFUL FOR CARBON CREDIT INSURANCE")
print(" Selaras dengan Paper: Compound NB-Lognormal, Grid-Based Pricing")
print("=" * 80)

print("\n[0] LOADING DATA...")

files = sorted(glob.glob(VIIRS_PAT))
frames = []
for f in files:
    yr = int(os.path.basename(f).split('_')[1][:4])
    tmp = pd.read_csv(f)
    tmp['year'] = yr
    frames.append(tmp)
    print(f"   ✓ VIIRS {yr}: {len(tmp):,} baris")
df_viirs = pd.concat(frames, ignore_index=True)
print(f"   Total VIIRS: {len(df_viirs):,} baris")

df_carb = pd.read_csv(SUB_CARB_CSV)
print(f"   ✓ Subnational Carbon: {len(df_carb):,} baris")

df_prim = pd.read_csv(SUB_PRIM_CSV)
print(f"   ✓ Subnational Primary Loss: {len(df_prim):,} baris")

df_tree = pd.read_csv(SUB_TREE_CSV)
print(f"   ✓ Subnational Tree Cover Loss: {len(df_tree):,} baris")

try:
    df_emis = pd.read_csv(SUB_EMIS_CSV)
    EMISSIONS_AVAILABLE = True
    print(f"   ✓ Subnational Emissions: {len(df_emis):,} baris")
except:
    EMISSIONS_AVAILABLE = False
    print(f"   ⚠ Subnational Emissions tidak ditemukan")


# ============================================================
# STEP 1: EKSTRAKSI PARAMETER KALIMANTAN TIMUR
# ============================================================
print("\n[1] EKSTRAKSI PARAMETER KALIMANTAN TIMUR...")

PROV = "Kalimantan Timur"

carb_kaltim = df_carb[
    (df_carb['subnational1'].str.strip() == PROV) &
    (df_carb['umd_tree_cover_density_2000__threshold'] == THR)
].copy()

if len(carb_kaltim) == 0:
    raise ValueError("Data Kalimantan Timur tidak ditemukan!")

A_REGIONAL = carb_kaltim['umd_tree_cover_extent_2000__ha'].values[0]
gross_removals = abs(carb_kaltim['gfw_forest_carbon_gross_removals__Mg_CO2_yr-1'].values[0])

# R_SEQ: paper menggunakan 2.3844 tCO2/ha/yr
# Hitung dari data, tapi validasi terhadap paper
if EMISSIONS_AVAILABLE:
    emis_kaltim = df_emis[df_emis['subnational1'].str.strip() == PROV]
    if len(emis_kaltim) > 0:
        emis_cols = [c for c in emis_kaltim.columns if 'emissions' in c.lower() and 'Mg_CO2' in c]
        if emis_cols:
            annual_emissions = abs(emis_kaltim[emis_cols[0]].values[0])
            AVOIDED_EMISSIONS_RATE = max(0, annual_emissions * 0.6)
            R_SEQ = AVOIDED_EMISSIONS_RATE / A_REGIONAL
            R_SEQ_SOURCE = "AVOIDED EMISSIONS"
        else:
            R_SEQ = gross_removals / A_REGIONAL
            R_SEQ_SOURCE = "GROSS REMOVALS"
    else:
        R_SEQ = gross_removals / A_REGIONAL
        R_SEQ_SOURCE = "GROSS REMOVALS"
else:
    REDD_ELIGIBILITY_FACTOR = 0.55
    R_SEQ = (gross_removals * REDD_ELIGIBILITY_FACTOR) / A_REGIONAL
    R_SEQ_SOURCE = f"GROSS REMOVALS × {REDD_ELIGIBILITY_FACTOR:.0%}"

print(f"   ✓ A_REGIONAL: {A_REGIONAL:,.0f} ha")
print(f"   ✓ R_SEQ (dari data): {R_SEQ:.4f} tCO2/ha/yr")
print(f"   ✓ R_SEQ source: {R_SEQ_SOURCE}")
print(f"   ℹ Paper menggunakan R_SEQ = 2.3844 tCO2/ha/yr")

# Primary loss untuk kalibrasi (paper: 524,514 ha untuk 2012–2024)
prim_kaltim = df_prim[
    (df_prim['subnational1'].str.strip() == PROV) &
    (df_prim['threshold'] == THR)
].copy()

if len(prim_kaltim) > 0:
    loss_cols = [c for c in prim_kaltim.columns if c.startswith('tc_loss_ha_')]
    prim_loss = {}
    for col in loss_cols:
        yr = int(col.replace('tc_loss_ha_', ''))
        if 2012 <= yr <= 2024:
            prim_loss[yr] = prim_kaltim[col].values[0]
    TOTAL_PRIMARY_LOSS = sum(prim_loss.values())
    LOSS_BY_YEAR = prim_loss
    print(f"   ✓ Total Primary Loss GFW (2012-2024): {TOTAL_PRIMARY_LOSS:,.0f} ha")
    print(f"   ℹ Paper: 524,514 ha")
else:
    TOTAL_PRIMARY_LOSS = None
    LOSS_BY_YEAR = {}
    print("   ⚠ Data primary loss tidak tersedia")


# ============================================================
# STEP 2: FILTER VIIRS KE KALTIM
# ============================================================
print("\n[2] FILTER VIIRS KE KALIMANTAN TIMUR...")

n0 = len(df_viirs)
df_v = df_viirs[df_viirs['confidence'].str.strip().str.lower().isin(['n', 'h'])].copy()
df_v = df_v[df_v['type'] == 0].copy()
df_v = df_v[df_v['frp'].notna() & (df_v['frp'] > 0)].copy()
df_v['acq_date'] = pd.to_datetime(df_v['acq_date'])
n_conf = len(df_v)

df_v = df_v[
    (df_v['latitude'] >= KALTIM_LAT_MIN) & (df_v['latitude'] <= KALTIM_LAT_MAX) &
    (df_v['longitude'] >= KALTIM_LON_MIN) & (df_v['longitude'] <= KALTIM_LON_MAX)
].copy()

print(f"   Raw nasional: {n0:,}")
print(f"   After conf/type/frp: {n_conf:,} ({n_conf/n0*100:.1f}%)")
print(f"   After Kaltim bbox: {len(df_v):,} ({len(df_v)/n0*100:.1f}%)")
print(f"   ℹ Paper: 325,126 deteksi (2012–2024)")


# ============================================================
# STEP 3: DBSCAN CLUSTERING (Sesuai paper §2.2)
# ============================================================
print("\n[3] DBSCAN CLUSTERING (Spasiotemporal: 48 jam, 5 km)...")

eps_rad = EPS_KM / 6371
all_events = []

for yr in sorted(df_v['year'].unique()):
    df_yr = df_v[df_v['year'] == yr].copy().reset_index(drop=True)
    df_yr = df_yr.sort_values('acq_date').reset_index(drop=True)
    df_yr['acq_ts'] = df_yr['acq_date'].astype(np.int64) // 10**9
    t0 = df_yr['acq_ts'].min()
    df_yr['time_win'] = ((df_yr['acq_ts'] - t0) // (TIME_WINDOW * 3600)).astype(int)
    
    cluster_global = -np.ones(len(df_yr), dtype=int)
    cluster_counter = 0
    
    for tw in sorted(df_yr['time_win'].unique()):
        mask = (df_yr['time_win'] == tw).values
        sub = df_yr[mask]
        if len(sub) < MIN_SAMPLES:
            continue
        coords = np.radians(sub[['latitude', 'longitude']].values)
        db = DBSCAN(eps=eps_rad, min_samples=MIN_SAMPLES,
                    algorithm='ball_tree', metric='haversine').fit(coords)
        labels = db.labels_
        valid = labels >= 0
        if valid.sum() >= MIN_SAMPLES:
            unique_labels = np.unique(labels[valid])
            original_indices = df_yr.index[mask].values
            for label in unique_labels:
                mask_label = (labels == label)
                cluster_global[original_indices[mask_label]] = cluster_counter
                cluster_counter += 1
    
    df_yr['cluster_id'] = cluster_global
    
    for cid in df_yr[df_yr['cluster_id'] != -1]['cluster_id'].unique():
        cluster_df = df_yr[df_yr['cluster_id'] == cid]
        all_events.append({
            'cluster_id': cid,
            'year': yr,
            'n_hotspot': len(cluster_df),
            'total_frp': cluster_df['frp'].sum(),
            'date': cluster_df['acq_date'].min(),
            'lat': cluster_df['latitude'].mean(),
            'lon': cluster_df['longitude'].mean(),
            'hotspots': cluster_df
        })

print(f"   Total events: {len(all_events):,}")
print(f"   ℹ Paper: 23,036 event terkalibrasi (2012–2024)")


# ============================================================
# STEP 4: ESTIMASI BURNED AREA + KALIBRASI GFW (Sesuai paper §2.3)
# ============================================================
print("\n[4] ESTIMASI BURNED AREA (Convex Hull + Kalibrasi GFW)...")

estimator_raw = BurnedAreaEstimator(calibration_factor=1.0)
events_raw = []
for ev in all_events:
    burned_ha = estimator_raw.estimate(ev['hotspots'])
    events_raw.append({
        'year': ev['year'],
        'n_hotspot': ev['n_hotspot'],
        'burned_ha_raw': burned_ha
    })

df_events_raw = pd.DataFrame(events_raw)
total_burned_raw = df_events_raw['burned_ha_raw'].sum()
print(f"   Raw burned area (convex hull): {total_burned_raw:,.0f} ha")
print(f"   ℹ Paper: 40,088,273 ha (76.4× overestimate)")

# Kalibrasi
if TOTAL_PRIMARY_LOSS and TOTAL_PRIMARY_LOSS > 0:
    calibration_factor = TOTAL_PRIMARY_LOSS / total_burned_raw
    print(f"   Calibration factor: {calibration_factor:.4f}")
    print(f"   ℹ Paper: CF = 0.0131")
else:
    calibration_factor = 0.0131  # fallback sesuai paper
    print(f"   ⚠ FALLBACK: calibration_factor = 0.0131 (sesuai paper)")

# Terapkan kalibrasi + hitung loss (paper §2.4)
# L_i = A_i × r_seq × p_carbon × φ
estimator_calibrated = BurnedAreaEstimator(calibration_factor=calibration_factor)
events = []

for ev in all_events:
    burned_ha = estimator_calibrated.estimate(ev['hotspots'])
    emission = burned_ha * R_SEQ
    claimable_emission = emission * PHI  # φ = 0.70 (paper §2.4)
    loss_idr = claimable_emission * price_map.get(ev['year'], 58800)
    
    events.append({
        'cluster_id': ev['cluster_id'],
        'year': ev['year'],
        'n_hotspot': ev['n_hotspot'],
        'burned_ha': burned_ha,
        'emission_tco2': emission,
        'claimable_emission_tco2': claimable_emission,
        'loss_idr': loss_idr,
        'date': ev['date'],
        'lat': ev['lat'],
        'lon': ev['lon']
    })

events_df = pd.DataFrame(events)
events_df['date'] = pd.to_datetime(events_df['date'])

print(f"\n   SETELAH KALIBRASI + BASIS RISK (φ={PHI}):")
print(f"   Total burned area: {events_df['burned_ha'].sum():,.0f} ha")
print(f"   Total claimable emissions: {events_df['claimable_emission_tco2'].sum():,.0f} tCO2")
print(f"   Mean loss per event: IDR {events_df['loss_idr'].mean():,.0f}")
print(f"   ℹ Paper: E[X] = IDR 666,742")


# ============================================================
# STEP 5: FREQUENCY MODEL (Sesuai paper §2.5)
# ============================================================
print("\n[5] FREQUENCY MODEL (Negative Binomial Dual-Regime)...")

# Klasifikasi ENSO berdasarkan NOAA ONI (paper §2.1)
# Paper: ENSO years = 2015, 2016, 2019, 2023
ENSO_YEARS = [2015, 2016, 2019, 2023]

events_per_year = events_df.groupby('year').size().to_dict()
all_years = sorted(events_per_year.keys())

# Pisahkan ENSO vs non-ENSO
enso_counts = [events_per_year[y] for y in all_years if y in ENSO_YEARS]
non_enso_counts = [events_per_year[y] for y in all_years if y not in ENSO_YEARS]

lambda_normal = np.mean(non_enso_counts) if non_enso_counts else 0
lambda_enso = np.mean(enso_counts) if enso_counts else 0
enso_multiplier = lambda_enso / lambda_normal if lambda_normal > 0 else 0

print(f"   Events per year: {events_per_year}")
print(f"   λ_normal (non-ENSO): {lambda_normal:,.1f} events/yr")
print(f"   λ_ENSO: {lambda_enso:,.1f} events/yr")
print(f"   ENSO multiplier: {enso_multiplier:.2f}×")
print(f"   ℹ Paper: λ_normal = 1,177.2; λ_ENSO = 3,110.2; multiplier = 2.64×")

# Dispersion index
annual_mean = np.mean(list(events_per_year.values()))
annual_var = np.var(list(events_per_year.values()), ddof=1)
disp_idx = annual_var / annual_mean if annual_mean > 0 else 1

print(f"\n   Mean events/year: {annual_mean:.2f}")
print(f"   Dispersion index: {disp_idx:.1f}")
print(f"   ℹ Paper: I = 414.6 (Negative Binomial, bukan Poisson)")

# Mann-Kendall trend test (paper §2.5)
def mann_kendall(x):
    n = len(x)
    s = 0
    for i in range(n-1):
        for j in range(i+1, n):
            s += np.sign(x[j] - x[i])
    var_s = n*(n-1)*(2*n+5)/18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return s, z, p

years_sorted = sorted(events_per_year.keys())
counts_sorted = [events_per_year[y] for y in years_sorted]
mk_s, mk_z, mk_p = mann_kendall(counts_sorted)

print(f"\n   Mann-Kendall trend test:")
print(f"   S = {mk_s:.0f}, Z = {mk_z:.3f}, p = {mk_p:.3f}")
print(f"   ℹ Paper: slope = −203.7 events/yr; p = 0.135 (stasioner)")

# Frequency model: Negative Binomial (overdispersed)
if disp_idx > 1.5:
    freq_model = "Negative Binomial"
    # Method of moments untuk NB
    r_nb = max(0.1, annual_mean**2 / (annual_var - annual_mean)) if annual_var > annual_mean else 1
    p_nb = r_nb / (r_nb + annual_mean)
    print(f"\n   NB parameters: r = {r_nb:.2f}, p = {p_nb:.6f}")
else:
    freq_model = "Poisson"
    r_nb, p_nb = None, None

print(f"   Frequency model: {freq_model}")


# ============================================================
# STEP 6: SEVERITY MODEL (Sesuai paper §2.4)
# ============================================================
print("\n[6] SEVERITY MODEL (Lognormal vs Gamma vs Weibull)...")

loss_vals = events_df['loss_idr'].values
loss_vals = loss_vals[loss_vals > 0]

severity_model = SeverityModel(loss_vals)

print(f"\n   Best severity model: {severity_model.best_name}")
print(f"   E[X]: IDR {severity_model.mean:,.0f}")
print(f"   CV: {severity_model.cv:.2f}")
print(f"   ℹ Paper: E[X] = IDR 666,742; CV = 5.02")


# ============================================================
# STEP 7: GRID-BASED PRICING (Sesuai paper §2.6)
# ============================================================
print("\n[7] GRID-BASED PRICING (0.1° × 0.1°)...")

# Bagi Kaltim menjadi grid 0.1° × 0.1°
events_df['grid_lat'] = (events_df['lat'] / GRID_SIZE_DEG).round() * GRID_SIZE_DEG
events_df['grid_lon'] = (events_df['lon'] / GRID_SIZE_DEG).round() * GRID_SIZE_DEG
events_df['grid_id'] = events_df['grid_lat'].astype(str) + "_" + events_df['grid_lon'].astype(str)

# Hitung mean annual expected loss per grid
T_YEARS = len(years_sorted)

grid_losses = []
for grid_id, group in events_df.groupby('grid_id'):
    total_loss = group['loss_idr'].sum()
    mean_annual_loss = total_loss / T_YEARS
    grid_losses.append({
        'grid_id': grid_id,
        'total_loss': total_loss,
        'mean_annual_loss': mean_annual_loss,
        'n_events': len(group)
    })

grid_df = pd.DataFrame(grid_losses)
active_cells = len(grid_df)

print(f"   Active grid cells: {active_cells:,}")
print(f"   ℹ Paper: 1,217 active cells")
print(f"\n   Grid-level mean annual expected loss:")
print(f"   P50: IDR {grid_df['mean_annual_loss'].quantile(0.50):,.0f}")
print(f"   P75: IDR {grid_df['mean_annual_loss'].quantile(0.75):,.0f}")
print(f"   Mean: IDR {grid_df['mean_annual_loss'].mean():,.0f}")
print(f"   ℹ Paper: P50 = IDR 993,761; P75 = IDR 2,375,789; mean = IDR 2,670,781")

# Pricing basis: P75 (paper §3.2)
L_BASIS = grid_df['mean_annual_loss'].quantile(0.75)
print(f"\n   L_basis (P75): IDR {L_BASIS:,.0f}")
print(f"   ℹ Paper: L_basis = IDR 2,375,789")


# ============================================================
# STEP 8: ACTUARIAL PRICING (Sesuai paper §2.6 & §3.3)
# ============================================================
print("\n[8] ACTUARIAL PRICING (Compound NB-Lognormal)...")
print("=" * 80)
print(f"\n   Formula (paper §2.6):")
print(f"   E[S] = L_basis × (A / A_grid)^δ")
print(f"   P_gross = E[S] × (1 + θ) / (1 - w)")
print(f"   R_net = A × r_seq × p_carbon × η")
print(f"\n   Parameter: δ={DELTA}, θ={THETA}, w={W_FEE}, φ={PHI}, η={ETA}")

carbon_price = price_map.get(2025, 58800)

print("\n   " + "=" * 100)
print(f"   {'Segmen Klien':<28} {'Luas (ha)':<12} {'Net Revenue/thn':<22} {'E[S]/thn':<18} {'Premi/thn':<18} {'Rate':<10}")
print("   " + "-" * 100)

results = []

for client_name, client_info in TARGET_CLIENTS.items():
    area = client_info["area"]
    
    # E[S] = L_basis × (A / A_grid)^δ  (paper §2.6)
    # Catatan: paper menggunakan δ = 0.50
    # Untuk area kecil (A < A_grid), scaling proporsional (δ = 1) sesuai paper §2.6
    if area <= GRID_AREA_HA:
        # Small plots: δ = 1 (proporsional)
        E_S = L_BASIS * (area / GRID_AREA_HA)
    else:
        # Large landholdings: δ = 0.50 (sub-linear)
        E_S = L_BASIS * (area / GRID_AREA_HA) ** DELTA
    
    # Net carbon revenue (paper §2.6)
    # R_net = A × r_seq × p_carbon × η
    gross_carbon_revenue = area * R_SEQ * carbon_price
    net_carbon_revenue = gross_carbon_revenue * ETA
    
    # Gross premium (paper §2.6)
    # P_gross = E[S] × (1 + θ) / (1 - w)
    gross_premium = E_S * (1 + THETA) / (1 - W_FEE)
    
    # Premium rate
    premium_rate = (gross_premium / net_carbon_revenue) * 100 if net_carbon_revenue > 0 else 0
    
    # Coverage ratio
    coverage_ratio = net_carbon_revenue / gross_premium if gross_premium > 0 else 0
    
    results.append({
        'client': client_name,
        'area_ha': area,
        'gross_carbon_revenue_idr': gross_carbon_revenue,
        'net_carbon_revenue_idr': net_carbon_revenue,
        'E_S_idr': E_S,
        'gross_premium_idr': gross_premium,
        'premium_rate_pct': premium_rate,
        'coverage_ratio': coverage_ratio,
        'is_entry_point': client_info['is_entry_point']
    })
    
    print(f"   {client_name:<28} {area:>12,.0f}  IDR {net_carbon_revenue:>16,.0f}   IDR {E_S:>12,.0f}   IDR {gross_premium:>12,.0f}   {premium_rate:>6.2f}%")

print("   " + "=" * 100)

# Rekomendasi entry point (paper: 2,500 ha)
rec = [r for r in results if r['is_entry_point']][0]
print(f"\n   ✓ REKOMENDASI ENTRY POINT: {rec['client']} ({rec['area_ha']:,} ha)")
print(f"     - Net carbon revenue: IDR {rec['net_carbon_revenue_idr']:,.0f}")
print(f"     - E[S]: IDR {rec['E_S_idr']:,.0f}")
print(f"     - Gross premium: IDR {rec['gross_premium_idr']:,.0f}")
print(f"     - Premium rate: {rec['premium_rate_pct']:.2f}% of net revenue")
print(f"     - Coverage ratio: {rec['coverage_ratio']:.0f}×")
print(f"     ℹ Paper: rate = 1.20%, coverage = 83×")

# Validasi range premi (paper: 0.54% – 1.20%)
all_rates = [r['premium_rate_pct'] for r in results]
print(f"\n   Premium range: {min(all_rates):.2f}% – {max(all_rates):.2f}%")
print(f"   ℹ Paper: 0.54% – 1.20%")


# ============================================================
# STEP 9: MONTE CARLO SIMULATION (VaR/TVaR)
# ============================================================
print("\n[9] MONTE CARLO SIMULATION (Aggregate Loss Distribution)...")

np.random.seed(42)
n_sim = 100000

simulated_losses = []
simulated_losses_enso = []

for _ in range(n_sim):
    # Normal year
    if freq_model == "Negative Binomial":
        n_events = np.random.negative_binomial(r_nb, p_nb)
    else:
        n_events = np.random.poisson(annual_mean)
    losses = severity_model.sample(n_events)
    simulated_losses.append(losses.sum() if len(losses) > 0 else 0)
    
    # ENSO year (stress)
    n_events_enso = np.random.poisson(lambda_enso)
    losses_enso = severity_model.sample(n_events_enso)
    simulated_losses_enso.append(losses_enso.sum() if len(losses_enso) > 0 else 0)

simulated_losses = np.array(simulated_losses)
simulated_losses_enso = np.array(simulated_losses_enso)

VaR_95 = np.percentile(simulated_losses, 95)
VaR_99 = np.percentile(simulated_losses, 99)
TVaR_95 = simulated_losses[simulated_losses > VaR_95].mean()
TVaR_99 = simulated_losses[simulated_losses > VaR_99].mean()

VaR_95_enso = np.percentile(simulated_losses_enso, 95)
TVaR_95_enso = simulated_losses_enso[simulated_losses_enso > VaR_95_enso].mean()

print(f"\n   NORMAL YEAR (λ = {annual_mean:.1f}):")
print(f"   E[S]: IDR {np.mean(simulated_losses):,.0f}")
print(f"   VaR 95%: IDR {VaR_95:,.0f}")
print(f"   VaR 99%: IDR {VaR_99:,.0f}")
print(f"   TVaR 95%: IDR {TVaR_95:,.0f}")
print(f"   TVaR 99%: IDR {TVaR_99:,.0f}")

print(f"\n   ENSO YEAR (λ = {lambda_enso:.1f}):")
print(f"   E[S]: IDR {np.mean(simulated_losses_enso):,.0f}")
print(f"   VaR 95%: IDR {VaR_95_enso:,.0f}")
print(f"   TVaR 95%: IDR {TVaR_95_enso:,.0f}")


# ============================================================
# STEP 10: SAVE RESULTS
# ============================================================
print("\n[10] SAVING RESULTS...")

output = {
    'timestamp': datetime.now().isoformat(),
    'model_version': 'KafalaCarbon v1.0 (Paper-Aligned)',
    'paper_reference': 'Parametric Takaful for Wildfire-Induced Carbon Credit Reversal Risk',
    'parameters': {
        'A_REGIONAL_ha': float(A_REGIONAL),
        'R_SEQ_tCO2_ha_yr': float(R_SEQ),
        'R_SEQ_paper': 2.3844,
        'CALIBRATION_FACTOR': float(calibration_factor),
        'CALIBRATION_FACTOR_paper': 0.0131,
        'DELTA': DELTA,
        'THETA': THETA,
        'W_FEE': W_FEE,
        'PHI': PHI,
        'ETA': ETA,
        'GRID_SIZE_DEG': GRID_SIZE_DEG,
        'GRID_AREA_HA': GRID_AREA_HA
    },
    'fire_statistics': {
        'total_hotspots_kaltim': len(df_v),
        'total_events': len(events_df),
        'total_burned_area_ha': float(events_df['burned_ha'].sum()),
        'total_claimable_emissions_tco2': float(events_df['claimable_emission_tco2'].sum()),
        'events_per_year': {int(k): int(v) for k, v in events_per_year.items()},
        'lambda_normal': float(lambda_normal),
        'lambda_enso': float(lambda_enso),
        'enso_multiplier': float(enso_multiplier),
        'dispersion_index': float(disp_idx),
        'mann_kendall_p': float(mk_p)
    },
    'severity_statistics': {
        'best_model': severity_model.best_name,
        'E_X_idr': float(severity_model.mean),
        'CV': float(severity_model.cv),
        'E_X_paper': 666742,
        'CV_paper': 5.02
    },
    'grid_statistics': {
        'active_cells': active_cells,
        'P50': float(grid_df['mean_annual_loss'].quantile(0.50)),
        'P75': float(grid_df['mean_annual_loss'].quantile(0.75)),
        'mean': float(grid_df['mean_annual_loss'].mean()),
        'L_basis': float(L_BASIS)
    },
    'risk_metrics': {
        'VaR_95_normal': float(VaR_95),
        'VaR_99_normal': float(VaR_99),
        'TVaR_95_normal': float(TVaR_95),
        'TVaR_99_normal': float(TVaR_99),
        'VaR_95_enso': float(VaR_95_enso),
        'TVaR_95_enso': float(TVaR_95_enso)
    },
    'pricing_results': [
        {
            'client_segment': r['client'],
            'area_ha': r['area_ha'],
            'gross_carbon_revenue_idr': float(r['gross_carbon_revenue_idr']),
            'net_carbon_revenue_idr': float(r['net_carbon_revenue_idr']),
            'expected_annual_loss_idr': float(r['E_S_idr']),
            'gross_premium_idr': float(r['gross_premium_idr']),
            'premium_rate_pct': float(r['premium_rate_pct']),
            'coverage_ratio': float(r['coverage_ratio'])
        }
        for r in results
    ]
}

output_file = os.path.join(BASE, 'KafalaCarbon_results.json')
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"   ✓ Saved to: {output_file}")


# ============================================================
# STEP 11: VISUALISASI
# ============================================================
print("\n[11] GENERATING VISUALIZATION...")

fig = plt.figure(figsize=(22, 14))
fig.suptitle('KafalaCarbon - Parametric Takaful for Carbon Credit Insurance\n'
             f'Kalimantan Timur | Compound NB-Lognormal | Grid 0.1° | '
             f'Premium: {min(all_rates):.2f}%–{max(all_rates):.2f}% of net revenue',
             fontsize=13, fontweight='bold')

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.35)

# Panel 1: Hotspot per tahun
ax1 = fig.add_subplot(gs[0, 0])
hs_annual = df_v.groupby('year').size()
colors = ['#E24B4A' if y in ENSO_YEARS else '#185FA5' for y in hs_annual.index]
ax1.bar(hs_annual.index, hs_annual.values, color=colors, edgecolor='white')
ax1.set_title('Hotspot per Tahun (ENSO = merah)', fontsize=10, fontweight='bold')
ax1.set_xlabel('Tahun'); ax1.set_ylabel('Jumlah Hotspot')
for x, v in zip(hs_annual.index, hs_annual.values):
    ax1.text(x, v + hs_annual.max()*0.02, f'{v/1000:.0f}k', ha='center', fontsize=7)

# Panel 2: Events per tahun (ENSO vs non-ENSO)
ax2 = fig.add_subplot(gs[0, 1])
colors2 = ['#E24B4A' if y in ENSO_YEARS else '#1D9E75' for y in events_per_year.keys()]
ax2.bar(events_per_year.keys(), events_per_year.values(), color=colors2, edgecolor='white')
ax2.axhline(lambda_normal, color='#1D9E75', linestyle='--', linewidth=2, 
            label=f'λ_normal = {lambda_normal:,.0f}')
ax2.axhline(lambda_enso, color='#E24B4A', linestyle='--', linewidth=2, 
            label=f'λ_ENSO = {lambda_enso:,.0f}')
ax2.set_title('Fire Events per Tahun (Negative Binomial)', fontsize=10, fontweight='bold')
ax2.set_xlabel('Tahun'); ax2.set_ylabel('Jumlah Events')
ax2.legend(fontsize=8)

# Panel 3: Severity Distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(loss_vals, bins=50, color='#185FA5', alpha=0.7, edgecolor='white')
ax3.axvline(severity_model.mean, color='#E24B4A', linestyle='--', linewidth=2, 
            label=f'E[X] = IDR {severity_model.mean:,.0f}')
ax3.set_title(f'Severity ({severity_model.best_name}, CV={severity_model.cv:.2f})', 
              fontsize=10, fontweight='bold')
ax3.set_xlabel('Loss per Event (IDR)'); ax3.set_ylabel('Frekuensi')
ax3.legend(fontsize=8)
ax3.ticklabel_format(style='scientific', axis='x', scilimits=())

# Panel 4: QQ-Plot
ax4 = fig.add_subplot(gs[1, 0])
if not severity_model.fallback_mode and severity_model.best_name == 'Lognormal':
    log_loss = np.log(loss_vals[loss_vals > 0])
    (osm, osr), (slope, intercept, r) = stats.probplot(log_loss, dist='norm')
    ax4.set_title(f'QQ-Plot Lognormal (R² = {r**2:.4f})', fontsize=10, fontweight='bold')
else:
    (osm, osr), (slope, intercept, r) = stats.probplot(loss_vals, dist='norm')
    ax4.set_title(f'QQ-Plot (R² = {r**2:.4f})', fontsize=10, fontweight='bold')
ax4.plot(osm, osr, 'o', color='#185FA5', markersize=2, alpha=0.5)
ax4.plot([min(osm), max(osm)], [slope*min(osm)+intercept, slope*max(osm)+intercept], 
         'r--', linewidth=2)
ax4.set_xlabel('Theoretical Quantiles'); ax4.set_ylabel('Sample Quantiles')

# Panel 5: Premium Rate vs Area
ax5 = fig.add_subplot(gs[1, 1])
areas_plot = [r['area_ha'] for r in results]
rates_plot = [r['premium_rate_pct'] for r in results]
ax5.plot(areas_plot, rates_plot, 'o-', color='#534AB7', linewidth=2, markersize=10)
ax5.axhline(1.20, color='#E24B4A', linestyle='--', linewidth=1, label='Paper max = 1.20%')
ax5.axhline(0.54, color='#1D9E75', linestyle='--', linewidth=1, label='Paper min = 0.54%')
ax5.set_title('Premium Rate vs Area (Sub-linear Diversification)', fontsize=10, fontweight='bold')
ax5.set_xlabel('Area (ha)'); ax5.set_ylabel('Premium Rate (% net revenue)')
ax5.set_xscale('log')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)
for area, rate in zip(areas_plot, rates_plot):
    ax5.annotate(f'{rate:.2f}%', (area, rate), textcoords="offset points", 
                 xytext=(5,5), fontsize=8)

# Panel 6: Grid-level Loss Distribution
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(grid_df['mean_annual_loss'], bins=50, color='#185FA5', alpha=0.7, edgecolor='white')
ax6.axvline(L_BASIS, color='#E24B4A', linestyle='--', linewidth=2, 
            label=f'P75 (L_basis) = IDR {L_BASIS:,.0f}')
ax6.axvline(grid_df['mean_annual_loss'].quantile(0.50), color='#1D9E75', 
            linestyle='--', linewidth=2, label=f'P50 = IDR {grid_df["mean_annual_loss"].quantile(0.50):,.0f}')
ax6.set_title(f'Grid-Level Mean Annual Loss ({active_cells} cells)', fontsize=10, fontweight='bold')
ax6.set_xlabel('Mean Annual Loss (IDR)'); ax6.set_ylabel('Jumlah Grid')
ax6.legend(fontsize=8)
ax6.ticklabel_format(style='scientific', axis='x', scilimits=())

# Panel 7: Aggregate Loss Distribution
ax7 = fig.add_subplot(gs[2, 0])
ax7.hist(simulated_losses, bins=100, color='#185FA5', alpha=0.5, density=True, label='Normal')
ax7.hist(simulated_losses_enso, bins=100, color='#E24B4A', alpha=0.5, density=True, label='ENSO')
ax7.axvline(VaR_95, color='#185FA5', linestyle='--', linewidth=2, label=f'VaR 95% Normal')
ax7.axvline(VaR_95_enso, color='#E24B4A', linestyle='--', linewidth=2, label=f'VaR 95% ENSO')
ax7.set_title('Aggregate Loss Distribution (Monte Carlo)', fontsize=10, fontweight='bold')
ax7.set_xlabel('Annual Loss (IDR)'); ax7.set_ylabel('Density')
ax7.legend(fontsize=7)
ax7.ticklabel_format(style='scientific', axis='x', scilimits=())

# Panel 8: Pricing Results Table
ax8 = fig.add_subplot(gs[2, 1])
ax8.axis('off')
table_text = "PRICING RESULTS (Paper-Aligned)\n" + "="*55 + "\n\n"
table_text += f"{'Segment':<26} {'Area':>8} {'Rate':>8}\n"
table_text += "-"*55 + "\n"
for r in results:
    marker = " ★" if r['is_entry_point'] else ""
    table_text += f"{r['client']:<26} {r['area_ha']:>8,} {r['premium_rate_pct']:>7.2f}%{marker}\n"
table_text += "\n" + "="*55 + "\n"
table_text += f"Paper range: 0.54% – 1.20%\n"
table_text += f"Entry point: 2,500 ha (1.20%, 83×)\n"
table_text += f"\nKey Parameters:\n"
table_text += f"  δ = {DELTA}, θ = {THETA}, w = {W_FEE}\n"
table_text += f"  φ = {PHI}, η = {ETA}\n"
table_text += f"  L_basis = IDR {L_BASIS:,.0f}\n"
table_text += f"  E[X] = IDR {severity_model.mean:,.0f}\n"
table_text += f"  CV = {severity_model.cv:.2f}\n"
ax8.text(0.02, 0.98, table_text, transform=ax8.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace')
ax8.set_title('Pricing Summary', fontsize=10, fontweight='bold')

# Panel 9: Coverage Ratio
ax9 = fig.add_subplot(gs[2, 2])
coverage_ratios = [r['coverage_ratio'] for r in results]
client_short = [r['client'].split('(')[0].strip()[:20] for r in results]
bars = ax9.barh(client_short, coverage_ratios, color='#1D9E75', edgecolor='white')
ax9.set_title('Coverage Ratio (Net Revenue / Premium)', fontsize=10, fontweight='bold')
ax9.set_xlabel('Coverage Ratio (×)')
for bar, cr in zip(bars, coverage_ratios):
    ax9.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
             f'{cr:.0f}×', va='center', fontsize=8)
ax9.axvline(83, color='#E24B4A', linestyle='--', linewidth=1, label='Paper: 83×')
ax9.legend(fontsize=8)

plt.tight_layout()
output_img = os.path.join(BASE, 'KafalaCarbon_visualization.png')
plt.savefig(output_img, dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print(f"   ✓ Saved to: {output_img}")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 80)
print(" FINAL SUMMARY - KAFALACARBON")
print("=" * 80)
print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    KAFALACARBON - PARAMETRIC TAKAFUL                        │
│                 Wildfire-Induced Carbon Credit Reversal Risk                │
│                     (Paper-Aligned Implementation)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PARAMETER KALTIM:                                                          │
│  • A_REGIONAL              : {A_REGIONAL:>12,.0f} ha                       │
│  • R_SEQ (data)            : {R_SEQ:>12.4f} tCO2/ha/yr                    │
│  • R_SEQ (paper)           : {2.3844:>12.4f} tCO2/ha/yr                    │
│  • Calibration factor      : {calibration_factor:>12.4f}                    │
│  • Calibration (paper)     : {0.0131:>12.4f}                    │
│                                                                             │
│  FIRE STATISTICS:                                                           │
│  • Total events            : {len(events_df):>12,}                        │
│  • λ_normal                : {lambda_normal:>12,.1f} events/yr             │
│  • λ_ENSO                  : {lambda_enso:>12,.1f} events/yr             │
│  • ENSO multiplier         : {enso_multiplier:>12.2f}×                     │
│  • Dispersion index        : {disp_idx:>12.1f}                        │
│                                                                             │
│  SEVERITY:                                                                  │
│  • Best model              : {severity_model.best_name:>12}                        │
│  • E[X]                    : IDR {severity_model.mean:>10,.0f}                    │
│  • E[X] (paper)            : IDR {666742:>10,}                    │
│  • CV                      : {severity_model.cv:>12.2f}                        │
│  • CV (paper)              : {5.02:>12.2f}                        │
│                                                                             │
│  GRID PRICING:                                                              │
│  • Active cells            : {active_cells:>12,}                        │
│  • P50                     : IDR {grid_df['mean_annual_loss'].quantile(0.50):>10,.0f}                    │
│  • P75 (L_basis)           : IDR {L_BASIS:>10,.0f}                    │
│                                                                             │
│  PRICING RESULTS:                                                           │
│  • Premium range           : {min(all_rates):.2f}% – {max(all_rates):.2f}% of net revenue        │
│  • Paper range             : 0.54% – 1.20%                                  │
│  • Entry point (2,500 ha)  : {rec['premium_rate_pct']:.2f}% ({rec['coverage_ratio']:.0f}× coverage)          │
│  • Paper entry point       : 1.20% (83× coverage)                           │
│                                                                             │
│  RISK METRICS:                                                              │
│  • VaR 95% normal          : IDR {VaR_95:>10,.0f}                    │
│  • VaR 95% ENSO            : IDR {VaR_95_enso:>10,.0f}                    │
│  • TVaR 95% normal         : IDR {TVaR_95:>10,.0f}                    │
│                                                                             │
│  SHARIA COMPLIANCE:                                                         │
│  • Contract                : Wakalah bil Ujrah + Tabarru'                  │
│  • Fatwa                  : DSN-MUI No. 52 & 53 (2006)                    │
│  • Haq Mali framework      : Intangible carbon income as valid object       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")
print("=" * 80)
print("KAFALACARBON EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 80)
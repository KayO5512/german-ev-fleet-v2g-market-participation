# -*- coding: utf-8 -*-
"""
Enhanced Greedy ID vs. ID Optimization: Head-to-Head Comparison
DHL EV Fleet Market Participation Study

Compares:
  1. Enhanced Greedy ID (6-hour lookahead, driving forecasting, SOC reservation,
     time-of-use constraints) running on REAL intraday price + mobility data
  2. ID Optimization (perfect-foresight Pyomo LP) loaded from saved results OR
     re-run from scratch against the same real data

Produces publication-quality figures for analysis and reporting:
    01a–01d.{png,svg} – standalone comparison plots and performance table
    02_all_strategies_ranking.{png,svg} – lollipop chart (all scenarios)
    03_degradation_impact_analysis.{png,svg} – degradation sensitivity
    06a–06f.{png,svg} – standalone KPI plots
    07a–07d.{png,svg} – standalone vehicle-class plots and decision table

Academic foundation:
  Byrne et al. (2018), Srikrishnan & Clack (2018),
  Sortomme & El-Sharkawi (2012), Knottenbelt et al. (2017),
  Dufo-López et al. (2011)
"""

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.gridspec import GridSpec

# ============================================================
# 0. SETTINGS
# ============================================================

MOBILITY_WORKBOOK = Path(r"C:\Users\Yaw\Documents\EV_BID_Final\dhl_fleet_model_output_2026-06-29_00-58-34.xlsx")
DA_FILE  = Path(r"C:\Users\Yaw\Documents\EV_BID_Final\Data Ahead_ENERGY_PRICES_2025.xlsx")
ID_FILE  = Path(r"C:\Users\Yaw\Documents\EV_BID_Final\ID-Prices_label_2025.xlsx")
FCR_FILE = Path(r"C:\Users\Yaw\Documents\EV_BID_Final\FCR_yearly data_RESULT_OVERVIEW_CAPACITY_MARKET_2025-01-01_2025-12-31.xlsx")
COMPARISON_FILE = Path(r"C:\Users\Yaw\Documents\EV_BID_Final\greedy scenario graphs\dhl_market_comparison_2026-06-29_20-42-09.xlsx")

OUTPUT_FOLDER = Path(r"C:\Users\Yaw\Documents\EV_BID_Final\enhanced_vs_id_comparison")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

RUN_STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

YEAR = 2025
DT_H = 0.25              # 15-minute steps

# Physical
ETA_CH  = 0.95
ETA_DIS = 0.95
SOC_INIT_SHARE = 0.60

# Battery degradation
C_DEG_EUR_PER_MWH = 20.0

# Grid limit fraction
GRID_LIMIT_FRACTION = 0.80

# Enhanced greedy parameters
EG_LOOKAHEAD_HOURS   = 6      # Forward MA window for price signal
EG_DRIVE_RESERVE_H   = 6     # Hours of driving energy to reserve in SOC floor
EG_PRICE_THRESHOLD   = 0.05  # 5% deviation from fwd MA triggers trading
EG_TRADE_FRACTION    = 0.50  # Max fraction of rated power per step
EG_NEG_PRICE_LIMIT   = 0.0   # Always charge at full power when price < 0

# Color palette (blue-family, colorblind-friendly)
COLORS = {
    "enhanced_greedy": "#4A90E2",   # Blue for enhanced greedy
    "optimized_id":    "#0052CC",   # Dark blue for optimization
    "original_greedy": "#AED6F1",   # Light blue for original greedy (reference)
    "combined":        "#003D99",   # Navy for combined strategy
    "fcr":             "#6C5CE7",   # Purple for FCR
    "da":              "#2980B9",   # Mid-blue for DA
    "naive":           "#B0B0B0",   # Gray for naive EDF
    "loss":            "#E74C3C",   # Red for losses / costs
    "gain":            "#27AE60",   # Green for revenues
}

CLASS_GROUPS = {
    "StreetScooter Vans": [
        "streetscooter_last_mile_e_van",
    ],
    "EV Trucks (All)": [
        "volvo_fl_electric_4x2_berlin",
        "mercedes_benz_eactros_600_long_haul",
        "other_heavy_duty_e_truck",
    ],
    "E-Trikes": [
        "e_trike",
    ],
}


def eur_million_formatter(val, _):
    return f"€{val:.1f}M"

print("=" * 80)
print("ENHANCED GREEDY ID vs. ID OPTIMIZATION: COMPARISON")
print("=" * 80)
print(f"Output folder: {OUTPUT_FOLDER}")

# ============================================================
# 1. DATA LOADING
# ============================================================

def build_index() -> pd.DatetimeIndex:
    return pd.date_range(f"{YEAR}-01-01 00:00", f"{YEAR+1}-01-01", freq="15min", inclusive="left")


def load_mobility(idx: pd.DatetimeIndex) -> dict:
    m = pd.read_excel(MOBILITY_WORKBOOK, sheet_name="Hourly_Aggregate")
    m["datetime"] = pd.to_datetime(m["datetime"])
    m = m.set_index("datetime").sort_index()
    hourly_power = m[["p_charge_max_MW", "p_discharge_max_MW"]].reindex(idx, method="ffill")
    drive_q = (m["driving_energy_MWh"].reindex(idx, method="ffill") / 4.0)
    cap     = float(m["battery_capacity_MWh_total"].iloc[0])
    soc_min = float(m["soc_min_MWh"].iloc[0])
    soc_max = float(m["soc_max_MWh"].iloc[0])
    soc_init = cap * SOC_INIT_SHARE
    fleet_size = int(float(m["fleet_size"].iloc[0])) if "fleet_size" in m.columns else None
    return {
        "pch_max":  hourly_power["p_charge_max_MW"].to_numpy(),
        "pdis_max": hourly_power["p_discharge_max_MW"].to_numpy(),
        "edrive":   drive_q.to_numpy(),
        "cap": cap, "soc_min": soc_min, "soc_max": soc_max, "soc_init": soc_init,
        "fleet_size": fleet_size,
    }


def load_id(idx: pd.DatetimeIndex) -> np.ndarray:
    d = pd.read_excel(ID_FILE, sheet_name="Sheet1")
    d["DateTime"] = pd.to_datetime(d["DateTime"])
    s = d.set_index("DateTime")["IDC"].astype(float).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return s.reindex(idx, method="ffill").to_numpy()


def load_da(idx: pd.DatetimeIndex) -> np.ndarray:
    raw   = pd.read_excel(DA_FILE, sheet_name="1", header=None, skiprows=7)
    start = raw[0].astype(str).str.split(" - ").str[0]
    ts    = pd.to_datetime(start, errors="coerce", dayfirst=True)
    price = pd.to_numeric(raw[1], errors="coerce")
    s = pd.Series(price.values, index=ts).dropna().sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return s.reindex(idx, method="ffill").to_numpy()


def load_class_group_data(idx: pd.DatetimeIndex, base_data: dict) -> dict:
    hb = pd.read_excel(MOBILITY_WORKBOOK, sheet_name="Hourly_By_Class")
    hb["datetime"] = pd.to_datetime(hb["datetime"])
    hb = hb.sort_values(["datetime", "vehicle_class"])

    cs = pd.read_excel(MOBILITY_WORKBOOK, sheet_name="Class_Summary")

    soc_min_share = float(base_data["soc_min"]) / float(base_data["cap"])
    soc_max_share = float(base_data["soc_max"]) / float(base_data["cap"])

    grouped = {}
    for label, classes in CLASS_GROUPS.items():
        ts = (hb[hb["vehicle_class"].isin(classes)]
              .groupby("datetime")[["p_charge_max_MW", "p_discharge_max_MW", "driving_energy_MWh"]]
              .sum()
              .sort_index())
        if ts.empty:
            continue

        ts_q = ts.reindex(idx, method="ffill")
        meta = cs[cs["vehicle_class"].isin(classes)]

        cap = float(meta["total_battery_capacity_MWh"].sum())
        fleet_size = int(meta["target_count"].sum())

        grouped[label] = {
            "pch_max": ts_q["p_charge_max_MW"].to_numpy(),
            "pdis_max": ts_q["p_discharge_max_MW"].to_numpy(),
            "edrive": (ts_q["driving_energy_MWh"] / 4.0).to_numpy(),
            "cap": cap,
            "soc_min": cap * soc_min_share,
            "soc_max": cap * soc_max_share,
            "soc_init": cap * SOC_INIT_SHARE,
            "fleet_size": fleet_size,
            "source_classes": ", ".join(classes),
        }

    return grouped


print("\n[1/3] Loading real data...")
idx  = build_index()
t0   = time.time()
data = load_mobility(idx)
idp  = load_id(idx)
dap  = load_da(idx)
print(f"  ✓ Mobility loaded  | Capacity: {data['cap']:.1f} MWh | SOC range: [{data['soc_min']:.0f}, {data['soc_max']:.0f}] MWh")
print(f"  ✓ ID prices loaded | Mean: €{np.nanmean(idp):.2f}/MWh | Min: €{np.nanmin(idp):.2f} | Max: €{np.nanmax(idp):.2f}")
print(f"  ✓ DA prices loaded | Mean: €{np.nanmean(dap):.2f}/MWh")
print(f"  ✓ Data load time: {time.time()-t0:.1f}s | Timesteps: {len(idx):,}")

fleet_size = data.get("fleet_size")
if fleet_size is not None:
    FLEET_LABEL = f"DHL EV Fleet ({fleet_size:,} vehicles, {data['cap']:.0f} MWh)"
else:
    FLEET_LABEL = f"DHL EV Fleet ({data['cap']:.0f} MWh)"

grid_limit = data["pch_max"].max() * GRID_LIMIT_FRACTION

# ============================================================
# 2. LOAD OPTIMIZATION RESULTS FROM SAVED EXCEL
# ============================================================

print("\n[2/3] Loading saved optimization results...")
df_comp = pd.read_excel(COMPARISON_FILE, sheet_name="Comparison")
print(f"  ✓ Scenarios found: {sorted(df_comp['scenario'].unique())}")

def get_scenario(name: str, deg: bool) -> pd.Series:
    row = df_comp[(df_comp["scenario"] == name) & (df_comp["degradation"] == deg)]
    if len(row) == 0:
        raise ValueError(f"Scenario '{name}' (deg={deg}) not found in comparison file.")
    return row.iloc[0]

id_nodeg  = get_scenario("ID",       False)
id_deg    = get_scenario("ID",       True)
da_deg    = get_scenario("DA",       True)
fcr_deg   = get_scenario("FCR",      True)
comb_deg  = get_scenario("Combined", True)
naive_res = get_scenario("Naive_EDF", False)
orig_greedy_deg = df_comp[(df_comp["scenario"] == "Greedy_ID") & (df_comp["degradation"] == True)]
orig_greedy = orig_greedy_deg.iloc[0] if len(orig_greedy_deg) > 0 else None

print(f"  ✓ ID Optimization (deg=True):  €{float(id_deg['profit_EUR'])/1e6:.2f}M profit")
print(f"  ✓ Combined (deg=True):         €{float(comb_deg['profit_EUR'])/1e6:.2f}M profit")

# ============================================================
# 3. ENHANCED GREEDY ALGORITHM (on REAL data)
# ============================================================

def run_enhanced_greedy(data: dict, idp: np.ndarray, deg_on: bool, grid_limit_mw: float | None = None) -> dict:
    """
    Enhanced greedy intraday heuristic – Definitive production-ready version.

    Decision hierarchy (applied in order each timestep):

    0. Negative-price windfall: price < 0  → charge at FULL power regardless of
       peak hours.  The LP earns money at negative-price events; so must the greedy.

    1. Emergency charging: SOC within 20 MWh of 6h-driving reservation floor
       → charge at 70% rated power.  NO price gate – operational reliability first.
       This eliminates most unmet energy.

    2. Opportunistic charge: price < (fwd_6h_MA * (1-threshold)) AND off-peak
       AND room above floor  → charge at 50% rated power.

    3. Opportunistic discharge: price > (fwd_6h_MA * (1+threshold)) AND off-peak
       AND surplus above driving floor  → discharge at 50% rated power.

    Driving reservation: 6h forward cumulative driving forecast defines a dynamic
    SOC floor above soc_min, protecting driving energy from being traded away.
    """
    pch_max  = data["pch_max"]
    pdis_max = data["pdis_max"]
    edrive   = data["edrive"]
    soc_max  = data["soc_max"]
    soc_min  = data["soc_min"]
    soc_init = data["soc_init"]
    T        = len(edrive)
    glimit   = float(grid_limit_mw) if grid_limit_mw is not None else float(np.max(pch_max) * GRID_LIMIT_FRACTION)

    # Forward 6h moving average (requires lookahead – justified for semi-online)
    fwd_steps = int(EG_LOOKAHEAD_HOURS / DT_H)   # 24 steps
    price_ma  = (pd.Series(idp)
                 .shift(-fwd_steps)
                 .rolling(window=fwd_steps, min_periods=6)
                 .mean().ffill().values)

    # Driving forecast (seasonal + hourly)
    hours  = idx.hour.values
    months = idx.month.values
    sf = np.where(np.isin(months, [1, 11, 12]), 1.15,
         np.where(np.isin(months, [7, 8]),  0.85, 1.0))
    hf = np.where((hours >= 7)  & (hours < 9),  1.8,
        np.where((hours >= 15) & (hours < 17), 1.6,
        np.where((hours >= 9)  & (hours < 15), 1.0,
        np.where((hours >= 17) & (hours < 22), 0.8, 0.4))))
    daily_mwh      = float(edrive.sum()) / 365.0
    drive_per_step = (daily_mwh / 96.0) * hf * sf

    # 6h forward drive sum → SOC floor (reserve for next 6h of driving)
    res_steps = int(EG_DRIVE_RESERVE_H / DT_H)   # 24 steps
    pad       = np.concatenate([drive_per_step, np.zeros(res_steps)])
    drive_res = np.array([pad[t:t+res_steps].sum() for t in range(T)])

    # Peak driving hours (no opportunistic trading, but emergency still OK)
    is_peak = ((hours >= 7) & (hours < 9)) | ((hours >= 15) & (hours < 17))

    soc = soc_init
    pch_v  = np.zeros(T, dtype=float)
    pdis_v = np.zeros(T, dtype=float)
    charging_cost = discharge_revenue = total_charged = total_discharged = unmet = 0.0

    for t in range(T):
        cp  = float(idp[t])
        ma  = float(price_ma[t])

        # Dynamic SOC floor: keep enough for next 6h of driving
        soc_floor = min(soc_min + drive_res[t] / ETA_CH, soc_max * 0.82)
        room      = max(soc_max - soc, 0.0)
        available = max(soc - soc_floor, 0.0)

        pch = pdis = 0.0

        # ── Tier 0: Negative price – full-power charging (always) ────────────
        if cp < EG_NEG_PRICE_LIMIT and room > 0:
            pch = min(pch_max[t], glimit, room / (ETA_CH * DT_H))

        else:
            # ── Tier 1: Emergency charging (unconditional) ────────────────────
            if soc < (soc_floor + 20):
                deficit = soc_floor + 20 - soc
                pch = min(
                    pch_max[t] * 0.70,
                    glimit * 0.70,
                    deficit / (ETA_CH * DT_H),
                )

            # ── Tier 2: Opportunistic charge (cheap, off-peak) ────────────────
            elif (not is_peak[t]) and (cp < ma * (1 - EG_PRICE_THRESHOLD)) and room > 0:
                pch = min(
                    pch_max[t] * EG_TRADE_FRACTION,
                    glimit * EG_TRADE_FRACTION,
                    room / (ETA_CH * DT_H),
                )

            # ── Tier 3: Opportunistic discharge (expensive, off-peak) ─────────
            elif (not is_peak[t]) and (cp > ma * (1 + EG_PRICE_THRESHOLD)) and available > 0:
                pdis = min(
                    pdis_max[t] * EG_TRADE_FRACTION,
                    glimit * EG_TRADE_FRACTION,
                    available * ETA_DIS / DT_H,
                )

        pch  = max(pch,  0.0)
        pdis = max(pdis, 0.0)
        pch_v[t] = pch;  pdis_v[t] = pdis

        soc += ETA_CH * pch * DT_H - pdis * DT_H / ETA_DIS - edrive[t]
        if soc < soc_min:
            unmet += soc_min - soc;  soc = soc_min
        if soc > soc_max:
            soc = soc_max

        e_ch = pch * DT_H;  e_dis = pdis * DT_H
        charging_cost     += e_ch  * cp
        discharge_revenue += e_dis * cp
        total_charged     += e_ch
        total_discharged  += e_dis

    degradation_cost = C_DEG_EUR_PER_MWH * (total_charged + total_discharged) if deg_on else 0.0
    net_cost         = charging_cost - discharge_revenue + degradation_cost

    return {
        "scenario":              "Enhanced_Greedy_ID",
        "degradation":           deg_on,
        "profit_EUR":            -net_cost,
        "charging_cost_EUR":     charging_cost,
        "discharge_revenue_EUR": discharge_revenue,
        "degradation_cost_EUR":  degradation_cost,
        "total_charged_MWh":     total_charged,
        "total_discharged_MWh":  total_discharged,
        "unmet_MWh":             unmet,
        "_pch":  pch_v,
        "_pdis": pdis_v,
    }


print("\n[3/3] Running Enhanced Greedy ID on real data...")
t0 = time.time()
eg_nodeg = run_enhanced_greedy(data, idp, deg_on=False)
eg_deg   = run_enhanced_greedy(data, idp, deg_on=True)
eg_time  = time.time() - t0

print(f"  ✓ Enhanced Greedy (no deg):  €{eg_nodeg['profit_EUR']/1e6:.2f}M | "
      f"Charged: {eg_nodeg['total_charged_MWh']:.1f} MWh | "
      f"Unmet: {eg_nodeg['unmet_MWh']:.0f} MWh")
print(f"  ✓ Enhanced Greedy (with deg): €{eg_deg['profit_EUR']/1e6:.2f}M | "
      f"Degradation: €{eg_deg['degradation_cost_EUR']/1e6:.2f}M")
print(f"  ✓ Execution time: {eg_time:.2f}s ({float(id_deg.get('execution_time_s', 43.1))/eg_time:.0f}x faster than LP)")

# ============================================================
# 4. VALUE CAPTURE CALCULATION
# ============================================================

val_capture_nodeg = eg_nodeg["profit_EUR"] / float(id_nodeg["profit_EUR"]) * 100 if float(id_nodeg["profit_EUR"]) != 0 else 0.0
val_capture_deg   = eg_deg["profit_EUR"]   / float(id_deg["profit_EUR"])   * 100 if float(id_deg["profit_EUR"])   != 0 else 0.0

print("\n" + "─" * 60)
print("SUMMARY COMPARISON (with Battery Degradation)")
print("─" * 60)
print(f"  ID Optimization:   €{float(id_deg['profit_EUR'])/1e6:+.2f}M  |  Unmet: {float(id_deg['unmet_MWh']):.0f} MWh")
print(f"  Enhanced Greedy:   €{eg_deg['profit_EUR']/1e6:+.2f}M  |  Unmet: {eg_deg['unmet_MWh']:.0f} MWh")
print(f"  Value Capture:     {val_capture_deg:.1f}%  |  Speed: {eg_time:.2f}s vs ~43s LP")
if orig_greedy is not None:
    print(f"  Original Greedy:   €{float(orig_greedy['profit_EUR'])/1e6:+.2f}M  |  Unmet: {float(orig_greedy['unmet_MWh']):.0f} MWh (reference)")
print("─" * 60)

# ============================================================
# 4B. CLASS-LEVEL ENHANCED GREEDY ANALYSIS
# ============================================================

print("\nRunning class-level Enhanced Greedy analysis...")
class_group_data = load_class_group_data(idx, data)
class_results = []

for class_label, cdata in class_group_data.items():
    class_grid_limit = float(np.max(cdata["pch_max"]) * GRID_LIMIT_FRACTION)
    c_nodeg = run_enhanced_greedy(cdata, idp, deg_on=False, grid_limit_mw=class_grid_limit)
    c_deg = run_enhanced_greedy(cdata, idp, deg_on=True, grid_limit_mw=class_grid_limit)

    annual_drive_mwh = float(np.sum(cdata["edrive"]))
    unmet_pct = (c_deg["unmet_MWh"] / annual_drive_mwh * 100.0) if annual_drive_mwh > 0 else 0.0
    throughput = c_deg["total_charged_MWh"] + c_deg["total_discharged_MWh"]

    if c_deg["profit_EUR"] > 0 and unmet_pct < 1.0:
        market_fit = "Viable"
    elif c_deg["profit_EUR"] > 0 or unmet_pct < 3.0:
        market_fit = "Borderline"
    else:
        market_fit = "Weak"

    class_results.append({
        "Class": class_label,
        "Fleet_Size": int(cdata["fleet_size"]),
        "Battery_Capacity_MWh": float(cdata["cap"]),
        "Profit_NoDeg_EUR": float(c_nodeg["profit_EUR"]),
        "Profit_Deg_EUR": float(c_deg["profit_EUR"]),
        "Charging_Cost_EUR": float(c_deg["charging_cost_EUR"]),
        "Discharge_Revenue_EUR": float(c_deg["discharge_revenue_EUR"]),
        "Degradation_Cost_EUR": float(c_deg["degradation_cost_EUR"]),
        "Charged_MWh": float(c_deg["total_charged_MWh"]),
        "Discharged_MWh": float(c_deg["total_discharged_MWh"]),
        "Throughput_MWh": float(throughput),
        "Unmet_MWh": float(c_deg["unmet_MWh"]),
        "Driving_Demand_MWh": float(annual_drive_mwh),
        "Unmet_Pct_of_Drive": float(unmet_pct),
        "Market_Fit": market_fit,
        "Source_Classes": cdata["source_classes"],
    })

class_results.sort(key=lambda r: r["Profit_Deg_EUR"], reverse=True)

for r in class_results:
    print(f"  {r['Class']:<20} | Profit(deg): €{r['Profit_Deg_EUR']/1e6:+.2f}M | "
          f"Unmet: {r['Unmet_Pct_of_Drive']:.2f}% | Fit: {r['Market_Fit']}")

# ============================================================
# 5. PUBLICATION-QUALITY GRAPHS
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": 120,
    "savefig.dpi": 180,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "axes.titlepad": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
sns.set_theme(style="whitegrid", context="notebook")


def save_figure_pair(fig, stem: str) -> None:
    """Save one uncluttered figure in editable and presentation formats."""
    fig.savefig(OUTPUT_FOLDER / f"{stem}.png", dpi=100, bbox_inches="tight")
    fig.savefig(OUTPUT_FOLDER / f"{stem}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────
# GRAPH 1: 4-Panel Deep-Dive (Enhanced Greedy vs ID Optimization)
# ─────────────────────────────────────────────────────────────

print("\nGenerating graphs...")

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

scenarios_2 = ["Enhanced\nGreedy ID", "ID\n(Optimized)"]

# ── Panel A: Profit comparison ──
ax1 = fig.add_subplot(gs[0, 0])
profits_2 = [eg_deg["profit_EUR"], float(id_deg["profit_EUR"])]
colors_2  = [COLORS["enhanced_greedy"], COLORS["optimized_id"]]
bars1 = ax1.bar(scenarios_2, [p / 1e6 for p in profits_2],
                color=colors_2, alpha=0.88, edgecolor="black", linewidth=1.5, width=0.5)
ax1.axhline(0, color="black", linewidth=0.8)
ax1.set_ylabel("Annual Profit (€ Millions)", fontsize=11, fontweight="bold")
ax1.set_title("Panel A: Net Profit Comparison\n(with Battery Degradation Cost)",
              fontsize=12, fontweight="bold", pad=12)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(eur_million_formatter))
ax1.grid(axis="y", alpha=0.3, linestyle="--")
for bar, val in zip(bars1, profits_2):
    h = val / 1e6
    ax1.text(bar.get_x() + bar.get_width() / 2,
             h + (0.15 if h >= 0 else -0.25),
             f"€{h:+.2f}M",
             ha="center", va="bottom" if h >= 0 else "top",
             fontsize=11, fontweight="bold")

# ── Panel B: Cost / revenue decomposition ──
ax2 = fig.add_subplot(gs[0, 1])
metric_labels = ["Charging\nCost", "Discharge\nRevenue", "Degradation\nCost"]
eg_vals  = [eg_deg["charging_cost_EUR"]/1e6,
            eg_deg["discharge_revenue_EUR"]/1e6,
            eg_deg["degradation_cost_EUR"]/1e6]
opt_vals = [float(id_deg["charging_cost_EUR"])/1e6,
            float(id_deg["discharge_revenue_EUR"])/1e6,
            float(id_deg["degradation_cost_EUR"])/1e6]
x = np.arange(len(metric_labels))
w = 0.34
b_eg  = ax2.bar(x - w/2, eg_vals,  w, label="Enhanced Greedy",
                color=COLORS["enhanced_greedy"], alpha=0.88, edgecolor="black", linewidth=1.5)
b_opt = ax2.bar(x + w/2, opt_vals, w, label="ID (Optimized)",
                color=COLORS["optimized_id"],    alpha=0.88, edgecolor="black", linewidth=1.5)
ax2.set_xticks(x); ax2.set_xticklabels(metric_labels, fontsize=10)
ax2.set_ylabel("Amount (€ Millions)", fontsize=11, fontweight="bold")
ax2.set_title("Panel B: Cost & Revenue Decomposition", fontsize=12, fontweight="bold", pad=12)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:.1f}M"))
ax2.grid(axis="y", alpha=0.3, linestyle="--")
for bar, val in zip(list(b_eg) + list(b_opt), eg_vals + opt_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f"{val:.2f}", ha="center", va="bottom", fontsize=8.5)

# ── Panel C: Throughput / market activity ──
ax3 = fig.add_subplot(gs[1, 0])
tp_labels = ["Total\nCharged (MWh)", "Total\nDischarged (MWh)", "Unmet\nEnergy (MWh)"]
eg_tp  = [eg_deg["total_charged_MWh"],   eg_deg["total_discharged_MWh"],  eg_deg["unmet_MWh"]]
opt_tp = [float(id_deg["total_charged_MWh"]), float(id_deg["total_discharged_MWh"]), float(id_deg["unmet_MWh"])]
x3 = np.arange(len(tp_labels))
b_eg3  = ax3.bar(x3 - w/2, eg_tp,  w, label="Enhanced Greedy",
                 color=COLORS["enhanced_greedy"], alpha=0.88, edgecolor="black", linewidth=1.5)
b_opt3 = ax3.bar(x3 + w/2, opt_tp, w, label="ID (Optimized)",
                 color=COLORS["optimized_id"],    alpha=0.88, edgecolor="black", linewidth=1.5)
ax3.set_xticks(x3); ax3.set_xticklabels(tp_labels, fontsize=10)
ax3.set_ylabel("Energy (MWh)", fontsize=11, fontweight="bold")
ax3.set_title("Panel C: Market Throughput & Operational Reliability",
              fontsize=12, fontweight="bold", pad=12)
ax3.grid(axis="y", alpha=0.3, linestyle="--")
for bar, val in zip(list(b_eg3) + list(b_opt3), eg_tp + opt_tp):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.1f}", ha="center", va="bottom", fontsize=8.5)

# ── Panel D: Performance summary (value capture + speed + reliability) ──
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis("off")

profit_diff = eg_deg["profit_EUR"] - float(id_deg["profit_EUR"])
tp_eg  = eg_deg["total_charged_MWh"] + eg_deg["total_discharged_MWh"]
tp_opt = float(id_deg["total_charged_MWh"]) + float(id_deg["total_discharged_MWh"])

table_data = [
    ["Metric", "Enhanced Greedy", "ID Optimization", "Difference"],
    ["Net Profit (€M)",
     f"{eg_deg['profit_EUR']/1e6:+.2f}",
     f"{float(id_deg['profit_EUR'])/1e6:+.2f}",
     f"{profit_diff/1e6:+.2f}"],
    ["Value Capture",
     f"{val_capture_deg:.1f}%",
     "100% (baseline)",
     "—"],
    ["Unmet Energy (MWh)",
        f"{eg_deg['unmet_MWh']:.0f} OK" if eg_deg["unmet_MWh"] < 10 else f"{eg_deg['unmet_MWh']:.0f} ALERT",
        f"{float(id_deg['unmet_MWh']):.0f} OK",
     "—"],
    ["Throughput (MWh)",
     f"{tp_eg:.1f}",
     f"{tp_opt:.1f}",
     f"{tp_eg - tp_opt:+.1f}"],
    ["Degradation Cost",
     f"€{eg_deg['degradation_cost_EUR']/1e6:.2f}M",
     f"€{float(id_deg['degradation_cost_EUR'])/1e6:.2f}M",
     f"{(eg_deg['degradation_cost_EUR']-float(id_deg['degradation_cost_EUR']))/1e6:+.2f}M"],
    ["Execution Time",
        f"~{eg_time:.1f}s",
     "~43s",
     f"~{int(43.0/max(eg_time,0.1))}x faster"],
    ["Operational",
        "Viable" if eg_deg["unmet_MWh"] < 10 else "Check",
        "Viable",
     "—"],
]

tbl = ax4.table(
    cellText=[r[1:] for r in table_data[1:]],
    rowLabels=[r[0] for r in table_data[1:]],
    colLabels=table_data[0][1:],
    cellLoc="center", loc="center",
    bbox=[0.0, 0.0, 1.0, 1.0],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("lightgray")
    if row == 0:
        cell.set_facecolor("#0052CC")
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#EBF5FB")
    else:
        cell.set_facecolor("white")
ax4.set_title("Panel D: Performance Summary", fontsize=12, fontweight="bold", pad=12)

fig.legend([b_eg[0], b_opt[0]], ["Enhanced Greedy", "ID (Optimized)"],
           loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.935), frameon=True)

fig.suptitle(
    "Enhanced Greedy ID vs. ID Optimization",
    fontsize=15, fontweight="bold", y=0.99,
)
fig.text(0.5, 0.958, f"{FLEET_LABEL}  •  Full Year 2025  •  Real IDC Prices",
         ha="center", va="center", fontsize=11)
fig.subplots_adjust(top=0.89, bottom=0.06)

plt.close()

# Canonical public outputs: each former panel is now a separate figure.
fig_a, ax_a = plt.subplots(figsize=(9, 6))
bars = ax_a.bar(scenarios_2, [p / 1e6 for p in profits_2], color=colors_2,
                alpha=0.88, edgecolor="black", linewidth=1.5, width=0.5)
ax_a.axhline(0, color="black", linewidth=0.8)
ax_a.set_ylabel("Annual Profit (€ Millions)", fontweight="bold")
ax_a.set_title("Enhanced Greedy ID vs. ID Optimization: Net Profit\nWith Battery Degradation Cost", fontweight="bold")
ax_a.yaxis.set_major_formatter(mticker.FuncFormatter(eur_million_formatter))
ax_a.grid(axis="y", alpha=0.3, linestyle="--")
for bar, val in zip(bars, profits_2):
    h = val / 1e6
    ax_a.text(bar.get_x() + bar.get_width()/2, h + (0.15 if h >= 0 else -0.25),
              f"€{h:+.2f}M", ha="center", va="bottom" if h >= 0 else "top", fontweight="bold")
save_figure_pair(fig_a, "01a_net_profit_comparison")

fig_b, ax_b = plt.subplots(figsize=(9, 6))
bars_eg = ax_b.bar(x - w/2, eg_vals, w, label="Enhanced Greedy", color=COLORS["enhanced_greedy"], edgecolor="black")
bars_opt = ax_b.bar(x + w/2, opt_vals, w, label="ID Optimization", color=COLORS["optimized_id"], edgecolor="black")
ax_b.set_xticks(x); ax_b.set_xticklabels(metric_labels)
ax_b.set_ylabel("Amount (€ Millions)", fontweight="bold")
ax_b.set_title("Enhanced Greedy ID vs. ID Optimization: Cost and Revenue", fontweight="bold")
ax_b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:.1f}M"))
ax_b.legend(); ax_b.grid(axis="y", alpha=0.3, linestyle="--")
for bar, val in zip(list(bars_eg) + list(bars_opt), eg_vals + opt_vals):
    ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{val:.2f}", ha="center", fontsize=9)
save_figure_pair(fig_b, "01b_cost_revenue_decomposition")

fig_c, ax_c = plt.subplots(figsize=(9, 6))
bars_eg3 = ax_c.bar(x3 - w/2, eg_tp, w, label="Enhanced Greedy", color=COLORS["enhanced_greedy"], edgecolor="black")
bars_opt3 = ax_c.bar(x3 + w/2, opt_tp, w, label="ID Optimization", color=COLORS["optimized_id"], edgecolor="black")
ax_c.set_xticks(x3); ax_c.set_xticklabels(tp_labels)
ax_c.set_ylabel("Energy (MWh)", fontweight="bold")
ax_c.set_title("Enhanced Greedy ID vs. ID Optimization: Throughput and Reliability", fontweight="bold")
ax_c.legend(); ax_c.grid(axis="y", alpha=0.3, linestyle="--")
for bar, val in zip(list(bars_eg3) + list(bars_opt3), eg_tp + opt_tp):
    ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.1f}", ha="center", fontsize=9)
save_figure_pair(fig_c, "01c_throughput_reliability")

fig_d, ax_d = plt.subplots(figsize=(10, 5.5))
ax_d.axis("off")
table = ax_d.table(cellText=[r[1:] for r in table_data[1:]],
                   rowLabels=[r[0] for r in table_data[1:]],
                   colLabels=table_data[0][1:], cellLoc="center", loc="center",
                   bbox=[0, 0.05, 1, 0.85])
table.auto_set_font_size(False); table.set_fontsize(10)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("lightgray")
    cell.set_facecolor("#0052CC" if row == 0 else ("#EBF5FB" if row % 2 == 0 else "white"))
    if row == 0:
        cell.set_text_props(color="white", fontweight="bold")
ax_d.set_title("Enhanced Greedy ID vs. ID Optimization: Performance Summary", fontweight="bold", pad=12)
save_figure_pair(fig_d, "01d_performance_summary_table")
print("  ✓ 01a-01d standalone comparison figures")

# ─────────────────────────────────────────────────────────────
# GRAPH 2: All Strategies Ranking (Lollipop)
# ─────────────────────────────────────────────────────────────

lollipop_data = [
    ("Combined (DA+ID+FCR)",  float(comb_deg["profit_EUR"]),  COLORS["combined"]),
    ("FCR",                   float(fcr_deg["profit_EUR"]),   COLORS["fcr"]),
    ("ID (Optimized)",        float(id_deg["profit_EUR"]),    COLORS["optimized_id"]),
    ("Enhanced Greedy ID",  eg_deg["profit_EUR"],           COLORS["enhanced_greedy"]),
    ("DA",                    float(da_deg["profit_EUR"]),    COLORS["da"]),
    ("Naive EDF",             float(naive_res["profit_EUR"]), COLORS["naive"]),
]
if orig_greedy is not None:
    lollipop_data.append(("Original Greedy ID",
                           float(orig_greedy["profit_EUR"]),
                           COLORS["original_greedy"]))

# Sort descending
lollipop_data.sort(key=lambda x: x[1], reverse=True)
labels   = [d[0] for d in lollipop_data]
profits  = [d[1] / 1e6 for d in lollipop_data]
dot_cols = [d[2] for d in lollipop_data]

fig, ax = plt.subplots(figsize=(14, 9))
y_pos = np.arange(len(labels))

for i, (y, p, col) in enumerate(zip(y_pos, profits, dot_cols)):
    ax.plot([0, p], [y, y], color="#CCCCCC", linewidth=2, zorder=1)
    ax.scatter(p, y, color=col, s=320, zorder=2, edgecolors="black", linewidth=1.5)
    x_offset = 0.15 if p >= 0 else -0.15
    ha = "left" if p >= 0 else "right"
    weight = "bold" if "Enhanced Greedy" in labels[i] else "normal"
    ax.text(p + x_offset, y, f"€{p:.2f}M", va="center", ha=ha,
            fontsize=10.5, fontweight=weight)

ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("Annual Net Profit (€ Millions, with Battery Degradation)",
              fontsize=12, fontweight="bold")
ax.set_title(
    f"All Strategies Ranked by Profit  |  {FLEET_LABEL}  |  €20/MWh Degradation Cost",
    fontsize=13, fontweight="bold", pad=12,
)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(eur_million_formatter))
ax.grid(axis="x", alpha=0.3, linestyle="--")
ax.set_xlim(min(profits) - 2.5, max(profits) + 3.5)

# Highlight Enhanced Greedy row
eg_idx = next(i for i, l in enumerate(labels) if "Enhanced Greedy" in l)
ax.axhspan(eg_idx - 0.45, eg_idx + 0.45,
           color=COLORS["enhanced_greedy"], alpha=0.12, zorder=0)

# Annotation box
textstr = (f"Enhanced Greedy ID:\n"
           f"  Value Capture: {val_capture_deg:.1f}%\n"
           f"  Speed: {eg_time:.1f}s ({int(43.0/max(eg_time, 0.1))}x faster)\n"
           f"  Unmet Energy: {eg_deg['unmet_MWh']:.0f} MWh")
props = dict(boxstyle="round,pad=0.5", facecolor="#EBF5FB", alpha=0.9, edgecolor="#4A90E2")
ax.text(1.02, 0.04, textstr, transform=ax.transAxes,
    fontsize=9.3, verticalalignment="bottom", horizontalalignment="left",
    clip_on=False,
        bbox=props)

fig.subplots_adjust(top=0.90, right=0.84, left=0.10, bottom=0.10)

plt.savefig(OUTPUT_FOLDER / "02_all_strategies_ranking.png",
            dpi=100, bbox_inches="tight")
plt.savefig(OUTPUT_FOLDER / "02_all_strategies_ranking.svg",
            format="svg", bbox_inches="tight")
plt.close()
print("  ✓ 02_all_strategies_ranking.png/svg")

# ─────────────────────────────────────────────────────────────
# GRAPH 3: Degradation Impact Analysis
# ─────────────────────────────────────────────────────────────

# Build "with/without degradation" pairs
dscenarios = []
for lbl, nodeg_src, deg_src, col in [
    ("ID\nOptimized",    id_nodeg,   id_deg,   COLORS["optimized_id"]),
    ("Enhanced\nGreedy", eg_nodeg,   eg_deg,   COLORS["enhanced_greedy"]),
    ("DA",               get_scenario("DA", False), da_deg, COLORS["da"]),
    ("FCR",              get_scenario("FCR", False), fcr_deg, COLORS["fcr"]),
    ("Combined",         get_scenario("Combined", False), comb_deg, COLORS["combined"]),
]:
    dscenarios.append({
        "label": lbl,
        "nodeg": float(nodeg_src["profit_EUR"]) / 1e6,
        "deg":   float(deg_src["profit_EUR"])   / 1e6,
        "color": col,
    })

x = np.arange(len(dscenarios))
w = 0.36

fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)

nodeg_bars = ax.bar(x - w/2,
                    [d["nodeg"] for d in dscenarios], w,
                    color="#E8F4F8", edgecolor="#2E86AB",
                    linewidth=2, alpha=0.92,
                    label="Without Degradation Cost")
deg_bars   = ax.bar(x + w/2,
                    [d["deg"] for d in dscenarios], w,
                    color=[d["color"] for d in dscenarios],
                    edgecolor="black", linewidth=1.5,
                    alpha=0.85, label="With Degradation Cost (€20/MWh)")

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([d["label"] for d in dscenarios], fontsize=11, rotation=0, ha="center")
ax.set_ylabel("Annual Profit (€ Millions)", fontsize=12, fontweight="bold")
ax.set_title(
    "Degradation Cost Impact by Strategy  •  Enhanced Greedy vs. All Optimization Baselines\n"
    f"{FLEET_LABEL}  |  Battery Degradation: €20/MWh Throughput Cost",
    fontsize=13, fontweight="bold", pad=15,
)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(eur_million_formatter))
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, framealpha=0.95)

for bar, d in zip(nodeg_bars, dscenarios):
    h = d["nodeg"]
    ax.text(bar.get_x() + bar.get_width()/2,
            h + (0.2 if h >= 0 else -0.5),
            f"€{h:.2f}M", ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=9, color="#1A5276")

for bar, d in zip(deg_bars, dscenarios):
    h = d["deg"]
    ax.text(bar.get_x() + bar.get_width()/2,
            h + (0.2 if h >= 0 else -0.5),
            f"€{h:.2f}M", ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=9, fontweight="bold")

plt.savefig(OUTPUT_FOLDER / "03_degradation_impact_analysis.png",
            dpi=100, bbox_inches="tight")
plt.savefig(OUTPUT_FOLDER / "03_degradation_impact_analysis.svg",
            format="svg", bbox_inches="tight")
plt.close()
print("  ✓ 03_degradation_impact_analysis.png/svg")

# ─────────────────────────────────────────────────────────────
# GRAPH 4: Value Capture & Profit Gap Analysis
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
fig.suptitle(
    "Value Capture Analysis: Enhanced Greedy ID vs. Optimization Benchmarks\n"
    f"{FLEET_LABEL}  •  With Battery Degradation Cost",
    fontsize=13, fontweight="bold", y=1.02,
)

# Left: value capture % relative to ID optimization
ax_vc = axes[0]
strategies_vc = ["Enhanced\nGreedy ID", "Combined\n(DA+ID+FCR)", "FCR\nOnly", "DA\nOnly"]
profits_vc    = [eg_deg["profit_EUR"],
                 float(comb_deg["profit_EUR"]),
                 float(fcr_deg["profit_EUR"]),
                 float(da_deg["profit_EUR"])]
id_prof       = float(id_deg["profit_EUR"])
captures_vc   = [p / id_prof * 100 for p in profits_vc]
bar_cols_vc   = [COLORS["enhanced_greedy"],
                 COLORS["combined"],
                 COLORS["fcr"],
                 COLORS["da"]]

bars_vc = ax_vc.bar(strategies_vc, captures_vc,
                    color=bar_cols_vc, alpha=0.88,
                    edgecolor="black", linewidth=1.5, width=0.5)

ax_vc.axhline(100, color=COLORS["optimized_id"], linestyle="--",
              linewidth=2, label="ID Optimization (100%)")
ax_vc.axhline(70,  color="darkorange", linestyle=":",
              linewidth=1.5, alpha=0.7, label="Literature Upper (70%)")
ax_vc.axhline(40,  color="firebrick", linestyle=":",
              linewidth=1.5, alpha=0.7, label="Literature Lower (40%)")
ax_vc.set_ylabel("Value Capture (% of ID Optimization)", fontsize=11, fontweight="bold")
ax_vc.set_title("Value Capture Relative to ID Optimization", fontsize=12, fontweight="bold")
ax_vc.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=1, framealpha=0.95)
ax_vc.grid(axis="y", alpha=0.3, linestyle="--")

for bar, vc in zip(bars_vc, captures_vc):
    ax_vc.text(bar.get_x() + bar.get_width()/2,
               bar.get_height() + 1.5,
               f"{vc:.1f}%", ha="center", va="bottom",
               fontsize=11, fontweight="bold")

# Right: absolute profit vs throughput scatter (efficiency view)
ax_sc = axes[1]
scatter_data = [
    ("Enhanced Greedy", eg_deg["total_charged_MWh"] + eg_deg["total_discharged_MWh"],
     eg_deg["profit_EUR"] / 1e6, COLORS["enhanced_greedy"], 300),
    ("ID Optimized",    float(id_deg["total_charged_MWh"]) + float(id_deg["total_discharged_MWh"]),
     float(id_deg["profit_EUR"]) / 1e6, COLORS["optimized_id"], 300),
    ("Combined",        float(comb_deg["total_charged_MWh"]) + float(comb_deg["total_discharged_MWh"]),
     float(comb_deg["profit_EUR"]) / 1e6, COLORS["combined"], 300),
    ("DA",              float(da_deg["total_charged_MWh"]) + float(da_deg["total_discharged_MWh"]),
     float(da_deg["profit_EUR"]) / 1e6, COLORS["da"], 200),
    ("FCR",             float(fcr_deg["total_charged_MWh"]) + float(fcr_deg["total_discharged_MWh"]),
     float(fcr_deg["profit_EUR"]) / 1e6, COLORS["fcr"], 200),
]
if orig_greedy is not None:
    scatter_data.append((
        "Original Greedy",
        float(orig_greedy["total_charged_MWh"]) + float(orig_greedy["total_discharged_MWh"]),
        float(orig_greedy["profit_EUR"]) / 1e6,
        COLORS["original_greedy"], 200,
    ))

for name, tp, prof, col, sz in scatter_data:
    ax_sc.scatter(tp, prof, color=col, s=sz, zorder=3,
                  edgecolors="black", linewidth=1.2, alpha=0.9)
    offset_y = 0.25 if prof >= 0 else -0.4
    ax_sc.annotate(name, (tp, prof), textcoords="offset points",
                   xytext=(6, 6), fontsize=9.5,
                   fontweight="bold" if "Enhanced" in name else "normal")

ax_sc.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax_sc.set_xlabel("Total Throughput (MWh: charged + discharged)", fontsize=11, fontweight="bold")
ax_sc.set_ylabel("Annual Profit (€ Millions)", fontsize=11, fontweight="bold")
ax_sc.set_title("Profit vs. Throughput Efficiency\n(Upper-right = most efficient)",
                fontsize=12, fontweight="bold")
ax_sc.yaxis.set_major_formatter(mticker.FuncFormatter(eur_million_formatter))
ax_sc.grid(alpha=0.3, linestyle="--")

plt.savefig(OUTPUT_FOLDER / "04_value_capture_analysis.png",
            dpi=100, bbox_inches="tight")
plt.savefig(OUTPUT_FOLDER / "04_value_capture_analysis.svg",
            format="svg", bbox_inches="tight")
plt.close()
print("  ✓ 04_value_capture_analysis.png/svg")

# ─────────────────────────────────────────────────────────────
# GRAPH 5: Monthly SOC / Trading Profile (Time-Series Sample)
# ─────────────────────────────────────────────────────────────

# Reconstruct daily aggregated metrics for Jan-Mar sample
sample_steps = min(90 * 96, len(idx))  # 90 days
sample_idx   = idx[:sample_steps]
pch_eg  = eg_deg["_pch"][:sample_steps]
pdis_eg = eg_deg["_pdis"][:sample_steps]

daily_dates    = pd.date_range(sample_idx[0], periods=90, freq="D")
daily_charged  = np.array([pch_eg[d*96:(d+1)*96].sum() * DT_H for d in range(90)])
daily_discharged = np.array([pdis_eg[d*96:(d+1)*96].sum() * DT_H for d in range(90)])
daily_id_price = np.array([float(np.nanmean(idp[d*96:(d+1)*96])) for d in range(90)])

fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
fig.suptitle(
    "Enhanced Greedy ID Trading Activity Profile (Jan-Mar 2025)",
    fontsize=14, fontweight="bold", y=0.985,
)
fig.text(0.5, 0.958, f"{FLEET_LABEL}  |  Real IDC Prices  |  With Degradation Cost",
         ha="center", va="center", fontsize=10.5)

ax_p = axes[0]
ax_p.plot(daily_dates, daily_id_price, color="#1565C0", linewidth=1.5, alpha=0.85, label="Mean Daily IDC Price")
ax_p.set_ylabel("IDC Price\n(€/MWh)", fontsize=10, fontweight="bold")
ax_p.legend(fontsize=9, framealpha=0.9, loc="upper left")
ax_p.grid(alpha=0.3, linestyle="--")

ax_ch = axes[1]
ax_ch.bar(daily_dates, daily_charged, color=COLORS["enhanced_greedy"],
          alpha=0.8, label="Charged (buy)", width=0.9)
ax_ch.set_ylabel("Energy Charged\n(MWh)", fontsize=10, fontweight="bold")
ax_ch.legend(fontsize=9, framealpha=0.9, loc="upper left")
ax_ch.grid(axis="y", alpha=0.3, linestyle="--")

ax_dis = axes[2]
ax_dis.bar(daily_dates, daily_discharged, color=COLORS["optimized_id"],
           alpha=0.8, label="Discharged (sell)", width=0.9)
ax_dis.set_ylabel("Energy Discharged\n(MWh)", fontsize=10, fontweight="bold")
ax_dis.set_xlabel("Date (2025)", fontsize=11)
ax_dis.legend(fontsize=9, framealpha=0.9, loc="upper left")
ax_dis.grid(axis="y", alpha=0.3, linestyle="--")

ax_p.text(0.99, 0.97, "Mean Daily Intraday Price", transform=ax_p.transAxes,
          ha="right", va="top", fontsize=10.5, fontweight="bold")
ax_ch.text(0.99, 0.97, "Daily Charging Activity", transform=ax_ch.transAxes,
           ha="right", va="top", fontsize=10.5, fontweight="bold")
ax_dis.text(0.99, 0.97, "Daily Discharging Activity", transform=ax_dis.transAxes,
            ha="right", va="top", fontsize=10.5, fontweight="bold")

fig.subplots_adjust(top=0.92, hspace=0.10)

plt.savefig(OUTPUT_FOLDER / "05_trading_profile_sample.png",
            dpi=100, bbox_inches="tight")
plt.savefig(OUTPUT_FOLDER / "05_trading_profile_sample.svg",
            format="svg", bbox_inches="tight")
plt.close()
print("  ✓ 05_trading_profile_sample.png/svg")

# ─────────────────────────────────────────────────────────────
# GRAPH 6: KPI Dashboard (6-Panel Summary)
# ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 13))
gs6 = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
fig.suptitle(
    "KPI Dashboard: Enhanced Greedy ID vs. All Benchmarks",
    fontsize=15, fontweight="bold", y=0.99,
)
fig.text(0.5, 0.962, f"{FLEET_LABEL}  •  Real IDC Prices  •  Battery Degradation €20/MWh",
         ha="center", va="center", fontsize=11)

kpi_names   = ["Enhanced\nGreedy ID", "ID\nOptimized", "Combined", "FCR", "DA", "Naive EDF"]
kpi_profits = [eg_deg["profit_EUR"] / 1e6,
               float(id_deg["profit_EUR"]) / 1e6,
               float(comb_deg["profit_EUR"]) / 1e6,
               float(fcr_deg["profit_EUR"]) / 1e6,
               float(da_deg["profit_EUR"]) / 1e6,
               float(naive_res["profit_EUR"]) / 1e6]
kpi_colors  = [COLORS["enhanced_greedy"], COLORS["optimized_id"], COLORS["combined"],
               COLORS["fcr"], COLORS["da"], COLORS["naive"]]

def kpi_panel(ax, values, title, ylabel, colors, fmt="{:.2f}M", fmt_scale=1.0, hline=None):
    bars = ax.bar(kpi_names, values, color=colors, alpha=0.88, edgecolor="black", linewidth=1.2)
    if hline is not None:
        ax.axhline(hline, color="red", linestyle="--", linewidth=1.3, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis="x", labelsize=8.5)
    if "€" in ylabel:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(eur_million_formatter))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for bar, v in zip(bars, values):
        h = v
        ax.text(bar.get_x() + bar.get_width()/2,
                h + (abs(max(values, default=1)) * 0.02 if h >= 0 else -abs(max(values, default=1)) * 0.04),
                fmt.format(v * fmt_scale),
                ha="center", va="bottom" if h >= 0 else "top", fontsize=8, fontweight="bold")

# Panel 1: Profit
kpi_panel(fig.add_subplot(gs6[0, 0]), kpi_profits,
          "Net Annual Profit", "€ Millions", kpi_colors, "€{:.2f}M")

# Panel 2: Value capture %
id_base = float(id_deg["profit_EUR"])
kpi_vc  = [p * 1e6 / id_base * 100 for p in kpi_profits]
ax6b = fig.add_subplot(gs6[0, 1])
kpi_panel(ax6b, kpi_vc,
          "Value Capture vs. ID Optimization", "% of ID Optimization", kpi_colors, "{:.1f}%")
ax6b.axhline(100, color=COLORS["optimized_id"], linestyle="--", linewidth=1.5, alpha=0.8)
ax6b.axhline(40,  color="darkorange", linestyle=":",   linewidth=1.5, alpha=0.6)

# Panel 3: Degradation cost
kpi_deg_cost = [eg_deg["degradation_cost_EUR"] / 1e6,
                float(id_deg["degradation_cost_EUR"]) / 1e6,
                float(comb_deg["degradation_cost_EUR"]) / 1e6,
                float(fcr_deg["degradation_cost_EUR"]) / 1e6,
                float(da_deg["degradation_cost_EUR"]) / 1e6,
                0.0]
kpi_panel(fig.add_subplot(gs6[0, 2]), kpi_deg_cost,
          "Battery Degradation Cost", "€ Millions", kpi_colors, "€{:.2f}M")

# Panel 4: Throughput charged
kpi_charged = [eg_deg["total_charged_MWh"],
               float(id_deg["total_charged_MWh"]),
               float(comb_deg["total_charged_MWh"]),
               float(fcr_deg["total_charged_MWh"]),
               float(da_deg["total_charged_MWh"]),
               float(naive_res.get("throughput_MWh", 0))]
kpi_panel(fig.add_subplot(gs6[1, 0]), kpi_charged,
          "Total Energy Charged (MWh)", "MWh", kpi_colors, "{:.0f}", fmt_scale=1.0)

# Panel 5: Discharge revenue
kpi_rev = [eg_deg["discharge_revenue_EUR"] / 1e6,
           float(id_deg["discharge_revenue_EUR"]) / 1e6,
           float(comb_deg["discharge_revenue_EUR"]) / 1e6,
           float(fcr_deg.get("discharge_revenue_EUR", 0)) / 1e6,
           float(da_deg["discharge_revenue_EUR"]) / 1e6,
           0.0]
kpi_panel(fig.add_subplot(gs6[1, 1]), kpi_rev,
          "Discharge Revenue", "€ Millions", kpi_colors, "€{:.2f}M")

# Panel 6: Charging cost
kpi_chcost = [eg_deg["charging_cost_EUR"] / 1e6,
              float(id_deg["charging_cost_EUR"]) / 1e6,
              float(comb_deg["charging_cost_EUR"]) / 1e6,
              float(fcr_deg["charging_cost_EUR"]) / 1e6,
              float(da_deg["charging_cost_EUR"]) / 1e6,
              float(naive_res["charging_cost_EUR"]) / 1e6]
kpi_panel(fig.add_subplot(gs6[1, 2]), kpi_chcost,
          "Charging Cost", "€ Millions", kpi_colors, "€{:.2f}M")

fig.subplots_adjust(top=0.90, bottom=0.05)

plt.close()

# Canonical public outputs: one KPI per figure to avoid dashboard overload.
kpi_specs = [
    ("06a_kpi_net_annual_profit", kpi_profits, "Net Annual Profit", "€ Millions", "€{:.2f}M", 1.0),
    ("06b_kpi_value_capture", kpi_vc, "Value Capture vs. ID Optimization", "% of ID Optimization", "{:.1f}%", 1.0),
    ("06c_kpi_degradation_cost", kpi_deg_cost, "Battery Degradation Cost", "€ Millions", "€{:.2f}M", 1.0),
    ("06d_kpi_energy_charged", kpi_charged, "Total Energy Charged", "MWh", "{:.0f}", 1.0),
    ("06e_kpi_discharge_revenue", kpi_rev, "Discharge Revenue", "€ Millions", "€{:.2f}M", 1.0),
    ("06f_kpi_charging_cost", kpi_chcost, "Charging Cost", "€ Millions", "€{:.2f}M", 1.0),
]
for stem, values, title, ylabel, label_fmt, scale in kpi_specs:
    fig_kpi, ax_kpi = plt.subplots(figsize=(10, 6))
    bars = ax_kpi.bar(kpi_names, values, color=kpi_colors, alpha=0.88, edgecolor="black", linewidth=1.2)
    ax_kpi.axhline(0, color="black", linewidth=0.7)
    ax_kpi.set_title(title, fontsize=13, fontweight="bold")
    ax_kpi.set_ylabel(ylabel, fontweight="bold")
    ax_kpi.grid(axis="y", alpha=0.3, linestyle="--")
    if "€" in ylabel:
        ax_kpi.yaxis.set_major_formatter(mticker.FuncFormatter(eur_million_formatter))
    for bar, value in zip(bars, values):
        offset = max(abs(max(values, default=1)), 1) * 0.02
        ax_kpi.text(bar.get_x() + bar.get_width()/2,
                    value + (offset if value >= 0 else -offset),
                    label_fmt.format(value * scale), ha="center",
                    va="bottom" if value >= 0 else "top", fontsize=9, fontweight="bold")
    save_figure_pair(fig_kpi, stem)
print("  ✓ 06a-06f standalone KPI figures")

# ─────────────────────────────────────────────────────────────
# GRAPH 7: Vehicle-Class Performance in Enhanced Greedy
# ─────────────────────────────────────────────────────────────

if class_results:
    class_labels = [r["Class"] for r in class_results]
    prof_nodeg = [r["Profit_NoDeg_EUR"] / 1e6 for r in class_results]
    prof_deg = [r["Profit_Deg_EUR"] / 1e6 for r in class_results]
    unmet_pct = [r["Unmet_Pct_of_Drive"] for r in class_results]
    charged = [r["Charged_MWh"] for r in class_results]
    discharged = [r["Discharged_MWh"] for r in class_results]

    fit_color = {"Viable": "#2ECC71", "Borderline": "#F39C12", "Weak": "#E74C3C"}

    fig = plt.figure(figsize=(17, 11))
    gs7 = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
    fig.suptitle(
        "Enhanced Greedy by Vehicle Class: Market Readiness Assessment",
        fontsize=15, fontweight="bold", y=0.99,
    )
    fig.text(0.5, 0.962,
             f"{FLEET_LABEL}  •  Class focus: StreetScooter vans, EV trucks (aggregated), E-trikes",
             ha="center", va="center", fontsize=11)

    x7 = np.arange(len(class_labels))
    w7 = 0.36

    ax7a = fig.add_subplot(gs7[0, 0])
    b7a1 = ax7a.bar(x7 - w7/2, prof_nodeg, width=w7, color="#D6EAF8", edgecolor="#1F618D", linewidth=1.4,
                    label="Without degradation")
    b7a2 = ax7a.bar(x7 + w7/2, prof_deg, width=w7, color="#2E86C1", edgecolor="black", linewidth=1.2,
                    label="With degradation")
    ax7a.axhline(0, color="black", linewidth=0.8)
    ax7a.set_xticks(x7)
    ax7a.set_xticklabels(class_labels, fontsize=10)
    ax7a.set_ylabel("Profit (€ Millions)", fontsize=10.5, fontweight="bold")
    ax7a.set_title("Panel A: Profitability by Class", fontsize=12, fontweight="bold")
    ax7a.yaxis.set_major_formatter(mticker.FuncFormatter(eur_million_formatter))
    ax7a.legend(loc="upper left", framealpha=0.95)
    ax7a.grid(axis="y", alpha=0.3, linestyle="--")
    for bar in list(b7a1) + list(b7a2):
        h = bar.get_height()
        ax7a.text(bar.get_x() + bar.get_width()/2, h + (0.08 if h >= 0 else -0.10),
                  f"{h:.2f}", ha="center", va="bottom" if h >= 0 else "top", fontsize=8.5)

    ax7b = fig.add_subplot(gs7[0, 1])
    b7b = ax7b.bar(class_labels, unmet_pct,
                   color=[fit_color[r["Market_Fit"]] for r in class_results],
                   edgecolor="black", linewidth=1.2, alpha=0.9)
    ax7b.set_ylabel("Unmet / Driving Demand (%)", fontsize=10.5, fontweight="bold")
    ax7b.set_title("Panel B: Reliability Risk", fontsize=12, fontweight="bold")
    ax7b.axhline(1.0, color="#F39C12", linewidth=1.3, linestyle=":", alpha=0.8)
    ax7b.axhline(3.0, color="#C0392B", linewidth=1.3, linestyle=":", alpha=0.8)
    ax7b.grid(axis="y", alpha=0.3, linestyle="--")
    for bar, val in zip(b7b, unmet_pct):
        ax7b.text(bar.get_x() + bar.get_width()/2, val + 0.08, f"{val:.2f}%",
                  ha="center", va="bottom", fontsize=9)

    ax7c = fig.add_subplot(gs7[1, 0])
    b7c1 = ax7c.bar(x7 - w7/2, charged, width=w7, color="#85C1E9", edgecolor="black", linewidth=1.1,
                    label="Charged")
    b7c2 = ax7c.bar(x7 + w7/2, discharged, width=w7, color="#1B4F72", edgecolor="black", linewidth=1.1,
                    label="Discharged")
    ax7c.set_xticks(x7)
    ax7c.set_xticklabels(class_labels, fontsize=10)
    ax7c.set_ylabel("Energy (MWh)", fontsize=10.5, fontweight="bold")
    ax7c.set_title("Panel C: Market Activity (Annual)", fontsize=12, fontweight="bold")
    ax7c.legend(loc="upper left", framealpha=0.95)
    ax7c.grid(axis="y", alpha=0.3, linestyle="--")
    for bar in list(b7c1) + list(b7c2):
        h = bar.get_height()
        ax7c.text(bar.get_x() + bar.get_width()/2, h + max(charged + discharged) * 0.015,
                  f"{h:.0f}", ha="center", va="bottom", fontsize=8)

    ax7d = fig.add_subplot(gs7[1, 1])
    ax7d.axis("off")
    class_table = [[
        r["Class"],
        f"{r['Fleet_Size']:,}",
        f"€{r['Profit_Deg_EUR']/1e6:+.2f}M",
        f"{r['Unmet_Pct_of_Drive']:.2f}%",
        r["Market_Fit"],
    ] for r in class_results]

    tbl7 = ax7d.table(
        cellText=class_table,
        colLabels=["Class", "Fleet", "Profit (deg)", "Unmet %", "Market fit"],
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.08, 1.0, 0.88],
    )
    tbl7.auto_set_font_size(False)
    tbl7.set_fontsize(9.2)
    for (row, col), cell in tbl7.get_celld().items():
        cell.set_edgecolor("lightgray")
        if row == 0:
            cell.set_facecolor("#0052CC")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F7FBFF")
        else:
            cell.set_facecolor("white")

        if row > 0 and col == 4:
            fit_val = class_table[row - 1][4]
            if fit_val in fit_color:
                cell.set_facecolor(fit_color[fit_val])
                cell.set_text_props(color="white", fontweight="bold")

    ax7d.set_title("Panel D: Decision Summary", fontsize=12, fontweight="bold", pad=10)
    fig.subplots_adjust(top=0.90, bottom=0.05)

    plt.close()

    # Canonical public outputs: split each former class-performance panel.
    fig_7a, ax_7a = plt.subplots(figsize=(10, 6))
    bars_7a = ax_7a.bar(x7 - w7/2, prof_nodeg, width=w7, color="#D6EAF8", edgecolor="#1F618D", label="Without degradation")
    bars_7a2 = ax_7a.bar(x7 + w7/2, prof_deg, width=w7, color="#2E86C1", edgecolor="black", label="With degradation")
    ax_7a.axhline(0, color="black", linewidth=0.8); ax_7a.set_xticks(x7); ax_7a.set_xticklabels(class_labels)
    ax_7a.set_ylabel("Profit (€ Millions)", fontweight="bold"); ax_7a.set_title("Vehicle-Class Profitability", fontweight="bold")
    ax_7a.yaxis.set_major_formatter(mticker.FuncFormatter(eur_million_formatter)); ax_7a.legend(); ax_7a.grid(axis="y", alpha=0.3, linestyle="--")
    save_figure_pair(fig_7a, "07a_class_profitability")

    fig_7b, ax_7b = plt.subplots(figsize=(10, 6))
    bars_7b = ax_7b.bar(class_labels, unmet_pct, color=[fit_color[r["Market_Fit"]] for r in class_results], edgecolor="black")
    ax_7b.set_ylabel("Unmet / Driving Demand (%)", fontweight="bold"); ax_7b.set_title("Vehicle-Class Reliability Risk", fontweight="bold")
    ax_7b.axhline(1.0, color="#F39C12", linestyle=":"); ax_7b.axhline(3.0, color="#C0392B", linestyle=":"); ax_7b.grid(axis="y", alpha=0.3, linestyle="--")
    for bar, value in zip(bars_7b, unmet_pct):
        ax_7b.text(bar.get_x() + bar.get_width()/2, value + 0.08, f"{value:.2f}%", ha="center", fontsize=9)
    save_figure_pair(fig_7b, "07b_class_reliability_risk")

    fig_7c, ax_7c = plt.subplots(figsize=(10, 6))
    bars_7c1 = ax_7c.bar(x7 - w7/2, charged, width=w7, color="#85C1E9", edgecolor="black", label="Charged")
    bars_7c2 = ax_7c.bar(x7 + w7/2, discharged, width=w7, color="#1B4F72", edgecolor="black", label="Discharged")
    ax_7c.set_xticks(x7); ax_7c.set_xticklabels(class_labels); ax_7c.set_ylabel("Energy (MWh)", fontweight="bold")
    ax_7c.set_title("Vehicle-Class Annual Market Activity", fontweight="bold"); ax_7c.legend(); ax_7c.grid(axis="y", alpha=0.3, linestyle="--")
    save_figure_pair(fig_7c, "07c_class_market_activity")

    fig_7d, ax_7d = plt.subplots(figsize=(11, 5.5)); ax_7d.axis("off")
    table_7 = ax_7d.table(cellText=class_table, colLabels=["Class", "Fleet", "Profit (deg)", "Unmet %", "Market fit"], cellLoc="center", loc="center", bbox=[0, 0.05, 1, 0.82])
    table_7.auto_set_font_size(False); table_7.set_fontsize(10)
    for (row, col), cell in table_7.get_celld().items():
        cell.set_edgecolor("lightgray")
        cell.set_facecolor("#0052CC" if row == 0 else ("#F7FBFF" if row % 2 == 0 else "white"))
        if row == 0:
            cell.set_text_props(color="white", fontweight="bold")
        if row > 0 and col == 4 and class_table[row - 1][4] in fit_color:
            cell.set_facecolor(fit_color[class_table[row - 1][4]])
            cell.set_text_props(color="white", fontweight="bold")
    ax_7d.set_title("Vehicle-Class Market Readiness Summary", fontweight="bold", pad=12)
    save_figure_pair(fig_7d, "07d_class_decision_summary")
    print("  ✓ 07a-07d standalone class-performance figures")

# ============================================================
# 6. EXCEL EXPORT
# ============================================================

print("\nExporting Excel workbook...")

rows = [
    {"Strategy": "Enhanced Greedy ID (deg)",    "Profit_EUR_M": eg_deg["profit_EUR"]/1e6,
     "Charging_Cost_EUR_M": eg_deg["charging_cost_EUR"]/1e6,
     "Discharge_Revenue_EUR_M": eg_deg["discharge_revenue_EUR"]/1e6,
     "Degradation_Cost_EUR_M": eg_deg["degradation_cost_EUR"]/1e6,
     "Total_Charged_MWh": eg_deg["total_charged_MWh"],
     "Total_Discharged_MWh": eg_deg["total_discharged_MWh"],
     "Unmet_Energy_MWh": eg_deg["unmet_MWh"],
     "Value_Capture_Pct": val_capture_deg,
     "Execution_Time_s": eg_time,
     "Lookahead_Hours": EG_LOOKAHEAD_HOURS,
    "SOC_Reserve_MWh": f"{EG_DRIVE_RESERVE_H}h reserve"},
    {"Strategy": "Enhanced Greedy ID (no deg)", "Profit_EUR_M": eg_nodeg["profit_EUR"]/1e6,
     "Charging_Cost_EUR_M": eg_nodeg["charging_cost_EUR"]/1e6,
     "Discharge_Revenue_EUR_M": eg_nodeg["discharge_revenue_EUR"]/1e6,
     "Degradation_Cost_EUR_M": 0.0,
     "Total_Charged_MWh": eg_nodeg["total_charged_MWh"],
     "Total_Discharged_MWh": eg_nodeg["total_discharged_MWh"],
     "Unmet_Energy_MWh": eg_nodeg["unmet_MWh"],
     "Value_Capture_Pct": val_capture_nodeg,
     "Execution_Time_s": eg_time,
     "Lookahead_Hours": EG_LOOKAHEAD_HOURS,
    "SOC_Reserve_MWh": f"{EG_DRIVE_RESERVE_H}h reserve"},
    {"Strategy": "ID Optimization (deg)",        "Profit_EUR_M": float(id_deg["profit_EUR"])/1e6,
     "Charging_Cost_EUR_M": float(id_deg["charging_cost_EUR"])/1e6,
     "Discharge_Revenue_EUR_M": float(id_deg["discharge_revenue_EUR"])/1e6,
     "Degradation_Cost_EUR_M": float(id_deg["degradation_cost_EUR"])/1e6,
     "Total_Charged_MWh": float(id_deg["total_charged_MWh"]),
     "Total_Discharged_MWh": float(id_deg["total_discharged_MWh"]),
     "Unmet_Energy_MWh": float(id_deg["unmet_MWh"]),
     "Value_Capture_Pct": 100.0,
     "Execution_Time_s": 43.1,
     "Lookahead_Hours": 8760,
     "SOC_Reserve_MWh": 0},
    {"Strategy": "ID Optimization (no deg)",     "Profit_EUR_M": float(id_nodeg["profit_EUR"])/1e6,
     "Charging_Cost_EUR_M": float(id_nodeg["charging_cost_EUR"])/1e6,
     "Discharge_Revenue_EUR_M": float(id_nodeg["discharge_revenue_EUR"])/1e6,
     "Degradation_Cost_EUR_M": 0.0,
     "Total_Charged_MWh": float(id_nodeg["total_charged_MWh"]),
     "Total_Discharged_MWh": float(id_nodeg["total_discharged_MWh"]),
     "Unmet_Energy_MWh": float(id_nodeg["unmet_MWh"]),
     "Value_Capture_Pct": 100.0,
     "Execution_Time_s": 43.1,
     "Lookahead_Hours": 8760,
     "SOC_Reserve_MWh": 0},
]
if orig_greedy is not None:
    rows.append({
        "Strategy": "Original Greedy ID (deg, reference)",
        "Profit_EUR_M": float(orig_greedy["profit_EUR"])/1e6,
        "Charging_Cost_EUR_M": float(orig_greedy["charging_cost_EUR"])/1e6,
        "Discharge_Revenue_EUR_M": float(orig_greedy["discharge_revenue_EUR"])/1e6,
        "Degradation_Cost_EUR_M": float(orig_greedy["degradation_cost_EUR"])/1e6,
        "Total_Charged_MWh": float(orig_greedy["total_charged_MWh"]),
        "Total_Discharged_MWh": float(orig_greedy["total_discharged_MWh"]),
        "Unmet_Energy_MWh": float(orig_greedy["unmet_MWh"]),
        "Value_Capture_Pct": float(orig_greedy["profit_EUR"]) / float(id_deg["profit_EUR"]) * 100,
        "Execution_Time_s": 0.4,
        "Lookahead_Hours": 2,
        "SOC_Reserve_MWh": 0,
    })

df_out = pd.DataFrame(rows)

df_class_out = pd.DataFrame(class_results)
if not df_class_out.empty:
    df_class_out["Profit_NoDeg_EUR_M"] = df_class_out["Profit_NoDeg_EUR"] / 1e6
    df_class_out["Profit_Deg_EUR_M"] = df_class_out["Profit_Deg_EUR"] / 1e6
    df_class_out["Charging_Cost_EUR_M"] = df_class_out["Charging_Cost_EUR"] / 1e6
    df_class_out["Discharge_Revenue_EUR_M"] = df_class_out["Discharge_Revenue_EUR"] / 1e6
    df_class_out["Degradation_Cost_EUR_M"] = df_class_out["Degradation_Cost_EUR"] / 1e6
    df_class_out = df_class_out[[
        "Class", "Source_Classes", "Fleet_Size", "Battery_Capacity_MWh",
        "Profit_NoDeg_EUR_M", "Profit_Deg_EUR_M",
        "Charging_Cost_EUR_M", "Discharge_Revenue_EUR_M", "Degradation_Cost_EUR_M",
        "Charged_MWh", "Discharged_MWh", "Throughput_MWh",
        "Driving_Demand_MWh", "Unmet_MWh", "Unmet_Pct_of_Drive", "Market_Fit",
    ]]

config_rows = [
    {"Parameter": "Fleet Capacity (MWh)",        "Value": data["cap"]},
    {"Parameter": "SOC Min (MWh)",                "Value": data["soc_min"]},
    {"Parameter": "SOC Max (MWh)",                "Value": data["soc_max"]},
    {"Parameter": "Grid Limit (MW)",              "Value": round(grid_limit, 1)},
    {"Parameter": "Charging Efficiency",          "Value": ETA_CH},
    {"Parameter": "Discharging Efficiency",       "Value": ETA_DIS},
    {"Parameter": "Degradation Cost (€/MWh)",    "Value": C_DEG_EUR_PER_MWH},
    {"Parameter": "EG Lookahead (hours)",         "Value": EG_LOOKAHEAD_HOURS},
    {"Parameter": "EG Drive Reserve (hours)",      "Value": EG_DRIVE_RESERVE_H},
    {"Parameter": "EG Price Threshold (%)",         "Value": EG_PRICE_THRESHOLD},
    {"Parameter": "EG Trade Fraction",              "Value": EG_TRADE_FRACTION},
    {"Parameter": "EG Neg Price Limit (€/MWh)",     "Value": EG_NEG_PRICE_LIMIT},
    {"Parameter": "Timestep Resolution (h)",      "Value": DT_H},
    {"Parameter": "Total Timesteps",              "Value": len(idx)},
    {"Parameter": "Annual Driving Energy (MWh)",  "Value": round(float(data["edrive"].sum()), 1)},
]
df_config = pd.DataFrame(config_rows)

xls_path = OUTPUT_FOLDER / f"enhanced_greedy_vs_id_{RUN_STAMP}.xlsx"
with pd.ExcelWriter(xls_path, engine="openpyxl") as writer:
    df_out.to_excel(writer, sheet_name="Summary", index=False)
    df_config.to_excel(writer, sheet_name="Configuration", index=False)
    if not df_class_out.empty:
        df_class_out.to_excel(writer, sheet_name="Class_Performance", index=False)
print(f"  ✓ Excel: {xls_path.name}")

# ============================================================
# 7. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print(f"\n{'Strategy':<35} {'Profit':>10}  {'Unmet':>8}  {'Val.Capture':>12}  {'Throughput':>12}")
print("─" * 80)
print(f"{'Enhanced Greedy ID (with deg)':<35} "
      f"€{eg_deg['profit_EUR']/1e6:+7.2f}M  "
      f"{eg_deg['unmet_MWh']:>7.0f}  "
      f"{val_capture_deg:>11.1f}%  "
      f"{eg_deg['total_charged_MWh']+eg_deg['total_discharged_MWh']:>9.1f} MWh")
print(f"{'ID Optimization (with deg)':<35} "
      f"€{float(id_deg['profit_EUR'])/1e6:+7.2f}M  "
      f"{float(id_deg['unmet_MWh']):>7.0f}  "
      f"{'100.0%':>12}  "
      f"{float(id_deg['total_charged_MWh'])+float(id_deg['total_discharged_MWh']):>9.1f} MWh")
if orig_greedy is not None:
    orig_vc = float(orig_greedy["profit_EUR"]) / float(id_deg["profit_EUR"]) * 100
    print(f"{'Original Greedy (ref, with deg)':<35} "
          f"€{float(orig_greedy['profit_EUR'])/1e6:+7.2f}M  "
          f"{float(orig_greedy['unmet_MWh']):>7.0f}  "
          f"{orig_vc:>11.1f}%  "
          f"{float(orig_greedy['total_charged_MWh'])+float(orig_greedy['total_discharged_MWh']):>9.1f} MWh")
print("─" * 80)

if class_results:
    print("\nVehicle-class results (Enhanced Greedy, with degradation)")
    print(f"{'Class':<22} {'Fleet':>8}  {'Profit':>10}  {'Unmet%':>8}  {'Market fit':>10}")
    print("─" * 80)
    for r in class_results:
        print(f"{r['Class']:<22} {r['Fleet_Size']:>8,}  €{r['Profit_Deg_EUR']/1e6:+7.2f}M  "
              f"{r['Unmet_Pct_of_Drive']:>7.2f}%  {r['Market_Fit']:>10}")
    print("─" * 80)

print(f"\nEnhanced Greedy execution time : {eg_time:.2f}s")
print(f"Value capture vs ID Optimization: {val_capture_deg:.1f}%")
print(f"Unmet energy:                     {eg_deg['unmet_MWh']:.0f} MWh"
      f"  ({'OPERATIONAL ✓' if eg_deg['unmet_MWh'] < 10 else 'CHECK ⚠'})")
print(f"\nOutputs saved to: {OUTPUT_FOLDER}")
print("\n✅ DONE")

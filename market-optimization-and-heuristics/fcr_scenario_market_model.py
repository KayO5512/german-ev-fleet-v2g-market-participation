# -*- coding: utf-8 -*-
"""
DHL EV fleet VPP: market optimization baseline, greedy intraday heuristic, and naive EDF baseline.

Three models, built to be compared:

1. Naive EDF baseline. Price-blind, unidirectional, no markets. Charges greedily
   to full whenever the fleet is plugged in, never discharges, never trades. This
   is the weak lower bound your optimized model is meant to beat.

2. Greedy Intraday Heuristic. Online lookahead algorithm for real-time trading in
   the intraday market. Uses a 2-hour moving average price window to make buy/sell
   decisions at each 15-min step (Srikrishnan & Clack 2018, Byrne et al. 2018).
   Respects all physical constraints. No perfect foresight. Expected to capture
   ~40-70% of optimization value while running in real-time (O(T) complexity vs
   O(T^3+) for LP).
     Greedy_ID : intraday arbitrage via online lookahead heuristic

3. Optimization baseline. An aggregate single-battery LP run four ways, matching
   your six-bar chart:
     DA       : energy arbitrage in the day-ahead market only
     ID       : energy arbitrage in the intraday market only (perfect foresight)
     FCR      : frequency containment reserve capacity only, no energy arbitrage
     Combined : DA and ID arbitrage plus FCR, sharing one SOC envelope
   Each is run twice, with and without a battery degradation cost, so you can see
   how much the headline depends on FCR being treated as costless.

Resolution: 15 minutes (35040 steps over 2025), so dt = 0.25 h.
Prices: day-ahead (ENTSO-E Sequence 1, hourly product), intraday (IDC index,
native 15-min), FCR (German symmetric NEGPOS settlement capacity price, 4-hour
blocks broadcast to 15-min).

The FCR symmetric reservation is the constraint that governs the result: to sell
R_fcr MW over a 4-hour block the battery must hold R_fcr * 0.25h of energy headroom
in both directions for every step of the block, so FCR competes with arbitrage for
the same SOC envelope. That is why the combined case is not the sum of the parts.
"""

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec


# ============================================================
# 1. SETTINGS
# ============================================================


MOBILITY_WORKBOOK = Path(r"C:\Users\Yaw\Documents\EV_BID_Final\dhl_fleet_model_output_2026-06-29_00-58-34.xlsx")
DA_FILE  = Path(r"C:\Users\Yaw\Documents\EV_BID_Final\Data Ahead_ENERGY_PRICES_2025.xlsx")
ID_FILE  = Path(r"C:\Users\Yaw\Documents\EV_BID_Final\ID-Prices_label_2025.xlsx")
FCR_FILE = Path(r"C:\Users\Yaw\Documents\EV_BID_Final\FCR_yearly data_RESULT_OVERVIEW_CAPACITY_MARKET_2025-01-01_2025-12-31.xlsx")
RUN_STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FOLDER = Path(r"C:\Users\Yaw\Documents\EV_BID_Final\fcr_realmarket_scenario")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Time.
YEAR = 2025
DT_H = 0.25                 # 15-minute steps
HORIZON_DAYS = None         # None = full year. Set to e.g. 14 for a quick test run.

# Physical.
ETA_CH = 0.95              # charging efficiency
ETA_DIS = 0.95             # discharging efficiency
SOC_INIT_SHARE = 0.60      # initial and terminal SOC floor, fraction of capacity
SOC_MIN_SHARE_CLASS = 0.20
SOC_MAX_SHARE_CLASS = 0.95

# FCR.
TRES_FCR_H = 0.25          # symmetric energy reservation duration (German LER, 15 min)

# Realistic FCR penalties and acceptance assumptions.
FCR_REQUIRED_RESPONSE_TIME_S = 30.0
FCR_ASSUMED_RESPONSE_DELAY_S = 5.0
FCR_EFFECTIVE_RESPONSE_WINDOW_S = 25.0

FCR_ACCEPTANCE_RATE = 0.80
FCR_AGGREGATOR_FEE_SHARE = 0.15

FCR_ACTIVATION_COST_EUR_PER_MW_BLOCK = 2.0
FCR_RAMP_DELAY_PENALTY_EUR_PER_MW_BLOCK = 5.0

# Policy presets used for baseline-versus-penalty comparison.
FCR_POLICY_BASELINE = {
    "name": "Baseline_NoPenalty",
    "acceptance_rate": 1.00,
    "aggregator_fee_share": 0.00,
    "activation_cost_eur_per_mw_block": 0.00,
    "ramp_delay_penalty_eur_per_mw_block": 0.00,
}

FCR_POLICY_PENALTY = {
    "name": "Penalty_Realistic",
    "acceptance_rate": FCR_ACCEPTANCE_RATE,
    "aggregator_fee_share": FCR_AGGREGATOR_FEE_SHARE,
    "activation_cost_eur_per_mw_block": FCR_ACTIVATION_COST_EUR_PER_MW_BLOCK,
    "ramp_delay_penalty_eur_per_mw_block": FCR_RAMP_DELAY_PENALTY_EUR_PER_MW_BLOCK,
}

FCR_POLICIES = [FCR_POLICY_BASELINE, FCR_POLICY_PENALTY]

# Battery degradation (toggle handled per run).
C_DEG_EUR_PER_MWH = 20.0   # linear throughput cost, from pack cost over cycle life

# Grid connection limit, fraction of peak available power (depot transformer proxy).
GRID_LIMIT_FRACTION = 0.80

# Penalty for unmet driving energy, keeps the LP feasible and flags shortfalls.
UNMET_PENALTY = 1.0e6

SOLVER = "appsi_highs"

CLASS_SCOPE_DEFINITIONS = [
    {
        "label": "StreetScooter vans",
        "slug": "streetscooter_vans",
        "classes": ["streetscooter_last_mile_e_van"],
    },
    {
        "label": "EV trucks (all)",
        "slug": "ev_trucks_all",
        "classes": [
            "volvo_fl_electric_4x2_berlin",
            "other_heavy_duty_e_truck",
            "mercedes_benz_eactros_600_long_haul",
        ],
    },
    {
        "label": "E-trikes",
        "slug": "e_trikes",
        "classes": ["e_trike"],
    },
]


# ============================================================
# 2. LOAD AND ALIGN DATA TO A 15-MIN INDEX
# ============================================================

def build_index() -> pd.DatetimeIndex:
    end = f"{YEAR + 1}-01-01"
    if HORIZON_DAYS is not None:
        end = (pd.Timestamp(f"{YEAR}-01-01") + pd.Timedelta(days=HORIZON_DAYS)).strftime("%Y-%m-%d")
    return pd.date_range(f"{YEAR}-01-01 00:00", end, freq="15min", inclusive="left")


def load_mobility(idx: pd.DatetimeIndex, vehicle_classes: list[str] | None = None) -> dict:
    """Read the mobility export and upsample to 15-min.

    When vehicle_classes is omitted, load the fleet-wide aggregate sheet.
    When vehicle_classes is provided, aggregate the selected classes from the
    per-class hourly sheet so class-specific market runs stay aligned with the
    same fleet construction assumptions.
    """
    if vehicle_classes is None:
        m = pd.read_excel(MOBILITY_WORKBOOK, sheet_name="Hourly_Aggregate")
        m["datetime"] = pd.to_datetime(m["datetime"])
        m = m.set_index("datetime").sort_index()

        hourly_power = m[["p_charge_max_MW", "p_discharge_max_MW"]].reindex(idx, method="ffill")
        drive_q = (m["driving_energy_MWh"].reindex(idx, method="ffill") / 4.0)

        cap = float(m["battery_capacity_MWh_total"].iloc[0])
        soc_min = float(m["soc_min_MWh"].iloc[0])
        soc_max = float(m["soc_max_MWh"].iloc[0])
        soc_init = cap * SOC_INIT_SHARE

        return {
            "pch_max": hourly_power["p_charge_max_MW"].to_numpy(),
            "pdis_max": hourly_power["p_discharge_max_MW"].to_numpy(),
            "edrive": drive_q.to_numpy(),
            "cap": cap,
            "soc_min": soc_min,
            "soc_max": soc_max,
            "soc_init": soc_init,
        }

    class_hourly = pd.read_excel(MOBILITY_WORKBOOK, sheet_name="Hourly_By_Class")
    class_hourly["datetime"] = pd.to_datetime(class_hourly["datetime"])
    class_hourly = class_hourly[class_hourly["vehicle_class"].isin(vehicle_classes)].copy()
    if class_hourly.empty:
        raise ValueError(f"No mobility rows found for vehicle classes: {vehicle_classes}")

    grouped = class_hourly.groupby("datetime", as_index=True)[
        ["p_charge_max_MW", "p_discharge_max_MW", "driving_energy_MWh"]
    ].sum().sort_index()

    hourly_power = grouped[["p_charge_max_MW", "p_discharge_max_MW"]].reindex(idx, method="ffill")
    drive_q = (grouped["driving_energy_MWh"].reindex(idx, method="ffill") / 4.0)

    class_summary = pd.read_excel(MOBILITY_WORKBOOK, sheet_name="Class_Summary")
    class_summary = class_summary[class_summary["vehicle_class"].isin(vehicle_classes)].copy()
    if class_summary.empty:
        raise ValueError(f"No class summary rows found for vehicle classes: {vehicle_classes}")

    cap = float(class_summary["total_battery_capacity_MWh"].sum())
    soc_min = cap * SOC_MIN_SHARE_CLASS
    soc_max = cap * SOC_MAX_SHARE_CLASS
    soc_init = cap * SOC_INIT_SHARE
    fleet_size = int(round(float(class_summary["target_count"].sum())))

    return {
        "pch_max": hourly_power["p_charge_max_MW"].to_numpy(),
        "pdis_max": hourly_power["p_discharge_max_MW"].to_numpy(),
        "edrive": drive_q.to_numpy(),
        "cap": cap,
        "soc_min": soc_min,
        "soc_max": soc_max,
        "soc_init": soc_init,
        "fleet_size": fleet_size,
        "source_classes": ", ".join(vehicle_classes),
    }


def load_da(idx: pd.DatetimeIndex) -> np.ndarray:
    """ENTSO-E export. Data starts at row 7, column 0 is the MTU range, column 1
    is Sequence 1 (hourly day-ahead). Parse the interval start as the timestamp."""
    raw = pd.read_excel(DA_FILE, sheet_name="1", header=None, skiprows=7)
    start = raw[0].astype(str).str.split(" - ").str[0]
    ts = pd.to_datetime(start, errors="coerce", dayfirst=True)
    price = pd.to_numeric(raw[1], errors="coerce")
    s = pd.Series(price.values, index=ts).dropna().sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return s.reindex(idx, method="ffill").to_numpy()


def load_id(idx: pd.DatetimeIndex) -> np.ndarray:
    """Intraday continuous index (IDC), native 15-min."""
    d = pd.read_excel(ID_FILE, sheet_name="Sheet1")
    d["DateTime"] = pd.to_datetime(d["DateTime"])
    s = d.set_index("DateTime")["IDC"].astype(float).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return s.reindex(idx, method="ffill").to_numpy()


def load_fcr(idx: pd.DatetimeIndex):
    """German symmetric FCR capacity price per 4-hour block, broadcast to 15-min.
    Returns the per-step price (EUR/MW for the block) and a per-step block id."""
    f = pd.read_excel(FCR_FILE, sheet_name="001")
    gcol = "GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"
    f["DATE_FROM"] = pd.to_datetime(f["DATE_FROM"]).dt.date
    f["block_start_h"] = f["PRODUCTNAME"].str.extract(r"_(\d{2})_\d{2}$").astype(int)
    f[gcol] = pd.to_numeric(f[gcol], errors="coerce")
    lookup = {}
    for _, r in f.iterrows():
        if pd.notna(r[gcol]):
            lookup[(r["DATE_FROM"], int(r["block_start_h"]))] = float(r[gcol])

    dates = idx.date
    block_start = (idx.hour // 4) * 4
    price = np.array([lookup.get((d, int(b)), np.nan) for d, b in zip(dates, block_start)])
    price = pd.Series(price).ffill().bfill().to_numpy()
    # Unique block id per (date, block_start) for the rfcr variable.
    block_id = np.array([f"{d}_{int(b):02d}" for d, b in zip(dates, block_start)])
    return price, block_id


# ============================================================
# 3. GREEDY INTRADAY HEURISTIC (Online Lookahead Algorithm)
# ============================================================
# Academic foundation: Srikrishnan & Clack (2018), Byrne et al. (2018)
#
# This implements a real-time trading heuristic for intraday markets using a
# finite lookahead window (2-4 hours). Key differences from optimization:
#   - Optimization: solves full year LP with perfect foresight (offline)
#   - Greedy: makes sequential decisions with local price context (online)
#   - Speed: greedy is O(T), optimization is O(T^3) or worse
#   - Optimality: greedy captures ~40-70% of optimization value (empirical)
#
# Decision rule at each timestep t:
#   1. Calculate moving average of ID prices over lookahead window
#   2. If price[t] < MA[t:t+lookahead]: CHARGE (buy low)
#   3. If price[t] > MA[t:t+lookahead]: DISCHARGE (sell high)
#   4. Respect SOC bounds [soc_min, soc_max] and power limits
#   5. Prioritize driving energy, then arbitrage

def run_greedy_intraday(data: dict, idp: np.ndarray, grid_limit: float,
                        deg_on: bool, lookahead_hours: float = 2.0) -> dict:
    """
    Online greedy intraday heuristic using finite lookahead.
    
    Args:
        data: Fleet mobility data (pch_max, pdis_max, edrive, SOC bounds)
        idp: Intraday market prices (15-min resolution)
        grid_limit: Grid connection limit (MW)
        deg_on: Include battery degradation cost
        lookahead_hours: Lookahead window for price averaging (hours)
    
    Returns:
        dict with profit, costs, throughput, and time series (_pch, _pdis)
    """
    pch_max = data["pch_max"]
    pdis_max = data["pdis_max"]
    edrive = data["edrive"]
    soc_max = data["soc_max"]
    soc_min = data["soc_min"]
    soc = data["soc_init"]
    
    T = len(edrive)
    lookahead_steps = int(lookahead_hours / DT_H)
    lookahead_steps = max(lookahead_steps, 1)
    
    pch_v = np.zeros(T, dtype=float)
    pdis_v = np.zeros(T, dtype=float)
    
    charging_cost = 0.0
    discharge_revenue = 0.0
    total_charged = 0.0
    total_discharged = 0.0
    unmet = 0.0
    
    for t in range(T):
        # 1. Compute lookahead reference price (moving average of next window)
        window_end = min(t + lookahead_steps, T)
        price_ma = float(idp[t:window_end].mean())
        current_price = float(idp[t])
        
        # 2. Satisfy driving energy demand first
        room = max(soc_max - soc, 0.0)
        pch_drive = max(0.0, min(edrive[t] / (ETA_CH * DT_H), pch_max[t]))
        
        # 3. Arbitrage decision: buy/sell based on price vs MA
        pch_arb = 0.0
        pdis_arb = 0.0
        
        room_after_drive = max(soc_max - (soc + ETA_CH * pch_drive * DT_H), 0.0)
        cap_after_drive = max(soc - (ETA_CH * pch_drive * DT_H), 0.0)
        
        if current_price < price_ma:
            # Price is low: try to charge (arbitrage buy)
            pch_arb = min(
                pch_max[t] - pch_drive,  # charger limit minus driving
                grid_limit - pch_drive,  # grid limit
                room_after_drive / (ETA_CH * DT_H) if ETA_CH > 0 else 0.0
            )
        elif current_price > price_ma:
            # Price is high: try to discharge (arbitrage sell)
            pdis_arb = min(
                pdis_max[t],
                grid_limit - pch_drive,
                cap_after_drive * ETA_DIS / DT_H
            )
        
        pch_v[t] = pch_drive + max(pch_arb, 0.0)
        pdis_v[t] = pdis_arb
        
        # 4. Update SOC and track costs
        soc_delta = ETA_CH * pch_v[t] * DT_H - pdis_v[t] * DT_H / ETA_DIS - edrive[t]
        soc += soc_delta
        
        # 5. Track shortfalls
        if soc < soc_min:
            unmet += (soc_min - soc)
            soc = soc_min
        
        # 6. Calculate market costs (ID prices only, no DA involved)
        e_ch = pch_v[t] * DT_H
        e_dis = pdis_v[t] * DT_H
        charging_cost += e_ch * current_price
        discharge_revenue += e_dis * current_price
        
        total_charged += e_ch
        total_discharged += e_dis
    
    # Calculate final metrics
    degradation_cost = (
        C_DEG_EUR_PER_MWH * (total_charged + total_discharged)
        if deg_on else 0.0
    )
    
    net_cost = charging_cost - discharge_revenue + degradation_cost
    
    return {
        "scenario": "Greedy_ID",
        "degradation": deg_on,
        "net_cost_EUR": net_cost,
        "profit_EUR": -net_cost,
        "charging_cost_EUR": charging_cost,
        "discharge_revenue_EUR": discharge_revenue,
        "fcr_revenue_EUR": 0.0,
        "fcr_revenue_gross_EUR": 0.0,
        "fcr_activation_cost_EUR": 0.0,
        "fcr_ramp_penalty_EUR": 0.0,
        "degradation_cost_EUR": degradation_cost,
        "unmet_MWh": unmet,
        "average_FCR_reserved_MW": 0.0,
        "max_FCR_reserved_MW": 0.0,
        "_pch": pch_v,
        "_pdis": pdis_v,
    }


# ============================================================
# 3B. NAIVE EDF BASELINE (price-blind, unidirectional, greedy)
# ============================================================

def run_naive(data: dict, da: np.ndarray, grid_limit: float) -> dict:
    pch_max, edrive = data["pch_max"], data["edrive"]
    soc_max, soc_min, soc = data["soc_max"], data["soc_min"], data["soc_init"]
    T = len(edrive)
    cost = 0.0
    unmet = 0.0
    throughput = 0.0
    for t in range(T):
        room = max(soc_max - soc, 0.0)
        pch = min(pch_max[t], grid_limit, room / (ETA_CH * DT_H) if ETA_CH > 0 else 0.0)
        pch = max(pch, 0.0)
        soc += ETA_CH * pch * DT_H - edrive[t]
        cost += pch * DT_H * da[t]
        throughput += pch * DT_H
        if soc < soc_min:
            unmet += (soc_min - soc)
            soc = soc_min
    return {"scenario": "Naive_EDF", "degradation": False,
            "net_cost_EUR": cost, "profit_EUR": -cost,
            "charging_cost_EUR": cost, "discharge_revenue_EUR": 0.0,
            "fcr_revenue_EUR": 0.0,
            "fcr_revenue_gross_EUR": 0.0,
            "fcr_activation_cost_EUR": 0.0,
            "fcr_ramp_penalty_EUR": 0.0,
            "degradation_cost_EUR": 0.0,
            "throughput_MWh": throughput, "unmet_MWh": unmet}


# ============================================================
# 4. OPTIMIZATION BUILDER
# ============================================================

def run_opt(data: dict, buy: np.ndarray, sell: np.ndarray, allow_discharge: bool,
            fcr_on: bool, fcr_price: np.ndarray, block_id: np.ndarray,
            deg_on: bool, grid_limit: float, label: str, fcr_policy: dict) -> dict:
    T = len(data["edrive"])
    pch_max = data["pch_max"]
    pdis_max = data["pdis_max"]
    edrive = data["edrive"]
    soc_min = data["soc_min"]
    soc_max = data["soc_max"]
    soc_init = data["soc_init"]

    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, T - 1)
    m.Tsoc = pyo.RangeSet(0, T)

    m.pch = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.pdis = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.unmet = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.soc = pyo.Var(m.Tsoc, domain=pyo.NonNegativeReals)

    # FCR blocks and one symmetric reserve-capacity decision per block.
    blocks = []
    block_pch_min = {}
    block_pdis_min = {}
    block_price = {}

    if fcr_on:
        blocks = list(dict.fromkeys(block_id))
        m.B = pyo.Set(initialize=blocks)
        m.rfcr = pyo.Var(m.B, domain=pyo.NonNegativeReals)

        for b in blocks:
            mask = block_id == b
            block_pch_min[b] = float(pch_max[mask].min())
            block_pdis_min[b] = float(pdis_max[mask].min())
            block_price[b] = float(fcr_price[mask][0])

    # Charger and grid limits.
    def ch_limit(m, t):
        return m.pch[t] <= float(pch_max[t])

    m.ch_limit = pyo.Constraint(m.T, rule=ch_limit)

    def dis_limit(m, t):
        limit = float(pdis_max[t]) if allow_discharge else 0.0
        return m.pdis[t] <= limit

    m.dis_limit = pyo.Constraint(m.T, rule=dis_limit)

    def grid_limit_rule(m, t):
        if fcr_on:
            return m.pch[t] + m.pdis[t] + m.rfcr[block_id[t]] <= grid_limit
        return m.pch[t] + m.pdis[t] <= grid_limit

    m.grid = pyo.Constraint(m.T, rule=grid_limit_rule)

    # SOC bounds.
    def soc_lo(m, t):
        return m.soc[t] >= soc_min

    def soc_hi(m, t):
        return m.soc[t] <= soc_max

    m.soc_lo = pyo.Constraint(m.Tsoc, rule=soc_lo)
    m.soc_hi = pyo.Constraint(m.Tsoc, rule=soc_hi)

    # Initial and terminal SOC.
    m.soc_init = pyo.Constraint(expr=m.soc[0] == soc_init)
    m.soc_term = pyo.Constraint(expr=m.soc[T] >= soc_init)

    # Energy balance. Unmet energy is a highly penalised feasibility slack.
    def balance(m, t):
        return m.soc[t + 1] == (
            m.soc[t]
            + ETA_CH * m.pch[t] * DT_H
            - m.pdis[t] * DT_H / ETA_DIS
            - float(edrive[t])
            + m.unmet[t]
        )

    m.balance = pyo.Constraint(m.T, rule=balance)

    # Symmetric FCR power and energy-headroom reservation.
    if fcr_on:
        bid = {t: block_id[t] for t in range(T)}

        def fcr_cap_up(m, t):
            return m.rfcr[bid[t]] <= block_pdis_min[bid[t]]

        def fcr_cap_dn(m, t):
            return m.rfcr[bid[t]] <= block_pch_min[bid[t]]

        def fcr_head_lo(m, t):
            return m.soc[t] - m.rfcr[bid[t]] * TRES_FCR_H >= soc_min

        def fcr_head_hi(m, t):
            return m.soc[t] + m.rfcr[bid[t]] * TRES_FCR_H <= soc_max

        m.fcr_cap_up = pyo.Constraint(m.T, rule=fcr_cap_up)
        m.fcr_cap_dn = pyo.Constraint(m.T, rule=fcr_cap_dn)
        m.fcr_head_lo = pyo.Constraint(m.T, rule=fcr_head_lo)
        m.fcr_head_hi = pyo.Constraint(m.T, rule=fcr_head_hi)

    # Objective: minimise net cost.
    energy_cost = sum(
        (buy[t] * m.pch[t] - sell[t] * m.pdis[t]) * DT_H
        for t in range(T)
    )

    deg_cost = (
        C_DEG_EUR_PER_MWH
        * sum((m.pch[t] + m.pdis[t]) * DT_H for t in range(T))
        if deg_on else 0.0
    )

    unmet_penalty = UNMET_PENALTY * sum(m.unmet[t] for t in range(T))

    fcr_revenue_gross = (
        sum(block_price[b] * m.rfcr[b] for b in blocks)
        if fcr_on else 0.0
    )
    acceptance_rate = float(fcr_policy["acceptance_rate"])
    aggregator_fee_share = float(fcr_policy["aggregator_fee_share"])
    activation_cost_per_mw = float(fcr_policy["activation_cost_eur_per_mw_block"])
    ramp_penalty_per_mw = float(fcr_policy["ramp_delay_penalty_eur_per_mw_block"])

    fcr_revenue_after_acceptance = fcr_revenue_gross * acceptance_rate
    fcr_revenue_after_fee = fcr_revenue_after_acceptance * (1.0 - aggregator_fee_share)
    fcr_activation_cost = (
        activation_cost_per_mw * sum(m.rfcr[b] for b in blocks)
        if fcr_on else 0.0
    )
    fcr_ramp_penalty = (
        ramp_penalty_per_mw * sum(m.rfcr[b] for b in blocks)
        if fcr_on else 0.0
    )
    fcr_revenue_net = fcr_revenue_after_fee - fcr_activation_cost - fcr_ramp_penalty

    m.obj = pyo.Objective(
        expr=energy_cost + deg_cost + unmet_penalty - fcr_revenue_net,
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory(SOLVER)
    solver.solve(m)

    # Extract optimized time series.
    pch_v = np.array([pyo.value(m.pch[t]) for t in range(T)], dtype=float)
    pdis_v = np.array([pyo.value(m.pdis[t]) for t in range(T)], dtype=float)
    unmet_v = np.array([pyo.value(m.unmet[t]) for t in range(T)], dtype=float)

    charging_cost = float((buy * pch_v * DT_H).sum())
    discharge_revenue = float((sell * pdis_v * DT_H).sum())
    total_charged = float((pch_v * DT_H).sum())
    total_discharged = float((pdis_v * DT_H).sum())
    degradation_cost = (
        C_DEG_EUR_PER_MWH * (total_charged + total_discharged)
        if deg_on else 0.0
    )

    if fcr_on:
        rfcr_by_t = np.array(
            [pyo.value(m.rfcr[block_id[t]]) for t in range(T)],
            dtype=float,
        )
        average_fcr_reserved = float(rfcr_by_t.mean())
        max_fcr_reserved = float(rfcr_by_t.max())
        fcr_revenue_gross_value = float(
            sum(block_price[b] * pyo.value(m.rfcr[b]) for b in blocks)
        )
        total_rfcr = float(sum(pyo.value(m.rfcr[b]) for b in blocks))
        fcr_activation_cost_value = activation_cost_per_mw * total_rfcr
        fcr_ramp_penalty_value = ramp_penalty_per_mw * total_rfcr
        fcr_revenue_value = (
            fcr_revenue_gross_value * acceptance_rate * (1.0 - aggregator_fee_share)
            - fcr_activation_cost_value
            - fcr_ramp_penalty_value
        )
    else:
        average_fcr_reserved = 0.0
        max_fcr_reserved = 0.0
        fcr_revenue_gross_value = 0.0
        fcr_activation_cost_value = 0.0
        fcr_ramp_penalty_value = 0.0
        fcr_revenue_value = 0.0

    net_cost = (
        charging_cost
        - discharge_revenue
        - fcr_revenue_value
        + degradation_cost
    )

    return {
        "scenario": label,
        "fcr_policy": fcr_policy["name"],
        "degradation": deg_on,
        "net_cost_EUR": net_cost,
        "profit_EUR": -net_cost,
        "charging_cost_EUR": charging_cost,
        "discharge_revenue_EUR": discharge_revenue,
        "fcr_revenue_EUR": fcr_revenue_value,
        "fcr_revenue_gross_EUR": fcr_revenue_gross_value,
        "fcr_activation_cost_EUR": fcr_activation_cost_value,
        "fcr_ramp_penalty_EUR": fcr_ramp_penalty_value,
        "degradation_cost_EUR": degradation_cost,
        "unmet_MWh": float(unmet_v.sum()),
        "average_FCR_reserved_MW": average_fcr_reserved,
        "max_FCR_reserved_MW": max_fcr_reserved,
        "_pch": pch_v,
        "_pdis": pdis_v,
    }


# ============================================================
# 5. ACADEMIC GRAPHING (Publication-Quality Figures)
# ============================================================

# Color palette: blue family (colorblind-friendly)
COLORS = {
    "naive": "#B0B0B0",           # Gray for baseline
    "greedy": "#4A90E2",          # Light blue for greedy heuristic
    "optimized": "#0052CC",       # Primary blue for optimization
    "combined": "#003D99",        # Dark blue for combined
    "fcr": "#6C5CE7",             # Purple for FCR
    "loss": "#FF6B35",            # Orange for losses
    "capacity": "#00897B",        # Teal for capacity
}

def create_greedy_vs_optimization_comparison(comparison_df: pd.DataFrame, output_folder: Path) -> None:
    """
    Create 4-panel academic comparison of Greedy_ID vs optimized ID.
    Panel 1: Profit comparison (with/without degradation)
    Panel 2: Revenue decomposition (charging cost vs discharge revenue)
    Panel 3: Market metrics (throughput, efficiency)
    Panel 4: Value capture ratio (greedy as % of optimization)
    """
    # Filter to degradation=True only (realistic scenario)
    df_deg = comparison_df[comparison_df["degradation"] == True].copy()
    
    # Extract greedy and optimized ID
    greedy = df_deg[df_deg["scenario"] == "Greedy_ID"].iloc[0] if any(df_deg["scenario"] == "Greedy_ID") else None
    opt_id = df_deg[df_deg["scenario"] == "ID"].iloc[0]
    
    if greedy is None:
        print("  ⚠️  Greedy_ID not found in comparison data, skipping graphs")
        return
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.32)
    
    # ==== Panel 1: Profit Comparison ====
    ax1 = fig.add_subplot(gs[0, 0])
    scenarios = ["Greedy_ID", "ID (Optimized)"]
    profits = [float(greedy["profit_EUR"]), float(opt_id["profit_EUR"])]
    colors_p1 = [COLORS["greedy"], COLORS["optimized"]]
    
    bars1 = ax1.bar(scenarios, profits, color=colors_p1, alpha=0.85, edgecolor="black", linewidth=1.5)
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax1.set_ylabel("Annual Profit (€)", fontsize=11, fontweight="bold")
    ax1.set_title("Panel A: Net Profit Comparison\n(with Battery Degradation Cost)", 
                  fontsize=12, fontweight="bold", pad=12)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"€{x/1e6:.1f}M"))
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, profits)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height, f"€{val/1e6:.2f}M",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    # ==== Panel 2: Revenue Decomposition ====
    ax2 = fig.add_subplot(gs[0, 1])
    
    metrics = ["Charging\nCost (€)", "Discharge\nRevenue (€)", "Degradation\nCost (€)"]
    greedy_vals = [
        float(greedy["charging_cost_EUR"]),
        float(greedy["discharge_revenue_EUR"]),
        float(greedy["degradation_cost_EUR"]),
    ]
    opt_vals = [
        float(opt_id["charging_cost_EUR"]),
        float(opt_id["discharge_revenue_EUR"]),
        float(opt_id["degradation_cost_EUR"]),
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars_g = ax2.bar(x - width/2, greedy_vals, width, label="Greedy_ID", 
                     color=COLORS["greedy"], alpha=0.85, edgecolor="black", linewidth=1.5)
    bars_o = ax2.bar(x + width/2, opt_vals, width, label="ID (Optimized)", 
                     color=COLORS["optimized"], alpha=0.85, edgecolor="black", linewidth=1.5)
    
    ax2.set_ylabel("Amount (€)", fontsize=11, fontweight="bold")
    ax2.set_title("Panel B: Cost & Revenue Decomposition", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"€{x/1e6:.1f}M" if abs(x) >= 1e6 else f"€{x/1e3:.0f}k"))
    ax2.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    
    # ==== Panel 3: Market Throughput ====
    ax3 = fig.add_subplot(gs[1, 0])
    
    throughput_metrics = ["Total\nCharged (MWh)", "Total\nDischarged (MWh)"]
    greedy_tp = [
        float(greedy["total_charged_MWh"]),
        float(greedy["total_discharged_MWh"]),
    ]
    opt_tp = [
        float(opt_id["total_charged_MWh"]),
        float(opt_id["total_discharged_MWh"]),
    ]
    
    x_tp = np.arange(len(throughput_metrics))
    bars_g_tp = ax3.bar(x_tp - width/2, greedy_tp, width, label="Greedy_ID", 
                        color=COLORS["greedy"], alpha=0.85, edgecolor="black", linewidth=1.5)
    bars_o_tp = ax3.bar(x_tp + width/2, opt_tp, width, label="ID (Optimized)", 
                        color=COLORS["optimized"], alpha=0.85, edgecolor="black", linewidth=1.5)
    
    ax3.set_ylabel("Energy Throughput (MWh/year)", fontsize=11, fontweight="bold")
    ax3.set_title("Panel C: Market Participation & Throughput", fontsize=12, fontweight="bold", pad=12)
    ax3.set_xticks(x_tp)
    ax3.set_xticklabels(throughput_metrics, fontsize=10)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x/1000:.1f}k"))
    ax3.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax3.grid(axis="y", alpha=0.3, linestyle="--")
    
    # ==== Panel 4: Detailed Failure Analysis ====
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Show key failure metrics
    failure_categories = ["Unmet Energy\n(MWh)", "Excess\nThroughput (×)", "Degradation\nCost (€B)", "Profit\nCapture (%)"]
    unmet_mwh = float(greedy.get("unmet_MWh", 0))
    excess_throughput = (float(greedy["total_charged_MWh"]) + float(greedy["total_discharged_MWh"])) / \
                        (float(opt_id["total_charged_MWh"]) + float(opt_id["total_discharged_MWh"]))
    deg_cost_b = float(greedy["degradation_cost_EUR"]) / 1e9
    profit_capture = (float(greedy["profit_EUR"]) / float(opt_id["profit_EUR"]) * 100) if opt_id["profit_EUR"] != 0 else 0
    
    # Normalize for visualization (0-100% scale)
    failure_vals = [
        min(unmet_mwh / 174, 100),  # Normalize unmet to 100% if reaches 17,400 MWh
        min(excess_throughput * 30, 100),  # Normalize excess throughput
        min(deg_cost_b * 50, 100),  # Normalize degradation cost
        max(profit_capture, 0)  # Profit capture as-is
    ]
    
    colors_fail = [COLORS["loss"], COLORS["loss"], COLORS["loss"], COLORS["greedy"]]
    bars_fail = ax4.bar(failure_categories, failure_vals, color=colors_fail, alpha=0.8, edgecolor="black", linewidth=1.5)
    
    ax4.axhline(y=100, color="red", linestyle="--", linewidth=2.5, alpha=0.7, label="Critical Level (100%)")
    ax4.set_ylabel("Failure Severity (Normalized 0-100%)", fontsize=12, fontweight="bold")
    ax4.set_title("Panel D: Why Greedy Fails - Detailed Analysis", fontsize=13, fontweight="bold", pad=12)
    ax4.set_ylim(0, 120)
    ax4.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax4.grid(axis="y", alpha=0.3, linestyle="--")
    
    # Add detailed value labels
    labels_detail = [
        f"{unmet_mwh:,.0f} MWh\n(17% of driving)",
        f"{excess_throughput:.1f}x\noptimization",
        f"€{deg_cost_b:.2f}B\n(€20/MWh)",
        f"{profit_capture:.0f}%\nof optimization"
    ]
    
    for bar, val, label in zip(bars_fail, failure_vals, labels_detail):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + 3, label,
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    # Main title
    fig.suptitle("Greedy Intraday Heuristic vs. Optimization: Empirical Validation",
                fontsize=14, fontweight="bold", y=0.995)
    
    # Save
    out_png = output_folder / "01_greedy_vs_optimization_comparison.png"
    out_svg = output_folder / "01_greedy_vs_optimization_comparison.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    print(f"  ✓ Saved: {out_png.name}")
    print(f"  ✓ Saved: {out_svg.name}")


def create_all_scenarios_comparison(comparison_df: pd.DataFrame, output_folder: Path) -> None:
    """
    All scenarios side-by-side profit ranking with greedy highlighted.
    Shows how greedy fits into the full strategy landscape.
    """
    df_deg = comparison_df[comparison_df["degradation"] == True].copy()
    df_deg = df_deg.sort_values("profit_EUR", ascending=False)
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Color mapping
    scenario_colors = {
        "Combined": COLORS["combined"],
        "ID": COLORS["optimized"],
        "Greedy_ID": COLORS["greedy"],
        "FCR": COLORS["fcr"],
        "DA": COLORS["optimized"],
        "Naive_EDF": COLORS["naive"],
    }
    colors = [scenario_colors.get(s, COLORS["optimized"]) for s in df_deg["scenario"]]
    
    # Lollipop chart
    y_pos = np.arange(len(df_deg))
    profits = df_deg["profit_EUR"].values / 1e6  # Convert to millions
    
    ax.hlines(y_pos, 0, profits, colors=colors, linewidth=3, alpha=0.8)
    ax.scatter(profits, y_pos, color=colors, s=350, alpha=0.9, edgecolors="black", linewidth=2, zorder=3)
    
    # Labels
    labels = df_deg["scenario"].values
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12, fontweight="bold")
    ax.set_xlabel("Annual Net Profit (€ Millions)", fontsize=13, fontweight="bold")
    ax.set_title("All Strategies: Profit Ranking with Battery Degradation\n(Greedy_ID vs. Optimization Baselines)",
                fontsize=14, fontweight="bold", pad=15)
    
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1.5, alpha=0.7)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    
    # Add value labels positioned outside bars
    for i, (profit, label) in enumerate(zip(profits, labels)):
        if profit >= 0:
            ax.text(profit + 0.3, i, f"€{profit:.2f}M", va="center", fontsize=11, fontweight="bold")
        else:
            ax.text(profit - 0.3, i, f"€{profit:.2f}M", va="center", ha="right", fontsize=11, fontweight="bold")
    
    # Legend for colors - positioned outside plot area
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["combined"], edgecolor="black", linewidth=1.5, label="Combined (DA+ID+FCR)"),
        Patch(facecolor=COLORS["optimized"], edgecolor="black", linewidth=1.5, label="ID (Optimized Perfect Foresight)"),
        Patch(facecolor=COLORS["greedy"], edgecolor="black", linewidth=1.5, label="Greedy_ID (Online Lookahead)"),
        Patch(facecolor=COLORS["fcr"], edgecolor="black", linewidth=1.5, label="FCR (Reserve Only)"),
        Patch(facecolor=COLORS["naive"], edgecolor="black", linewidth=1.5, label="Naive_EDF (Baseline)"),
    ]
    ax.legend(handles=legend_elements, loc="center left", fontsize=11, framealpha=0.95, 
             bbox_to_anchor=(1.02, 0.5), borderpad=1)
    
    ax.set_xlim(min(profits) - 2, max(profits) + 3)
    
    out_png = output_folder / "02_all_strategies_ranking.png"
    out_svg = output_folder / "02_all_strategies_ranking.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    print(f"  ✓ Saved: {out_png.name}")
    print(f"  ✓ Saved: {out_svg.name}")


def create_degradation_impact_analysis(comparison_df: pd.DataFrame, output_folder: Path) -> None:
    """
    Show degradation cost impact across all scenarios, including greedy vs optimization gap.
    """
    fig, ax = plt.subplots(figsize=(13, 7))
    
    scenarios = sorted(comparison_df["scenario"].unique())
    
    # Get profit with and without degradation
    profits_no_deg = []
    profits_deg = []
    scenario_labels = []
    
    for scenario in scenarios:
        data_no_deg = comparison_df[(comparison_df["scenario"] == scenario) & (comparison_df["degradation"] == False)]
        data_deg = comparison_df[(comparison_df["scenario"] == scenario) & (comparison_df["degradation"] == True)]
        
        if len(data_no_deg) > 0 and len(data_deg) > 0:
            profits_no_deg.append(float(data_no_deg.iloc[0]["profit_EUR"]) / 1e6)
            profits_deg.append(float(data_deg.iloc[0]["profit_EUR"]) / 1e6)
            scenario_labels.append(scenario)
    
    x = np.arange(len(scenario_labels))
    width = 0.35
    
    # New color scheme for better distinction
    color_no_deg = "#E8F4F8"      # Light gray-blue
    color_deg = "#2E86AB"         # Dark blue
    
    bars1 = ax.bar(x - width/2, profits_no_deg, width, label="Without Degradation", 
                   color=color_no_deg, alpha=0.7, edgecolor="black", linewidth=1.2)
    bars2 = ax.bar(x + width/2, profits_deg, width, label="With Degradation (€20/MWh)", 
                   color=color_deg, alpha=0.85, edgecolor="black", linewidth=1.2)
    
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_ylabel("Annual Net Profit (€ Millions)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Market Strategy", fontsize=12, fontweight="bold")
    ax.set_title("Battery Degradation Cost Impact on Profitability\n(Including Greedy_ID Heuristic)",
                fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=11, rotation=0, ha="center")
    ax.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    out_png = output_folder / "03_degradation_impact_analysis.png"
    out_svg = output_folder / "03_degradation_impact_analysis.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    print(f"  ✓ Saved: {out_png.name}")
    print(f"  ✓ Saved: {out_svg.name}")


def create_fcr_penalty_vs_baseline_comparison(comparison_df: pd.DataFrame, output_folder: Path) -> None:
    """Build a waterfall for FCR baseline gross revenue vs realistic penalty deductions."""
    df = comparison_df[
        (comparison_df["degradation"] == True)
        & (comparison_df["scenario"].isin(["FCR", "Combined"]))
        & (comparison_df["fcr_policy"].isin(["Baseline_NoPenalty", "Penalty_Realistic"]))
    ].copy()

    if df.empty:
        print("  Warning: No policy comparison data found, skipping FCR policy graph")
        return

    fcr_baseline = df[
        (df["scenario"] == "FCR")
        & (df["fcr_policy"] == "Baseline_NoPenalty")
    ]
    fcr_penalty = df[
        (df["scenario"] == "FCR")
        & (df["fcr_policy"] == "Penalty_Realistic")
    ]

    if len(fcr_baseline) == 0 or len(fcr_penalty) == 0:
        print("  Warning: FCR policy rows missing, skipping FCR policy graph")
        return

    b = fcr_baseline.iloc[0]
    p = fcr_penalty.iloc[0]

    baseline_gross = float(b["fcr_revenue_gross_EUR"])
    acceptance_rate = float(FCR_POLICY_PENALTY["acceptance_rate"])
    fee_share = float(FCR_POLICY_PENALTY["aggregator_fee_share"])

    acceptance_loss = baseline_gross * (1.0 - acceptance_rate)
    post_acceptance = baseline_gross - acceptance_loss
    aggregator_fee_loss = post_acceptance * fee_share
    activation_loss = float(p["fcr_activation_cost_EUR"])
    ramp_loss = float(p["fcr_ramp_penalty_EUR"])
    realistic_net = post_acceptance - aggregator_fee_loss - activation_loss - ramp_loss
    realized_net = float(p["fcr_revenue_EUR"])

    labels = [
        "Baseline\nGross",
        "Acceptance\nHaircut",
        "Aggregator\nFee",
        "Activation\nCost",
        "Ramp\nPenalty",
        "Realistic\nNet",
    ]
    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(13.5, 7.4))
    fig.patch.set_facecolor("#F2F2F2")
    ax.set_facecolor("#F2F2F2")

    color_baseline = "#1457C2"
    color_result = "#4D88CF"
    color_constraint = "#FF2A2A"

    ax.bar(x[0], baseline_gross, width=width, color=color_baseline, edgecolor="black", linewidth=1.5)

    losses = [acceptance_loss, aggregator_fee_loss, activation_loss, ramp_loss]
    running = baseline_gross
    max_abs = max(abs(baseline_gross), abs(realistic_net), *(abs(v) for v in losses), 1.0)
    for i, loss in enumerate(losses, start=1):
        ax.bar(
            x[i],
            -loss,
            width=width,
            bottom=running,
            color=color_constraint,
            edgecolor="black",
            linewidth=1.5,
            alpha=0.95,
        )
        ax.plot([x[i - 1] + width / 2, x[i] - width / 2], [running, running], linestyle="--", color="#666666", linewidth=1.1)
        ax.text(
            x[i],
            running + 0.018 * max_abs,
            f"-EUR {loss/1e6:.2f}M",
            ha="center",
            va="bottom",
            fontsize=9.8,
            fontweight="bold",
            color=color_constraint,
        )
        running -= loss

    ax.bar(x[-1], realistic_net, width=width, color=color_result, edgecolor="black", linewidth=1.5)

    ax.text(
        x[0],
        baseline_gross + 0.018 * max_abs,
        f"EUR {baseline_gross/1e6:.2f}M",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=color_baseline,
    )
    ax.text(
        x[-1],
        realistic_net + 0.018 * max_abs,
        f"EUR {realistic_net/1e6:.2f}M",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=color_baseline,
    )

    net_reduction_pct = (1.0 - (realistic_net / baseline_gross)) * 100.0 if baseline_gross > 0 else 0.0
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11.5, fontweight="bold")
    ax.set_ylabel("Annual Revenue (EUR millions)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Baseline vs Realistic FCR Revenue: Constraint Waterfall (Actual Data)\n"
        f"Baseline gross: EUR {baseline_gross/1e6:.2f}M | Reduction: {net_reduction_pct:.0f}%",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.grid(axis="y", alpha=0.35, linestyle=":")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"EUR {v/1e6:.1f}M"))

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color_baseline, edgecolor="black", label="Baseline / Result"),
        Patch(facecolor=color_constraint, edgecolor="black", label="Market Constraint Loss"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10.5, framealpha=0.95)

    ax.text(
        0.01,
        0.02,
        f"Realized LP net in penalty run: EUR {realized_net/1e6:.2f}M",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#1F1F1F",
    )

    y_top = max(baseline_gross, realistic_net) * 1.16
    ax.set_ylim(0.0, y_top)

    out_png = output_folder / "04_fcr_penalty_vs_baseline_optimization.png"
    out_svg = output_folder / "04_fcr_penalty_vs_baseline_optimization.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  ✓ Saved: {out_png.name}")
    print(f"  ✓ Saved: {out_svg.name}")


def create_fcr_penalty_advanced_graphs(comparison_df: pd.DataFrame, output_folder: Path) -> None:
    """Create advanced diagnostics for realistic FCR penalty mechanics."""
    df = comparison_df[
        (comparison_df["degradation"] == True)
        & (comparison_df["scenario"].isin(["FCR", "Combined"]))
        & (comparison_df["fcr_policy"].isin(["Baseline_NoPenalty", "Penalty_Realistic"]))
    ].copy()

    if df.empty:
        print("  Warning: No policy comparison data found, skipping advanced FCR diagnostics")
        return

    scenarios = ["FCR", "Combined"]
    x = np.arange(len(scenarios))
    width = 0.36

    baseline = df[df["fcr_policy"] == "Baseline_NoPenalty"].set_index("scenario").reindex(scenarios)
    penalty = df[df["fcr_policy"] == "Penalty_Realistic"].set_index("scenario").reindex(scenarios)

    base_gross = baseline["fcr_revenue_gross_EUR"].fillna(0.0).to_numpy(dtype=float)
    pen_gross = penalty["fcr_revenue_gross_EUR"].fillna(0.0).to_numpy(dtype=float)
    pen_net = penalty["fcr_revenue_EUR"].fillna(0.0).to_numpy(dtype=float)

    acceptance_loss = base_gross * (1.0 - float(FCR_POLICY_PENALTY["acceptance_rate"]))
    fee_loss = base_gross * float(FCR_POLICY_PENALTY["acceptance_rate"]) * float(FCR_POLICY_PENALTY["aggregator_fee_share"])
    activation_loss = penalty["fcr_activation_cost_EUR"].fillna(0.0).to_numpy(dtype=float)
    ramp_loss = penalty["fcr_ramp_penalty_EUR"].fillna(0.0).to_numpy(dtype=float)
    strategic_volume_shift = np.maximum(base_gross - pen_gross, 0.0)
    total_loss = np.maximum(base_gross - pen_net, 0.0)
    loss_share_pct = np.where(base_gross > 0, 100.0 * total_loss / base_gross, 0.0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    bars_base = ax.bar(x - width / 2, base_gross, width=width, color="#1B5CC8", edgecolor="black", linewidth=1.2, label="Baseline gross")
    bars_net = ax.bar(x + width / 2, pen_net, width=width, color="#0A3D91", edgecolor="black", linewidth=1.2, label="Realistic net")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11, fontweight="bold")
    ax.set_ylabel("Revenue (EUR)", fontsize=11, fontweight="bold")
    ax.set_title("Panel A: FCR Revenue Compression", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"EUR {v/1e6:.2f}M"))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for bbar, nbar in zip(bars_base, bars_net):
        ax.text(bbar.get_x() + bbar.get_width() / 2, bbar.get_height() + 0.015 * max(base_gross.max(), 1.0), f"{bbar.get_height()/1e6:.2f}M", ha="center", va="bottom", fontsize=8.5)
        ax.text(nbar.get_x() + nbar.get_width() / 2, nbar.get_height() + 0.015 * max(base_gross.max(), 1.0), f"{nbar.get_height()/1e6:.2f}M", ha="center", va="bottom", fontsize=8.5)
    ax.legend(loc="upper right", fontsize=8.8, framealpha=0.95)

    ax = axes[0, 1]
    bottoms = np.zeros_like(base_gross)
    for vals, label, color in [
        (acceptance_loss, "Acceptance haircut", "#FF2A2A"),
        (fee_loss, "Aggregator/BSP fee", "#FF4D4D"),
        (activation_loss, "Activation cost", "#FF7373"),
        (ramp_loss, "Ramp delay penalty", "#FF9B9B"),
        (strategic_volume_shift, "Strategic reserve cut", "#FFC2C2"),
    ]:
        ax.bar(x, vals, width=0.55, bottom=bottoms, label=label, color=color, edgecolor="black", linewidth=1.0)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11, fontweight="bold")
    ax.set_ylabel("Loss vs baseline gross (EUR)", fontsize=11, fontweight="bold")
    ax.set_title("Panel B: Constraint Loss Decomposition", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"EUR {v/1e6:.2f}M"))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.4, framealpha=0.95)

    ax = axes[1, 0]
    base_res = baseline["average_FCR_reserved_MW"].fillna(0.0).to_numpy(dtype=float)
    pen_res = penalty["average_FCR_reserved_MW"].fillna(0.0).to_numpy(dtype=float)
    ax.bar(x - width / 2, base_res, width=width, color="#8FBDF3", edgecolor="black", linewidth=1.2, label="Baseline")
    ax.bar(x + width / 2, pen_res, width=width, color="#2E6FBE", edgecolor="black", linewidth=1.2, label="Realistic")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11, fontweight="bold")
    ax.set_ylabel("Average reserved FCR (MW)", fontsize=11, fontweight="bold")
    ax.set_title("Panel C: Capacity Reservation Response", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=8.8, framealpha=0.95)

    ax = axes[1, 1]
    bars = ax.bar(x, loss_share_pct, width=0.55, color="#FF2A2A", edgecolor="black", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11, fontweight="bold")
    ax.set_ylabel("Loss from baseline gross (%)", fontsize=11, fontweight="bold")
    ax.set_title("Panel D: Relative Penalty Burden", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for bar, val in zip(bars, loss_share_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8, f"{val:.1f}%", ha="center", va="bottom", fontsize=9.0, fontweight="bold")

    fig.suptitle(
        "Advanced FCR Penalty Diagnostics (Degradation Enabled)",
        fontsize=13,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.90, right=0.82, hspace=0.32, wspace=0.25)

    out_png = output_folder / "08_fcr_penalty_advanced_diagnostics.png"
    out_svg = output_folder / "08_fcr_penalty_advanced_diagnostics.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  ✓ Saved: {out_png.name}")
    print(f"  ✓ Saved: {out_svg.name}")


def build_realistic_fcr_vehicle_class_comparison(
    idx: pd.DatetimeIndex,
    da: np.ndarray,
    idp: np.ndarray,
    fcr_price: np.ndarray,
    block_id: np.ndarray,
) -> pd.DataFrame:
    """Run realistic-penalty FCR and Combined cases for key vehicle classes."""
    combined_buy = np.minimum(da, idp)
    combined_sell = np.maximum(da, idp)
    rows = []

    for scope in CLASS_SCOPE_DEFINITIONS:
        class_data = load_mobility(idx, vehicle_classes=scope["classes"])
        grid_limit = GRID_LIMIT_FRACTION * float(
            max(class_data["pch_max"].max(), class_data["pdis_max"].max())
        )

        for label, buy, sell, allow_discharge, fcr_on in (
            ("FCR", da, da, False, True),
            ("Combined", combined_buy, combined_sell, True, True),
        ):
            result = run_opt(
                data=class_data,
                buy=buy,
                sell=sell,
                allow_discharge=allow_discharge,
                fcr_on=fcr_on,
                fcr_price=fcr_price,
                block_id=block_id,
                deg_on=True,
                grid_limit=grid_limit,
                label=label,
                fcr_policy=FCR_POLICY_PENALTY,
            )

            total_charged = float(np.asarray(result.pop("_pch"), dtype=float).sum() * DT_H)
            total_discharged = float(np.asarray(result.pop("_pdis"), dtype=float).sum() * DT_H)

            rows.append({
                "class_label": scope["label"],
                "class_slug": scope["slug"],
                "source_classes": class_data["source_classes"],
                "fleet_size": int(class_data["fleet_size"]),
                "battery_capacity_MWh_total": float(class_data["cap"]),
                "grid_limit_MW": grid_limit,
                "scenario": label,
                "fcr_policy": FCR_POLICY_PENALTY["name"],
                "degradation": True,
                "profit_EUR": float(result["profit_EUR"]),
                "fcr_revenue_EUR": float(result["fcr_revenue_EUR"]),
                "average_FCR_reserved_MW": float(result["average_FCR_reserved_MW"]),
                "max_FCR_reserved_MW": float(result["max_FCR_reserved_MW"]),
                "unmet_MWh": float(result["unmet_MWh"]),
                "total_charged_MWh": total_charged,
                "total_discharged_MWh": total_discharged,
                "profit_per_vehicle_EUR": float(result["profit_EUR"]) / max(int(class_data["fleet_size"]), 1),
                "profit_per_battery_MWh_EUR": float(result["profit_EUR"]) / max(float(class_data["cap"]), 1e-9),
            })

    return pd.DataFrame(rows)


def create_realistic_fcr_vehicle_class_comparison(class_df: pd.DataFrame, output_folder: Path) -> None:
    """Compare key vehicle classes under the realistic FCR penalty case."""
    if class_df.empty:
        print("  Warning: No class-level realistic FCR data found, skipping class comparison")
        return

    class_order = [scope["label"] for scope in CLASS_SCOPE_DEFINITIONS]
    scenario_order = ["FCR", "Combined"]
    colors = {"FCR": "#6C5CE7", "Combined": "#0052CC"}

    df = class_df.copy()
    df["class_label"] = pd.Categorical(df["class_label"], categories=class_order, ordered=True)
    df["scenario"] = pd.Categorical(df["scenario"], categories=scenario_order, ordered=True)
    df = df.sort_values(["class_label", "scenario"])

    x = np.arange(len(class_order))
    width = 0.34

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.3))
    panel_specs = [
        ("profit_EUR", "Annual Profit (EUR)", "Panel A: Net Profit", lambda v, _: f"EUR {v/1e6:.2f}M"),
        ("fcr_revenue_EUR", "Net FCR Revenue (EUR)", "Panel B: Net FCR Revenue", lambda v, _: f"EUR {v/1e6:.2f}M"),
        ("average_FCR_reserved_MW", "Average Reserved FCR (MW)", "Panel C: Reserved Capacity", lambda v, _: f"{v:.0f}"),
        ("profit_per_battery_MWh_EUR", "Profit per Battery Capacity (EUR/MWh)", "Panel D: Capital Productivity", lambda v, _: f"EUR {v/1e3:.0f}k"),
    ]

    for ax, (column, ylabel, title, formatter) in zip(axes.ravel(), panel_specs):
        for offset, scenario in zip((-width / 2, width / 2), scenario_order):
            subset = df[df["scenario"] == scenario].set_index("class_label").reindex(class_order)
            values = subset[column].fillna(0.0).to_numpy(dtype=float)
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=scenario,
                color=colors[scenario],
                edgecolor="black",
                linewidth=1.1,
                alpha=0.9,
            )
            if column in {"profit_EUR", "fcr_revenue_EUR"}:
                for bar, val in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + 0.012 * max(np.max(np.abs(values)), 1.0),
                        f"{val/1e6:.2f}M",
                        ha="center",
                        va="bottom",
                        fontsize=7.8,
                    )

        labels = []
        for class_name in class_order:
            fleet_size = int(df[df["class_label"] == class_name]["fleet_size"].iloc[0])
            labels.append(f"{class_name}\n(n={fleet_size:,})")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.8)
        ax.set_ylabel(ylabel, fontsize=10.5, fontweight="bold")
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(formatter))

    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", ncol=2, framealpha=0.95, fontsize=9.5)
    fig.suptitle(
        "Vehicle-Class Performance Under Realistic FCR Penalties (Degradation Enabled)",
        fontsize=13,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.88, hspace=0.34, wspace=0.26)

    out_png = output_folder / "05_vehicle_class_realistic_fcr_comparison.png"
    out_svg = output_folder / "05_vehicle_class_realistic_fcr_comparison.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  ✓ Saved: {out_png.name}")
    print(f"  ✓ Saved: {out_svg.name}")


def create_vehicle_class_fcr_combined_ratio_graph(class_df: pd.DataFrame, output_folder: Path) -> None:
    """Show reserve-capacity and performance ratios for Combined vs FCR-only by class."""
    if class_df.empty:
        print("  Warning: No class-level data found, skipping ratio graph")
        return

    class_order = [scope["label"] for scope in CLASS_SCOPE_DEFINITIONS]
    df = class_df.copy()
    df = df[df["scenario"].isin(["FCR", "Combined"])].copy()

    fcr = df[df["scenario"] == "FCR"].set_index("class_label").reindex(class_order)
    comb = df[df["scenario"] == "Combined"].set_index("class_label").reindex(class_order)

    fcr_reserve = fcr["average_FCR_reserved_MW"].to_numpy(dtype=float)
    comb_reserve = comb["average_FCR_reserved_MW"].to_numpy(dtype=float)
    fcr_profit_abs = np.abs(fcr["profit_EUR"].to_numpy(dtype=float))
    comb_profit_abs = np.abs(comb["profit_EUR"].to_numpy(dtype=float))

    reserve_ratio_pct = np.full_like(fcr_reserve, np.nan, dtype=float)
    np.divide(
        100.0 * comb_reserve,
        fcr_reserve,
        out=reserve_ratio_pct,
        where=fcr_reserve > 0,
    )

    performance_ratio_pct = np.full_like(fcr_profit_abs, np.nan, dtype=float)
    np.divide(
        100.0 * comb_profit_abs,
        fcr_profit_abs,
        out=performance_ratio_pct,
        where=fcr_profit_abs > 1e-9,
    )

    x = np.arange(len(class_order))
    width = 0.34

    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    bars_res = ax.bar(
        x - width / 2,
        reserve_ratio_pct,
        width,
        color="#1B5CC8",
        edgecolor="black",
        linewidth=1.2,
        label="Reserve capacity ratio (Combined/FCR)",
    )
    bars_perf = ax.bar(
        x + width / 2,
        performance_ratio_pct,
        width,
        color="#FF2A2A",
        edgecolor="black",
        linewidth=1.2,
        label="Performance ratio (|Combined profit|/|FCR profit|)",
    )

    ax.axhline(100.0, color="#444444", linestyle="--", linewidth=1.1, alpha=0.85)

    labels = []
    for class_name in class_order:
        fleet_size = int(df[df["class_label"] == class_name]["fleet_size"].iloc[0])
        labels.append(f"{class_name}\n(n={fleet_size:,})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Ratio (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Vehicle-Class Ratios: Combined vs FCR-only\nReserve Capacity and Profit Performance",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper center", ncol=2, fontsize=9, framealpha=0.95)

    for bars in (bars_res, bars_perf):
        for bar in bars:
            val = bar.get_height()
            if np.isnan(val):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 2.0,
                f"{val:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    out_png = output_folder / "09_vehicle_class_fcr_combined_ratio.png"
    out_svg = output_folder / "09_vehicle_class_fcr_combined_ratio.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  ✓ Saved: {out_png.name}")
    print(f"  ✓ Saved: {out_svg.name}")


# ============================================================
# 5. RUN ALL
# ============================================================

def main() -> None:
    idx = build_index()
    print(
        f"Steps: {len(idx)}  "
        f"({DT_H} h each, {len(idx) * DT_H / 24:.0f} days)"
    )

    data = load_mobility(idx)
    da = load_da(idx)
    idp = load_id(idx)
    fcr_price, block_id = load_fcr(idx)

    grid_limit = GRID_LIMIT_FRACTION * float(
        max(data["pch_max"].max(), data["pdis_max"].max())
    )

    battery_capacity = float(data["cap"])
    total_driving_energy = float(np.asarray(data["edrive"], dtype=float).sum())

    print(
        f"Capacity {battery_capacity:.0f} MWh | "
        f"SOC [{data['soc_min']:.0f}, {data['soc_max']:.0f}] | "
        f"grid limit {grid_limit:.0f} MW | "
        f"driving energy {total_driving_energy:,.1f} MWh"
    )

    rows = []

    # Naive baseline: all charging is valued as DA charging.
    naive = run_naive(data, da, grid_limit)
    naive_total_charged = float(naive.pop("throughput_MWh", 0.0))
    naive.update({
        "fcr_policy": "N/A",
        "battery_capacity_MWh_total": battery_capacity,
        "total_driving_energy_MWh": total_driving_energy,
        "total_DA_charged_MWh": naive_total_charged,
        "total_ID_charged_MWh": 0.0,
        "total_charged_MWh": naive_total_charged,
        "total_DA_discharged_MWh": 0.0,
        "total_ID_discharged_MWh": 0.0,
        "total_discharged_MWh": 0.0,
        "average_FCR_reserved_MW": 0.0,
        "max_FCR_reserved_MW": 0.0,
    })
    rows.append(naive)

    # Perfect-foresight best-of-DA/ID price signals used by the Combined benchmark.
    combined_buy = np.minimum(da, idp)
    combined_sell = np.maximum(da, idp)

    scenarios = [
        # label, buy price, sell price, allow discharge, FCR enabled
        ("DA", da, da, True, False),
        ("ID", idp, idp, True, False),
        ("FCR", da, da, False, True),
        ("Combined", combined_buy, combined_sell, True, True),
    ]

    dispatch_for_sheet = None

    # Run optimization scenarios (LP-based)
    for fcr_policy in FCR_POLICIES:
        print(f"\n  Optimization policy: {fcr_policy['name']}")
        for label, buy, sell, allow_discharge, fcr_on in scenarios:
            for deg_on in (False, True):
                t0 = time.time()

                result = run_opt(
                    data=data,
                    buy=buy,
                    sell=sell,
                    allow_discharge=allow_discharge,
                    fcr_on=fcr_on,
                    fcr_price=fcr_price,
                    block_id=block_id,
                    deg_on=deg_on,
                    grid_limit=grid_limit,
                    label=label,
                    fcr_policy=fcr_policy,
                )

                pch_v = np.asarray(result.pop("_pch"), dtype=float)
                pdis_v = np.asarray(result.pop("_pdis"), dtype=float)

                charged_energy = pch_v * DT_H
                discharged_energy = pdis_v * DT_H

                total_charged = float(charged_energy.sum())
                total_discharged = float(discharged_energy.sum())

                if label in ("DA", "FCR"):
                    da_charged = total_charged
                    id_charged = 0.0
                    da_discharged = total_discharged
                    id_discharged = 0.0

                elif label == "ID":
                    da_charged = 0.0
                    id_charged = total_charged
                    da_discharged = 0.0
                    id_discharged = total_discharged

                elif label == "Combined":
                    # Equal DA and ID prices are assigned to DA.
                    da_buy_mask = da <= idp
                    da_sell_mask = da >= idp

                    da_charged = float(charged_energy[da_buy_mask].sum())
                    id_charged = float(charged_energy[~da_buy_mask].sum())
                    da_discharged = float(discharged_energy[da_sell_mask].sum())
                    id_discharged = float(discharged_energy[~da_sell_mask].sum())

                else:
                    raise ValueError(f"Unknown scenario: {label}")

                result.update({
                    "battery_capacity_MWh_total": battery_capacity,
                    "total_driving_energy_MWh": total_driving_energy,
                    "total_DA_charged_MWh": da_charged,
                    "total_ID_charged_MWh": id_charged,
                    "total_charged_MWh": total_charged,
                    "total_DA_discharged_MWh": da_discharged,
                    "total_ID_discharged_MWh": id_discharged,
                    "total_discharged_MWh": total_discharged,
                })

                tag = "deg" if deg_on else "no-deg"
                print(
                    f"  {label:15s} {tag:7s} "
                    f"profit EUR {result['profit_EUR']:>14,.0f}  "
                    f"fcr {result['fcr_revenue_EUR']:>12,.0f}  "
                    f"charged {total_charged:>10,.1f} MWh  "
                    f"discharged {total_discharged:>10,.1f} MWh  "
                    f"unmet {result['unmet_MWh']:.1f}  "
                    f"({time.time() - t0:.1f}s)"
                )

                if label == "Combined" and not deg_on and fcr_policy["name"] == "Penalty_Realistic":
                    dispatch_for_sheet = pd.DataFrame({
                        "datetime": idx,
                        "DA_price_EUR_per_MWh": da,
                        "ID_price_EUR_per_MWh": idp,
                        "selected_buy_market": np.where(da <= idp, "DA", "ID"),
                        "selected_sell_market": np.where(da >= idp, "DA", "ID"),
                        "p_charge_MW": pch_v,
                        "p_discharge_MW": pdis_v,
                        "charged_energy_MWh": charged_energy,
                        "discharged_energy_MWh": discharged_energy,
                    })

                rows.append(result)
    
    # Run greedy intraday heuristic (online lookahead-based)
    print("\n  Greedy intraday (online lookahead algorithm):")
    for deg_on in (False, True):
        t0 = time.time()
        
        result = run_greedy_intraday(
            data=data,
            idp=idp,
            grid_limit=grid_limit,
            deg_on=deg_on,
            lookahead_hours=2.0  # 2-hour lookahead window
        )
        
        pch_v = np.asarray(result.pop("_pch"), dtype=float)
        pdis_v = np.asarray(result.pop("_pdis"), dtype=float)

        charged_energy = pch_v * DT_H
        discharged_energy = pdis_v * DT_H

        total_charged = float(charged_energy.sum())
        total_discharged = float(discharged_energy.sum())
        
        # All greedy trading is in ID market
        da_charged = 0.0
        id_charged = total_charged
        da_discharged = 0.0
        id_discharged = total_discharged

        result.update({
            "fcr_policy": "N/A",
            "battery_capacity_MWh_total": battery_capacity,
            "total_driving_energy_MWh": total_driving_energy,
            "total_DA_charged_MWh": da_charged,
            "total_ID_charged_MWh": id_charged,
            "total_charged_MWh": total_charged,
            "total_DA_discharged_MWh": da_discharged,
            "total_ID_discharged_MWh": id_discharged,
            "total_discharged_MWh": total_discharged,
        })

        tag = "deg" if deg_on else "no-deg"
        print(
            f"  {'Greedy_ID':15s} {tag:7s} "
            f"profit EUR {result['profit_EUR']:>14,.0f}  "
            f"fcr {result['fcr_revenue_EUR']:>12,.0f}  "
            f"charged {total_charged:>10,.1f} MWh  "
            f"discharged {total_discharged:>10,.1f} MWh  "
            f"unmet {result['unmet_MWh']:.1f}  "
            f"({time.time() - t0:.1f}s)"
        )
        
        rows.append(result)

    comparison_columns = [
        "scenario",
        "fcr_policy",
        "degradation",
        "battery_capacity_MWh_total",
        "total_driving_energy_MWh",
        "profit_EUR",
        "net_cost_EUR",
        "charging_cost_EUR",
        "discharge_revenue_EUR",
        "fcr_revenue_EUR",
        "fcr_revenue_gross_EUR",
        "fcr_activation_cost_EUR",
        "fcr_ramp_penalty_EUR",
        "degradation_cost_EUR",
        "total_DA_charged_MWh",
        "total_ID_charged_MWh",
        "total_charged_MWh",
        "total_DA_discharged_MWh",
        "total_ID_discharged_MWh",
        "total_discharged_MWh",
        "average_FCR_reserved_MW",
        "max_FCR_reserved_MW",
        "unmet_MWh",
    ]

    comparison = pd.DataFrame(rows)[comparison_columns]

    assumptions = pd.DataFrame([
        ("run_timestamp", RUN_STAMP),
        ("year", YEAR),
        ("resolution_h", DT_H),
        ("horizon_days", "full year" if HORIZON_DAYS is None else HORIZON_DAYS),
        ("battery_capacity_MWh_total", battery_capacity),
        ("total_driving_energy_MWh", total_driving_energy),
        ("eta_charge", ETA_CH),
        ("eta_discharge", ETA_DIS),
        ("soc_init_and_terminal_share", SOC_INIT_SHARE),
        ("fcr_reservation_h", TRES_FCR_H),
        ("fcr_required_response_time_s", FCR_REQUIRED_RESPONSE_TIME_S),
        ("fcr_assumed_response_delay_s", FCR_ASSUMED_RESPONSE_DELAY_S),
        ("fcr_effective_response_window_s", FCR_EFFECTIVE_RESPONSE_WINDOW_S),
        ("baseline_fcr_acceptance_rate", FCR_POLICY_BASELINE["acceptance_rate"]),
        ("baseline_fcr_aggregator_fee_share", FCR_POLICY_BASELINE["aggregator_fee_share"]),
        ("baseline_fcr_activation_cost_EUR_per_MW_block", FCR_POLICY_BASELINE["activation_cost_eur_per_mw_block"]),
        ("baseline_fcr_ramp_delay_penalty_EUR_per_MW_block", FCR_POLICY_BASELINE["ramp_delay_penalty_eur_per_mw_block"]),
        ("penalty_fcr_acceptance_rate", FCR_POLICY_PENALTY["acceptance_rate"]),
        ("penalty_fcr_aggregator_fee_share", FCR_POLICY_PENALTY["aggregator_fee_share"]),
        ("penalty_fcr_activation_cost_EUR_per_MW_block", FCR_POLICY_PENALTY["activation_cost_eur_per_mw_block"]),
        ("penalty_fcr_ramp_delay_penalty_EUR_per_MW_block", FCR_POLICY_PENALTY["ramp_delay_penalty_eur_per_mw_block"]),
        ("degradation_cost_EUR_per_MWh", C_DEG_EUR_PER_MWH),
        ("grid_limit_MW", round(grid_limit, 1)),
        ("combined_buy_rule", "minimum of realized DA and ID price"),
        ("combined_sell_rule", "maximum of realized DA and ID price"),
        ("market_tie_rule", "DA selected when DA and ID prices are equal"),
        ("da_source", "ENTSO-E Sequence 1 hourly"),
        ("id_source", "IDC index"),
        ("fcr_source", "German NEGPOS symmetric capacity price"),
    ], columns=["Parameter", "Value"])

    policy_delta_rows = []
    for scenario_name in ["FCR", "Combined"]:
        for deg_on in (False, True):
            baseline_row = comparison[
                (comparison["scenario"] == scenario_name)
                & (comparison["degradation"] == deg_on)
                & (comparison["fcr_policy"] == "Baseline_NoPenalty")
            ]
            penalty_row = comparison[
                (comparison["scenario"] == scenario_name)
                & (comparison["degradation"] == deg_on)
                & (comparison["fcr_policy"] == "Penalty_Realistic")
            ]

            if len(baseline_row) == 0 or len(penalty_row) == 0:
                continue

            b = baseline_row.iloc[0]
            p = penalty_row.iloc[0]
            policy_delta_rows.append({
                "scenario": scenario_name,
                "degradation": deg_on,
                "baseline_profit_EUR": float(b["profit_EUR"]),
                "penalty_profit_EUR": float(p["profit_EUR"]),
                "profit_delta_penalty_minus_baseline_EUR": float(p["profit_EUR"] - b["profit_EUR"]),
                "baseline_fcr_revenue_EUR": float(b["fcr_revenue_EUR"]),
                "penalty_fcr_revenue_EUR": float(p["fcr_revenue_EUR"]),
                "fcr_revenue_delta_penalty_minus_baseline_EUR": float(p["fcr_revenue_EUR"] - b["fcr_revenue_EUR"]),
                "baseline_average_FCR_reserved_MW": float(b["average_FCR_reserved_MW"]),
                "penalty_average_FCR_reserved_MW": float(p["average_FCR_reserved_MW"]),
            })

    policy_delta_df = pd.DataFrame(policy_delta_rows)

    class_policy_comparison = build_realistic_fcr_vehicle_class_comparison(
        idx=idx,
        da=da,
        idp=idp,
        fcr_price=fcr_price,
        block_id=block_id,
    )

    out = OUTPUT_FOLDER / f"fcr_policy_comparison_{RUN_STAMP}.xlsx"

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        comparison.to_excel(writer, sheet_name="Comparison", index=False)
        assumptions.to_excel(writer, sheet_name="Assumptions", index=False)

        if dispatch_for_sheet is not None:
            dispatch_for_sheet.to_excel(
                writer,
                sheet_name="Combined_Dispatch_nodeg",
                index=False,
            )
        if not policy_delta_df.empty:
            policy_delta_df.to_excel(writer, sheet_name="FCR_Policy_Delta", index=False)
        if not class_policy_comparison.empty:
            class_policy_comparison.to_excel(writer, sheet_name="Class_Realistic_FCR", index=False)

    print("\nComparison written to:", out)
    print(comparison.to_string(index=False))
    
    # Generate publication-quality graphs
    print("\n" + "=" * 80)
    print("GENERATING ACADEMIC PUBLICATION-QUALITY GRAPHS")
    print("=" * 80)
    
    try:
        create_greedy_vs_optimization_comparison(comparison, OUTPUT_FOLDER)
        create_all_scenarios_comparison(comparison, OUTPUT_FOLDER)
        create_degradation_impact_analysis(comparison, OUTPUT_FOLDER)
        create_fcr_penalty_vs_baseline_comparison(comparison, OUTPUT_FOLDER)
        create_fcr_penalty_advanced_graphs(comparison, OUTPUT_FOLDER)
        create_realistic_fcr_vehicle_class_comparison(class_policy_comparison, OUTPUT_FOLDER)
        create_vehicle_class_fcr_combined_ratio_graph(class_policy_comparison, OUTPUT_FOLDER)
        print("\n✓ All graphs generated successfully!")
    except Exception as e:
        print(f"\n✗ Error generating graphs: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

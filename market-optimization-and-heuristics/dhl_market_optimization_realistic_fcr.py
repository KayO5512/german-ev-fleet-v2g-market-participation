# -*- coding: utf-8 -*-
"""
DHL EV fleet VPP: market optimization baseline and naive EDF baseline.

Two models, built to be compared:

1. Naive EDF baseline. Price-blind, unidirectional, no markets. Charges greedily
   to full whenever the fleet is plugged in, never discharges, never trades. This
   is the weak lower bound your optimized model is meant to beat.

2. Optimization baseline. An aggregate single-battery LP run four ways, matching
   your four-bar chart:
     DA       : energy arbitrage in the day-ahead market only
     ID       : energy arbitrage in the intraday market only
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

This revised version adds a light-touch German-market realism layer for FCR:
auction acceptance haircut, aggregator fee, activation/SOC-drift cost, grid
capacity reservation for FCR, and a linear ramp/time-delay penalty. The model
remains an LP, so it avoids binary bid-clearing or strict 1 MW bid granularity.
"""

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo


# ============================================================
# 1. SETTINGS
# ============================================================

# All inputs and outputs live in this one folder.
BASE = Path(r"C:\Users\Yaw\Documents\EV_BID_Final")

MOBILITY_WORKBOOK = BASE / "dhl_fleet_model_output_2026-06-29_00-58-34.xlsx"
DA_FILE  = BASE / "Data Ahead_ENERGY_PRICES_2025.xlsx"
ID_FILE  = BASE / "ID-Prices_label_2025.xlsx"
FCR_FILE = BASE / "FCR_yearly data_RESULT_OVERVIEW_CAPACITY_MARKET_2025-01-01_2025-12-31.xlsx"
OUTPUT_FOLDER = BASE
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

RUN_STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Time.
YEAR = 2025
DT_H = 0.25                 # 15-minute steps
HORIZON_DAYS = None         # None = full year. Set to e.g. 14 for a quick test run.

# Physical.
ETA_CH = 0.95              # charging efficiency
ETA_DIS = 0.95             # discharging efficiency
SOC_INIT_SHARE = 0.60      # initial and terminal SOC floor, fraction of capacity

# FCR.
TRES_FCR_H = 0.25          # symmetric energy reservation duration (German LER, 15 min)

# Light-touch German-market realism parameters for FCR.
# These keep the model linear and transparent. Tune them in sensitivity analysis.
FCR_ACCEPTANCE_RATE = 0.80              # expected share of offered FCR capacity accepted in auction
FCR_AGGREGATOR_FEE_SHARE = 0.15         # share of gross FCR revenue retained by aggregator/BSP
FCR_ACTIVATION_COST_EUR_PER_MW_BLOCK = 2.0  # expected activation/SOC-management cost per MW per 4h block
RAMP_DELAY_PENALTY_EUR_PER_MW_BLOCK = 5.0   # penalty for response lag/communication/EVSE delay risk
FCR_GRID_RESERVE_ON = True              # reserve grid import/export capacity for FCR as well as arbitrage
FCR_MIN_REPORTING_MW = 1.0              # reporting threshold only; strict bid granularity needs MILP binaries

# Battery degradation (toggle handled per run).
C_DEG_EUR_PER_MWH = 20.0   # linear throughput cost, from pack cost over cycle life

# Grid connection limit, fraction of peak available power (depot transformer proxy).
GRID_LIMIT_FRACTION = 0.80

# Penalty for unmet driving energy, keeps the LP feasible and flags shortfalls.
UNMET_PENALTY = 1.0e6

SOLVER = "appsi_highs"


# ============================================================
# 2. LOAD AND ALIGN DATA TO A 15-MIN INDEX
# ============================================================

def build_index() -> pd.DatetimeIndex:
    end = f"{YEAR + 1}-01-01"
    if HORIZON_DAYS is not None:
        end = (pd.Timestamp(f"{YEAR}-01-01") + pd.Timedelta(days=HORIZON_DAYS)).strftime("%Y-%m-%d")
    return pd.date_range(f"{YEAR}-01-01 00:00", end, freq="15min", inclusive="left")


def load_mobility(idx: pd.DatetimeIndex) -> dict:
    """Read the hourly aggregate sheet and upsample to 15-min."""
    m = pd.read_excel(MOBILITY_WORKBOOK, sheet_name="Hourly_Aggregate")
    m["datetime"] = pd.to_datetime(m["datetime"])
    m = m.set_index("datetime").sort_index()

    # Reindex to 15-min: power and availability held constant within the hour,
    # driving energy split evenly across the four quarter-hours.
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
        "cap": cap, "soc_min": soc_min, "soc_max": soc_max, "soc_init": soc_init,
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
# 3. NAIVE EDF BASELINE (price-blind, unidirectional, greedy)
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
            "fcr_revenue_EUR": 0.0, "fcr_gross_revenue_EUR": 0.0,
            "fcr_acceptance_haircut_EUR": 0.0, "fcr_aggregator_fee_EUR": 0.0,
            "fcr_activation_cost_EUR": 0.0, "ramp_delay_penalty_EUR": 0.0,
            "degradation_cost_EUR": 0.0, "fcr_capacity_MW_blocks": 0.0,
            "fcr_avg_MW_when_positive": 0.0, "fcr_blocks_below_1MW": 0,
            "throughput_MWh": throughput, "unmet_MWh": unmet}


# ============================================================
# 4. OPTIMIZATION BUILDER
# ============================================================

def run_opt(data: dict, buy: np.ndarray, sell: np.ndarray, allow_discharge: bool,
            fcr_on: bool, fcr_price: np.ndarray, block_id: np.ndarray,
            deg_on: bool, grid_limit: float, label: str) -> dict:
    T = len(data["edrive"])
    pch_max, pdis_max, edrive = data["pch_max"], data["pdis_max"], data["edrive"]
    soc_min, soc_max, soc_init = data["soc_min"], data["soc_max"], data["soc_init"]

    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, T - 1)
    m.Tsoc = pyo.RangeSet(0, T)

    m.pch = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.pdis = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.unmet = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.soc = pyo.Var(m.Tsoc, domain=pyo.NonNegativeReals)

    # FCR blocks.
    if fcr_on:
        blocks = list(dict.fromkeys(block_id))
        m.B = pyo.Set(initialize=blocks)
        m.rfcr = pyo.Var(m.B, domain=pyo.NonNegativeReals)
        # Precompute per-block capacity ceilings and per-block price.
        block_pch_min, block_pdis_min, block_price = {}, {}, {}
        for b in blocks:
            mask = block_id == b
            block_pch_min[b] = float(pch_max[mask].min())
            block_pdis_min[b] = float(pdis_max[mask].min())
            block_price[b] = float(fcr_price[mask][0])

    # Charger and grid limits.
    def ch_limit(m, t): return m.pch[t] <= float(pch_max[t])
    m.ch_limit = pyo.Constraint(m.T, rule=ch_limit)

    def dis_limit(m, t):
        return m.pdis[t] <= (float(pdis_max[t]) if allow_discharge else 0.0)
    m.dis_limit = pyo.Constraint(m.T, rule=dis_limit)

    def grid(m, t): return m.pch[t] + m.pdis[t] <= grid_limit
    m.grid = pyo.Constraint(m.T, rule=grid)

    # SOC bounds.
    def soc_lo(m, t): return m.soc[t] >= soc_min
    def soc_hi(m, t): return m.soc[t] <= soc_max
    m.soc_lo = pyo.Constraint(m.Tsoc, rule=soc_lo)
    m.soc_hi = pyo.Constraint(m.Tsoc, rule=soc_hi)

    # Initial and terminal SOC (terminal floor kills the year-end liquidation).
    m.soc_init = pyo.Constraint(expr=m.soc[0] == soc_init)
    m.soc_term = pyo.Constraint(expr=m.soc[T] >= soc_init)

    # Energy balance (driving energy is mandatory, unmet is a penalised slack).
    def balance(m, t):
        return m.soc[t + 1] == (m.soc[t] + ETA_CH * m.pch[t] * DT_H
                                - m.pdis[t] * DT_H / ETA_DIS
                                - float(edrive[t]) + m.unmet[t])
    m.balance = pyo.Constraint(m.T, rule=balance)

    # FCR symmetric reservation.
    if fcr_on:
        bid = {t: block_id[t] for t in range(T)}

        def fcr_cap_up(m, t): return m.rfcr[bid[t]] <= block_pdis_min[bid[t]]
        def fcr_cap_dn(m, t): return m.rfcr[bid[t]] <= block_pch_min[bid[t]]
        m.fcr_cap_up = pyo.Constraint(m.T, rule=fcr_cap_up)
        m.fcr_cap_dn = pyo.Constraint(m.T, rule=fcr_cap_dn)

        def fcr_head_lo(m, t): return m.soc[t] - m.rfcr[bid[t]] * TRES_FCR_H >= soc_min
        def fcr_head_hi(m, t): return m.soc[t] + m.rfcr[bid[t]] * TRES_FCR_H <= soc_max
        m.fcr_head_lo = pyo.Constraint(m.T, rule=fcr_head_lo)
        m.fcr_head_hi = pyo.Constraint(m.T, rule=fcr_head_hi)

        # Reserve part of the depot grid connection for FCR response. Without this,
        # the same transformer capacity can be double-counted for arbitrage and reserve.
        if FCR_GRID_RESERVE_ON:
            def fcr_grid_reserve(m, t):
                return m.pch[t] + m.pdis[t] + m.rfcr[bid[t]] <= grid_limit
            m.fcr_grid_reserve = pyo.Constraint(m.T, rule=fcr_grid_reserve)

    # Objective: minimise net cost.
    energy_cost = sum((buy[t] * m.pch[t] - sell[t] * m.pdis[t]) * DT_H for t in range(T))
    deg_cost = (C_DEG_EUR_PER_MWH * sum((m.pch[t] + m.pdis[t]) * DT_H for t in range(T))
                if deg_on else 0.0)
    unmet_pen = UNMET_PENALTY * sum(m.unmet[t] for t in range(T))

    if fcr_on:
        # Gross market revenue is reduced by an expected auction acceptance rate and
        # aggregator/BSP fee. Activation and ramp-delay penalties are explicit costs.
        fcr_gross_rev = sum(block_price[b] * m.rfcr[b] for b in blocks)
        fcr_net_rev = sum(
            block_price[b] * FCR_ACCEPTANCE_RATE * (1.0 - FCR_AGGREGATOR_FEE_SHARE) * m.rfcr[b]
            for b in blocks
        )
        fcr_activation_cost = FCR_ACTIVATION_COST_EUR_PER_MW_BLOCK * sum(m.rfcr[b] for b in blocks)
        ramp_delay_pen = RAMP_DELAY_PENALTY_EUR_PER_MW_BLOCK * sum(m.rfcr[b] for b in blocks)
    else:
        fcr_gross_rev = 0.0
        fcr_net_rev = 0.0
        fcr_activation_cost = 0.0
        ramp_delay_pen = 0.0

    m.obj = pyo.Objective(
        expr=energy_cost + deg_cost + unmet_pen + fcr_activation_cost + ramp_delay_pen - fcr_net_rev,
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory(SOLVER)
    res = solver.solve(m)

    # Extract components.
    pch_v = np.array([pyo.value(m.pch[t]) for t in range(T)])
    pdis_v = np.array([pyo.value(m.pdis[t]) for t in range(T)])
    unmet_v = np.array([pyo.value(m.unmet[t]) for t in range(T)])
    charging_cost = float((buy * pch_v * DT_H).sum())
    discharge_rev = float((sell * pdis_v * DT_H).sum())
    throughput = float(((pch_v + pdis_v) * DT_H).sum())
    deg_val = C_DEG_EUR_PER_MWH * throughput if deg_on else 0.0

    if fcr_on:
        rfcr_v = {b: float(pyo.value(m.rfcr[b])) for b in blocks}
        fcr_gross_val = float(sum(block_price[b] * rfcr_v[b] for b in blocks))
        fcr_accepted_gross_val = float(sum(block_price[b] * FCR_ACCEPTANCE_RATE * rfcr_v[b] for b in blocks))
        fcr_net_val = float(sum(
            block_price[b] * FCR_ACCEPTANCE_RATE * (1.0 - FCR_AGGREGATOR_FEE_SHARE) * rfcr_v[b]
            for b in blocks
        ))
        fcr_acceptance_haircut_val = fcr_gross_val - fcr_accepted_gross_val
        fcr_aggregator_fee_val = fcr_accepted_gross_val * FCR_AGGREGATOR_FEE_SHARE
        fcr_activation_val = FCR_ACTIVATION_COST_EUR_PER_MW_BLOCK * sum(rfcr_v.values())
        ramp_delay_val = RAMP_DELAY_PENALTY_EUR_PER_MW_BLOCK * sum(rfcr_v.values())
        fcr_capacity_mw_blocks = float(sum(rfcr_v.values()))
        positive_blocks = [v for v in rfcr_v.values() if v > 1e-6]
        fcr_avg_positive = float(np.mean(positive_blocks)) if positive_blocks else 0.0
        fcr_blocks_below_1mw = int(sum((v > 1e-6) and (v < FCR_MIN_REPORTING_MW) for v in rfcr_v.values()))
    else:
        rfcr_v = {}
        fcr_gross_val = 0.0
        fcr_net_val = 0.0
        fcr_acceptance_haircut_val = 0.0
        fcr_aggregator_fee_val = 0.0
        fcr_activation_val = 0.0
        ramp_delay_val = 0.0
        fcr_capacity_mw_blocks = 0.0
        fcr_avg_positive = 0.0
        fcr_blocks_below_1mw = 0

    net_cost = (charging_cost - discharge_rev - fcr_net_val + deg_val
                + fcr_activation_val + ramp_delay_val)

    return {"scenario": label, "degradation": deg_on,
            "net_cost_EUR": net_cost, "profit_EUR": -net_cost,
            "charging_cost_EUR": charging_cost, "discharge_revenue_EUR": discharge_rev,
            "fcr_revenue_EUR": fcr_net_val, "fcr_gross_revenue_EUR": fcr_gross_val,
            "fcr_acceptance_haircut_EUR": fcr_acceptance_haircut_val,
            "fcr_aggregator_fee_EUR": fcr_aggregator_fee_val,
            "fcr_activation_cost_EUR": fcr_activation_val,
            "ramp_delay_penalty_EUR": ramp_delay_val,
            "degradation_cost_EUR": deg_val,
            "fcr_capacity_MW_blocks": fcr_capacity_mw_blocks,
            "fcr_avg_MW_when_positive": fcr_avg_positive,
            "fcr_blocks_below_1MW": fcr_blocks_below_1mw,
            "throughput_MWh": throughput, "unmet_MWh": float(unmet_v.sum()),
            "_pch": pch_v, "_pdis": pdis_v, "_rfcr": rfcr_v}


# ============================================================
# 5. RUN ALL
# ============================================================

def main() -> None:
    idx = build_index()
    print(f"Steps: {len(idx)}  ({DT_H} h each, {len(idx)*DT_H/24:.0f} days)")

    data = load_mobility(idx)
    da = load_da(idx)
    idp = load_id(idx)
    fcr_price, block_id = load_fcr(idx)
    grid_limit = GRID_LIMIT_FRACTION * float(max(data["pch_max"].max(), data["pdis_max"].max()))
    print(f"Capacity {data['cap']:.0f} MWh | SOC [{data['soc_min']:.0f}, {data['soc_max']:.0f}] | "
          f"grid limit {grid_limit:.0f} MW")

    rows = [run_naive(data, da, grid_limit)]

    combined_buy = np.minimum(da, idp)
    combined_sell = np.maximum(da, idp)
    scenarios = [
        ("DA",       da,          da,           True,  False),
        ("ID",       idp,         idp,          True,  False),
        ("FCR",      da,          da,           False, True),
        ("Combined", combined_buy, combined_sell, True, True),
    ]

    dispatch_for_sheet = None
    fcr_blocks_for_sheet = None
    for label, buy, sell, allow_dis, fcr_on in scenarios:
        for deg_on in (False, True):
            t0 = time.time()
            r = run_opt(data, buy, sell, allow_dis, fcr_on, fcr_price, block_id,
                        deg_on, grid_limit, label)
            tag = "deg" if deg_on else "no-deg"
            print(f"  {label:9s} {tag:7s} profit EUR {r['profit_EUR']:>14,.0f}  "
                  f"net FCR {r['fcr_revenue_EUR']:>12,.0f}  ramp penalty {r['ramp_delay_penalty_EUR']:>10,.0f}  "
                  f"unmet {r['unmet_MWh']:.1f}  ({time.time()-t0:.1f}s)")
            if label == "Combined" and not deg_on:
                dispatch_for_sheet = pd.DataFrame({
                    "datetime": idx, "p_charge_MW": r.pop("_pch"), "p_discharge_MW": r.pop("_pdis"),
                })
                if r.get("_rfcr"):
                    fcr_blocks_for_sheet = pd.DataFrame([
                        {"block_id": b, "rfcr_MW": v} for b, v in r["_rfcr"].items()
                    ])
            r.pop("_pch", None)
            r.pop("_pdis", None)
            r.pop("_rfcr", None)
            rows.append(r)

    comparison = pd.DataFrame(rows)[[
        "scenario", "degradation", "profit_EUR", "net_cost_EUR", "charging_cost_EUR",
        "discharge_revenue_EUR", "fcr_revenue_EUR", "fcr_gross_revenue_EUR",
        "fcr_acceptance_haircut_EUR", "fcr_aggregator_fee_EUR",
        "fcr_activation_cost_EUR", "ramp_delay_penalty_EUR",
        "degradation_cost_EUR", "fcr_capacity_MW_blocks", "fcr_avg_MW_when_positive",
        "fcr_blocks_below_1MW", "throughput_MWh", "unmet_MWh"]]

    assumptions = pd.DataFrame([
        ("run_timestamp", RUN_STAMP), ("year", YEAR), ("resolution_h", DT_H),
        ("horizon_days", "full year" if HORIZON_DAYS is None else HORIZON_DAYS),
        ("eta_charge", ETA_CH), ("eta_discharge", ETA_DIS),
        ("soc_init_and_terminal_share", SOC_INIT_SHARE),
        ("fcr_reservation_h", TRES_FCR_H),
        ("fcr_acceptance_rate", FCR_ACCEPTANCE_RATE),
        ("fcr_aggregator_fee_share", FCR_AGGREGATOR_FEE_SHARE),
        ("fcr_activation_cost_EUR_per_MW_block", FCR_ACTIVATION_COST_EUR_PER_MW_BLOCK),
        ("ramp_delay_penalty_EUR_per_MW_block", RAMP_DELAY_PENALTY_EUR_PER_MW_BLOCK),
        ("fcr_grid_reserve_on", FCR_GRID_RESERVE_ON),
        ("fcr_min_reporting_MW", FCR_MIN_REPORTING_MW),
        ("degradation_cost_EUR_per_MWh", C_DEG_EUR_PER_MWH),
        ("grid_limit_MW", round(grid_limit, 1)),
        ("da_source", "ENTSO-E Sequence 1 hourly"), ("id_source", "IDC index"),
        ("fcr_source", "German NEGPOS symmetric capacity price"),
    ], columns=["Parameter", "Value"])

    out = OUTPUT_FOLDER / f"dhl_market_comparison_{RUN_STAMP}.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        comparison.to_excel(w, sheet_name="Comparison", index=False)
        assumptions.to_excel(w, sheet_name="Assumptions", index=False)
        if dispatch_for_sheet is not None:
            dispatch_for_sheet.to_excel(w, sheet_name="Combined_Dispatch_nodeg", index=False)
        if fcr_blocks_for_sheet is not None:
            fcr_blocks_for_sheet.to_excel(w, sheet_name="Combined_FCR_blocks_nodeg", index=False)

    print("\nComparison written to:", out)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

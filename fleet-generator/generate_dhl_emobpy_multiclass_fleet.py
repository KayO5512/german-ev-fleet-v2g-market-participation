# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 04:14:14 2026

@author: Aidoo
"""

# -*- coding: utf-8 -*-
"""
Generate a synthetic DHL-style multiclass EV fleet dataset for optimisation with emobpy logic.

Purpose
-------
This script creates synthetic, anonymised, academic-use mobility and charging datasets similar to
what you requested from DHL: departure/return times, route distances, parking/charging availability,
battery sizes, charger ratings, SOC assumptions, V2G power availability and hourly fleet-level inputs
for Pyomo optimisation.

It explicitly includes the vehicle classes described in the project request:
- Volvo FL Electric 4x2 heavy trucks
- Mercedes-Benz eActros 600 long-haul trucks
- other heavy-duty electric trucks
- DHL/StreetScooter-type electric delivery vans
- 2-person-handling / bulky-goods electric vans
- e-trikes
- e-bikes
- optional biogas trucks as non-EV reference vehicles, excluded from EV optimisation by default

Method
------
emobpy is mainly built around the chain:
    mobility profile -> driving consumption -> grid availability -> grid demand

This script follows that same structure, while using a custom DHL commercial-fleet profile because
standard emobpy examples are mostly private-car oriented. If emobpy is installed, the metadata records
that the environment was emobpy-enabled. The exported trip table is intentionally emobpy-compatible
so that you can later map these trips into native emobpy objects if desired.

Input
-----
A DT-CARGO-derived Excel workbook, preferably the user's dtcargo_results.xlsx file. The script uses it
for German heavy-truck empirical patterns such as depot arrival/departure hours, route lengths and
parking durations. If the file is missing, the script falls back to transparent synthetic distributions.

Install
-------
pip install pandas numpy openpyxl emobpy

Run
---
python generate_dhl_emobpy_multiclass_fleet.py

Main output for optimisation
----------------------------
dhl_mobility_timeseries_1h_2025_synthetic_34000_multiclass.csv
"""



import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import emobpy  # noqa: F401
    EMOBPY_AVAILABLE = True
except Exception:
    EMOBPY_AVAILABLE = False


# ============================================================
# 1. USER SETTINGS
# ============================================================

# Relative paths so this runs the same for any team member who clones the repo.
# Put dtcargo_results.xlsx in a "data" folder next to this script.
DT_CARGO_WORKBOOK = Path("data/dtcargo_results.xlsx")

OUTPUT_FOLDER = Path("outputs")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

TARGET_YEAR = 2025
TARGET_TOTAL_EV_FLEET = 34_000

# Keeps disaggregated event files manageable. Increase to 34_000 for a full vehicle-level dataset.
SAMPLE_TOTAL_EV_FLEET = 34

# MVP STRATIFIED-SAMPLING CONTROL
# -------------------------------
# Our MVP pools the fleet into ONE aggregated virtual battery. What flows into Pyomo
# is hourly fleet totals (capacity, charge power, discharge power, driving energy),
# scaled by each class's target_weight. For that aggregation to represent the real
# DHL vehicle variety, every EV class must appear in the sample at least once.
#
# With pure proportional sampling, rare classes (the heavy trucks: 13 Volvo FL,
# 16 other heavy-duty, 30 eActros out of 34,000) round to zero and vanish. This
# floor guarantees each class is represented; target_weight then scales the single
# sampled truck up to its full class count. Raise it if you want smoother truck
# contributions to the hourly profile (each truck class currently rides on 1 sample).
MIN_SAMPLES_PER_CLASS = 1

SEED = 20250614
rng = np.random.default_rng(SEED)

# Fleet composition mode:
# - "scale_to_34000": includes all listed EV classes and scales their reference counts to exactly 34,000.
# - "raw_counts": uses the raw counts as listed below, which sum to more than 34,000.
FLEET_COMPOSITION_MODE = "scale_to_34000"

# Include non-EV biogas trucks in reference files only, not in EV optimisation totals.
INCLUDE_BIOGAS_REFERENCE = True

# Optimisation/SOC assumptions.
SOC_MIN_SHARE = 0.20
SOC_INITIAL_SHARE = 0.60
SOC_MAX_SHARE = 0.95
V2G_ENABLED_SHARE = 0.70
DISCHARGE_POWER_DERATE = 0.80
CHARGING_EFFICIENCY = 0.92

# Calendar behaviour.
SATURDAY_ACTIVITY_FACTOR = 0.60
SUNDAY_ACTIVITY_FACTOR = 0.15
WINTER_ENERGY_FACTOR = 1.08
SUMMER_ENERGY_FACTOR = 0.97


# ============================================================
# 2. VEHICLE CLASS DEFINITIONS
# ============================================================

@dataclass(frozen=True)
class VehicleClass:
    vehicle_class: str
    raw_reference_count: int
    battery_kWh: float
    ac_charger_kW: float
    dc_charger_kW: float
    charger_kW_used: float
    v2g_power_kW: float
    consumption_kWh_per_km: float
    range_km: float
    class_group: str
    is_ev: bool
    include_in_optimisation: bool
    first_operation_year: int
    empirical_profile: str
    notes: str


RAW_VEHICLE_CLASSES: List[VehicleClass] = [
    VehicleClass(
        vehicle_class="streetscooter_last_mile_e_van",
        raw_reference_count=21_500,
        battery_kWh=40,
        ac_charger_kW=11,
        dc_charger_kW=0,
        charger_kW_used=11,
        v2g_power_kW=10,
        consumption_kWh_per_km=0.22,
        range_km=160,
        class_group="light_commercial_van",
        is_ev=True,
        include_in_optimisation=True,
        first_operation_year=2025,
        empirical_profile="urban parcel delivery, depot-return, emobpy-style work-location parking",
        notes="Represents DHL StreetScooter-type electric vans and similar last-mile vans.",
    ),
    VehicleClass(
        vehicle_class="large_bulky_goods_e_van_2mh",
        raw_reference_count=21,
        battery_kWh=75,
        ac_charger_kW=22,
        dc_charger_kW=50,
        charger_kW_used=22,
        v2g_power_kW=20,
        consumption_kWh_per_km=0.32,
        range_km=220,
        class_group="light_commercial_van",
        is_ev=True,
        include_in_optimisation=True,
        first_operation_year=2025,
        empirical_profile="bulky goods / two-person handling regional delivery",
        notes="The 21 new electric vans for bulky-goods delivery across German states.",
    ),
    VehicleClass(
        vehicle_class="e_trike",
        raw_reference_count=5_000,
        battery_kWh=2.0,
        ac_charger_kW=0.8,
        dc_charger_kW=0,
        charger_kW_used=0.8,
        v2g_power_kW=0,
        consumption_kWh_per_km=0.035,
        range_km=45,
        class_group="micro_mobility",
        is_ev=True,
        include_in_optimisation=True,
        first_operation_year=2025,
        empirical_profile="dense urban mail and parcel delivery",
        notes="No V2G assumed because batteries are small/removable and usually not grid-export capable.",
    ),
    VehicleClass(
        vehicle_class="e_bike",
        raw_reference_count=8_000,
        battery_kWh=0.6,
        ac_charger_kW=0.25,
        dc_charger_kW=0,
        charger_kW_used=0.25,
        v2g_power_kW=0,
        consumption_kWh_per_km=0.012,
        range_km=60,
        class_group="micro_mobility",
        is_ev=True,
        include_in_optimisation=True,
        first_operation_year=2025,
        empirical_profile="dense urban mail and parcel delivery",
        notes="Included for energy demand and charging load, but not for V2G services.",
    ),
    VehicleClass(
        vehicle_class="volvo_fl_electric_4x2_berlin",
        raw_reference_count=13,
        battery_kWh=265,
        ac_charger_kW=22,
        dc_charger_kW=150,
        charger_kW_used=150,
        v2g_power_kW=120,
        consumption_kWh_per_km=0.90,
        range_km=300,
        class_group="medium_heavy_truck",
        is_ev=True,
        include_in_optimisation=True,
        first_operation_year=2025,
        empirical_profile="DT-CARGO German heavy-truck route and depot parking profile",
        notes="13 Volvo FL Electric 4x2 trucks in Berlin, 16.7 t gross weight, up to 130 kW motor, around 300 km range.",
    ),
    VehicleClass(
        vehicle_class="other_heavy_duty_e_truck",
        raw_reference_count=16,
        battery_kWh=320,
        ac_charger_kW=22,
        dc_charger_kW=150,
        charger_kW_used=150,
        v2g_power_kW=120,
        consumption_kWh_per_km=1.05,
        range_km=300,
        class_group="medium_heavy_truck",
        is_ev=True,
        include_in_optimisation=True,
        first_operation_year=2025,
        empirical_profile="DT-CARGO German heavy-truck route and depot parking profile",
        notes="Other heavy-duty electric trucks already in operation.",
    ),
    VehicleClass(
        vehicle_class="mercedes_benz_eactros_600_long_haul",
        raw_reference_count=30,
        battery_kWh=621,
        ac_charger_kW=22,
        dc_charger_kW=400,
        charger_kW_used=350,
        v2g_power_kW=250,
        consumption_kWh_per_km=1.15,
        range_km=500,
        class_group="long_haul_truck",
        is_ev=True,
        include_in_optimisation=True,
        first_operation_year=2026,
        empirical_profile="DT-CARGO German heavy-truck route profile, scaled to long-haul duty",
        notes="30 eActros 600 long-haul trucks. Included as a scenario vehicle class even though operational readiness is mid-2026.",
    ),
    VehicleClass(
        vehicle_class="biogas_regional_truck_reference_non_ev",
        raw_reference_count=350,
        battery_kWh=0,
        ac_charger_kW=0,
        dc_charger_kW=0,
        charger_kW_used=0,
        v2g_power_kW=0,
        consumption_kWh_per_km=0,
        range_km=600,
        class_group="non_ev_reference_truck",
        is_ev=False,
        include_in_optimisation=False,
        first_operation_year=2023,
        empirical_profile="DT-CARGO German truck reference only",
        notes="Biogas trucks are not EVs, so they are excluded from charging, SOC and V2G optimisation outputs by default.",
    ),
]


def build_target_fleet_table() -> pd.DataFrame:
    raw = pd.DataFrame([asdict(v) for v in RAW_VEHICLE_CLASSES])
    raw["target_count"] = 0

    ev_mask = raw["is_ev"] & raw["include_in_optimisation"]
    if FLEET_COMPOSITION_MODE == "raw_counts":
        raw.loc[ev_mask, "target_count"] = raw.loc[ev_mask, "raw_reference_count"].astype(int)
    elif FLEET_COMPOSITION_MODE == "scale_to_34000":
        raw_ev_total = raw.loc[ev_mask, "raw_reference_count"].sum()
        scaled = raw.loc[ev_mask, "raw_reference_count"] / raw_ev_total * TARGET_TOTAL_EV_FLEET
        floors = np.floor(scaled).astype(int)
        raw.loc[ev_mask, "target_count"] = floors
        remainder = TARGET_TOTAL_EV_FLEET - int(floors.sum())
        fractional_order = (scaled - floors).sort_values(ascending=False).index.tolist()
        for idx in fractional_order[:remainder]:
            raw.loc[idx, "target_count"] += 1
    else:
        raise ValueError("FLEET_COMPOSITION_MODE must be 'scale_to_34000' or 'raw_counts'.")

    if INCLUDE_BIOGAS_REFERENCE:
        raw.loc[~ev_mask, "target_count"] = raw.loc[~ev_mask, "raw_reference_count"]
    raw["sample_count"] = 0
    ev_target = raw.loc[ev_mask, "target_count"].sum()
    n_ev_classes = int(ev_mask.sum())

    # Guard: the sample must be large enough to give every class its minimum.
    min_total = MIN_SAMPLES_PER_CLASS * n_ev_classes
    if SAMPLE_TOTAL_EV_FLEET < min_total:
        raise ValueError(
            f"SAMPLE_TOTAL_EV_FLEET ({SAMPLE_TOTAL_EV_FLEET}) is too small to give each of the "
            f"{n_ev_classes} EV classes at least {MIN_SAMPLES_PER_CLASS} sample(s). "
            f"Set SAMPLE_TOTAL_EV_FLEET >= {min_total}."
        )

    # Keep the proportional ideal for reference/reporting.
    scale = ev_target / SAMPLE_TOTAL_EV_FLEET
    raw.loc[ev_mask, "sample_float"] = raw.loc[ev_mask, "target_count"] / scale

    # Stratified allocation: reserve MIN_SAMPLES_PER_CLASS per class so rare classes
    # (the heavy trucks) always appear, then hand out the remaining slots by largest
    # remainder in proportion to target_count.
    remaining = SAMPLE_TOTAL_EV_FLEET - min_total
    shares = raw.loc[ev_mask, "target_count"] / ev_target
    extra_float = shares * remaining
    extra_floor = np.floor(extra_float).astype(int)
    raw.loc[ev_mask, "sample_count"] = MIN_SAMPLES_PER_CLASS + extra_floor
    leftover = remaining - int(extra_floor.sum())
    fractional_order = (extra_float - extra_floor).sort_values(ascending=False).index.tolist()
    for idx in fractional_order[:leftover]:
        raw.loc[idx, "sample_count"] += 1
    raw.loc[~ev_mask, "sample_float"] = 0

    raw["target_weight"] = np.where(raw["sample_count"] > 0, raw["target_count"] / raw["sample_count"], 0)
    return raw


# ============================================================
# 3. DT-CARGO INPUT AND FALLBACKS
# ============================================================

def _safe_read_excel(path: Path, sheet_name: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except Exception:
        return pd.DataFrame()


def load_dt_cargo_tables(path: Path) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    if path.exists():
        tables["arrivals_departures"] = _safe_read_excel(path, "Depot Arrivals & Departures")
        tables["parking"] = _safe_read_excel(path, "Parking Durations")
        tables["routes"] = _safe_read_excel(path, "Route Lengths")
        tables["duty_cycles"] = _safe_read_excel(path, "Duty Cycles")
        tables["pyomo_parameters"] = _safe_read_excel(path, "Pyomo Parameters")
    else:
        tables = {k: pd.DataFrame() for k in ["arrivals_departures", "parking", "routes", "duty_cycles", "pyomo_parameters"]}

    ad = tables["arrivals_departures"].copy()
    if {"arrival_hour", "departure_hour"}.issubset(ad.columns):
        if "vehicle_id" in ad.columns:
            ad = ad[pd.to_numeric(ad["vehicle_id"], errors="coerce").notna()].copy()
        ad["arrival_hour"] = pd.to_numeric(ad["arrival_hour"], errors="coerce")
        ad["departure_hour"] = pd.to_numeric(ad["departure_hour"], errors="coerce")
        ad = ad.dropna(subset=["arrival_hour", "departure_hour"])
    else:
        ad = pd.DataFrame({"arrival_hour": rng.normal(16.5, 2.5, 2000), "departure_hour": rng.normal(7.0, 1.5, 2000)})
    tables["arrivals_departures"] = ad

    parking = tables["parking"].copy()
    if "parking_hrs" in parking.columns:
        parking["parking_hrs"] = pd.to_numeric(parking["parking_hrs"], errors="coerce")
        parking = parking.dropna(subset=["parking_hrs"])
    else:
        parking = pd.DataFrame({"parking_hrs": rng.gamma(shape=3.0, scale=3.0, size=2000)})
    tables["parking"] = parking

    routes = tables["routes"].copy()
    if "distance_km" in routes.columns:
        routes["distance_km"] = pd.to_numeric(routes["distance_km"], errors="coerce")
        routes = routes.dropna(subset=["distance_km"])
        routes = routes[(routes["distance_km"] > 0.5) & (routes["distance_km"] < routes["distance_km"].quantile(0.995))]
    else:
        routes = pd.DataFrame({"distance_km": rng.lognormal(mean=np.log(45), sigma=0.75, size=5000)})
    tables["routes"] = routes

    return tables


# ============================================================
# 4. SAMPLING LOGIC
# ============================================================

def sample_hour(series: pd.Series, default: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(vals) == 0:
        return default
    return float(rng.choice(vals))


def is_operating_day(day: pd.Timestamp, class_group: str) -> bool:
    wd = day.weekday()
    if wd < 5:
        base = 0.96
    elif wd == 5:
        base = SATURDAY_ACTIVITY_FACTOR
    else:
        base = SUNDAY_ACTIVITY_FACTOR
    if class_group in {"medium_heavy_truck", "long_haul_truck"}:
        base *= 0.90
    if class_group == "micro_mobility" and day.month in [12, 1, 2]:
        base *= 0.90
    return bool(rng.random() < base)


def seasonal_energy_multiplier(ts: pd.Timestamp) -> float:
    if ts.month in [12, 1, 2]:
        return WINTER_ENERGY_FACTOR
    if ts.month in [6, 7, 8]:
        return SUMMER_ENERGY_FACTOR
    return 1.0


def sample_work_window(vehicle_class: str, class_group: str, tables: Dict[str, pd.DataFrame]) -> Tuple[float, float, float]:
    ad = tables["arrivals_departures"]
    parking = tables["parking"]

    if class_group in {"medium_heavy_truck", "long_haul_truck"}:
        dep = np.clip(sample_hour(ad["departure_hour"], 7.0), 3.0, 12.0)
        arr = np.clip(sample_hour(ad["arrival_hour"], 17.0), 10.0, 23.5)
        if arr <= dep + 2:
            arr = min(dep + rng.uniform(6.0, 13.0), 23.5)
        if class_group == "long_haul_truck":
            dep = np.clip(dep - rng.uniform(0.5, 2.0), 1.0, 10.0)
            arr = np.clip(dep + rng.uniform(8.0, 16.0), 12.0, 23.8)
        park = np.clip(sample_hour(parking["parking_hrs"], 8.0), 1.0, 18.0)
        return float(dep), float(arr), float(park)

    if class_group == "light_commercial_van":
        dep = float(np.clip(rng.normal(7.2, 0.9), 5.0, 10.5))
        if vehicle_class == "large_bulky_goods_e_van_2mh":
            arr = float(np.clip(dep + rng.normal(8.5, 1.4), 13.0, 20.5))
        else:
            arr = float(np.clip(dep + rng.normal(7.6, 1.3), 12.0, 19.5))
        return dep, arr, float(24 - arr + dep)

    if class_group == "micro_mobility":
        dep = float(np.clip(rng.normal(8.1, 0.8), 6.0, 10.5))
        arr = float(np.clip(dep + rng.normal(6.0, 1.1), 12.0, 18.0))
        return dep, arr, float(24 - arr + dep)

    return 8.0, 16.0, 16.0


def sample_route_distance(row: pd.Series, tables: Dict[str, pd.DataFrame]) -> float:
    class_group = row["class_group"]
    vehicle_class = row["vehicle_class"]
    range_km = float(row["range_km"])
    routes = tables["routes"]

    if class_group == "medium_heavy_truck":
        n = int(rng.integers(2, 6))
        d = float(rng.choice(routes["distance_km"].to_numpy(), size=n, replace=True).sum())
        return float(np.clip(d, 25, range_km * 0.92))

    if class_group == "long_haul_truck":
        n = int(rng.integers(4, 10))
        d = float(rng.choice(routes["distance_km"].to_numpy(), size=n, replace=True).sum())
        return float(np.clip(d, 250, range_km * 0.92))

    if vehicle_class == "large_bulky_goods_e_van_2mh":
        return float(np.clip(rng.lognormal(mean=np.log(85), sigma=0.35), 25, 180))

    if class_group == "light_commercial_van":
        return float(np.clip(rng.lognormal(mean=np.log(65), sigma=0.40), 15, 140))

    if vehicle_class == "e_trike":
        return float(np.clip(rng.normal(24, 8), 4, 45))

    if vehicle_class == "e_bike":
        return float(np.clip(rng.normal(18, 7), 3, 45))

    return 50.0


# ============================================================
# 5. GENERATE EVENTS
# ============================================================

def generate_events(fleet: pd.DataFrame, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    days = pd.date_range(f"{TARGET_YEAR}-01-01", f"{TARGET_YEAR + 1}-01-01", freq="D", inclusive="left")
    events = []
    counter = 0

    active = fleet[(fleet["is_ev"]) & (fleet["include_in_optimisation"]) & (fleet["sample_count"] > 0)].copy()
    for _, row in active.iterrows():
        for local_i in range(int(row["sample_count"])):
            counter += 1
            vehicle_id = f"{row['vehicle_class']}_{counter:06d}"
            for day in days:
                if not is_operating_day(day, row["class_group"]):
                    continue

                dep_h, ret_h, park_h = sample_work_window(row["vehicle_class"], row["class_group"], tables)
                dep = day + pd.to_timedelta(dep_h, unit="h")
                ret = day + pd.to_timedelta(ret_h, unit="h")
                if ret <= dep:
                    ret += pd.Timedelta(days=1)

                distance_km = sample_route_distance(row, tables)
                driving_energy_kWh = distance_km * float(row["consumption_kWh_per_km"]) * seasonal_energy_multiplier(dep)
                driving_energy_kWh = min(driving_energy_kWh, float(row["battery_kWh"]) * 0.82)
                active_hours = max((ret - dep).total_seconds() / 3600, 0.25)

                events.append({
                    "vehicle_id": vehicle_id,
                    "vehicle_class": row["vehicle_class"],
                    "class_group": row["class_group"],
                    "date": day.date().isoformat(),
                    "departure_time": dep,
                    "return_time": ret,
                    "departure_hour": dep_h,
                    "return_hour": ret_h,
                    "active_hours": active_hours,
                    "parking_hrs_after_return": park_h,
                    "distance_km": distance_km,
                    "driving_energy_kWh": driving_energy_kWh,
                    "battery_kWh": float(row["battery_kWh"]),
                    "ac_charger_kW": float(row["ac_charger_kW"]),
                    "dc_charger_kW": float(row["dc_charger_kW"]),
                    "charger_kW": float(row["charger_kW_used"]),
                    "v2g_available_power_kW": float(row["v2g_power_kW"]),
                    "range_km": float(row["range_km"]),
                    "target_weight": float(row["target_weight"]),
                    "location_start": "depot",
                    "location_end": "depot",
                    "emobpy_state_while_driving": "driving",
                    "emobpy_state_while_parked": "workplace",
                    "emobpy_charging_point": "depot_or_workplace",
                    "soc_initial_share": SOC_INITIAL_SHARE,
                    "soc_min_share": SOC_MIN_SHARE,
                    "soc_max_share": SOC_MAX_SHARE,
                    "first_operation_year": int(row["first_operation_year"]),
                    "note": row["notes"],
                })

    return pd.DataFrame(events)


# ============================================================
# 6. AGGREGATE TO HOURLY TIMESERIES
# ============================================================

def aggregate_hourly(events: pd.DataFrame, fleet: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range(f"{TARGET_YEAR}-01-01 00:00:00", f"{TARGET_YEAR + 1}-01-01 00:00:00", freq="1h", inclusive="left")
    base = pd.DataFrame({"datetime": idx})
    base["date"] = base["datetime"].dt.date
    base["month"] = base["datetime"].dt.month
    base["day_of_year"] = base["datetime"].dt.dayofyear
    base["weekday"] = base["datetime"].dt.day_name()
    base["weekday_number"] = base["datetime"].dt.weekday
    base["hour"] = base["datetime"].dt.hour

    ev_fleet = fleet[(fleet["is_ev"]) & (fleet["include_in_optimisation"])]
    cap_total = float((ev_fleet["target_count"] * ev_fleet["battery_kWh"]).sum() / 1000)
    total_fleet = int(ev_fleet["target_count"].sum())

    fleet_arrays = {
        "driving_vehicles": np.zeros(len(base)),
        "available_vehicles": np.zeros(len(base)),
        "p_charge_max_MW": np.zeros(len(base)),
        "p_discharge_max_MW": np.zeros(len(base)),
        "driving_energy_MWh": np.zeros(len(base)),
    }

    class_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for vc in ev_fleet["vehicle_class"]:
        class_arrays[vc] = {k: np.zeros(len(base)) for k in fleet_arrays}

    time_to_pos = {t: i for i, t in enumerate(idx)}

    for e in events.itertuples(index=False):
        current = pd.Timestamp(e.departure_time).floor("1h")
        end = pd.Timestamp(e.return_time).ceil("1h")
        hourly_energy = e.driving_energy_kWh / max(e.active_hours, 0.25) / 1000 * e.target_weight
        while current < end:
            pos = time_to_pos.get(current)
            if pos is not None:
                fleet_arrays["driving_vehicles"][pos] += e.target_weight
                fleet_arrays["driving_energy_MWh"][pos] += hourly_energy
                class_arrays[e.vehicle_class]["driving_vehicles"][pos] += e.target_weight
                class_arrays[e.vehicle_class]["driving_energy_MWh"][pos] += hourly_energy
            current += pd.Timedelta(hours=1)

        current = pd.Timestamp(e.return_time).ceil("1h")
        park_end = current + pd.to_timedelta(e.parking_hrs_after_return, unit="h")
        while current < park_end:
            pos = time_to_pos.get(current)
            if pos is not None:
                charge_MW = e.charger_kW * e.target_weight / 1000
                discharge_MW = e.v2g_available_power_kW * V2G_ENABLED_SHARE * DISCHARGE_POWER_DERATE * e.target_weight / 1000
                fleet_arrays["available_vehicles"][pos] += e.target_weight
                fleet_arrays["p_charge_max_MW"][pos] += charge_MW
                fleet_arrays["p_discharge_max_MW"][pos] += discharge_MW
                class_arrays[e.vehicle_class]["available_vehicles"][pos] += e.target_weight
                class_arrays[e.vehicle_class]["p_charge_max_MW"][pos] += charge_MW
                class_arrays[e.vehicle_class]["p_discharge_max_MW"][pos] += discharge_MW
            current += pd.Timedelta(hours=1)

    hourly = base.copy()
    for k, arr in fleet_arrays.items():
        hourly[k] = arr
    hourly["fleet_size"] = total_fleet
    hourly["battery_capacity_MWh_total"] = cap_total
    hourly["soc_min_MWh"] = cap_total * SOC_MIN_SHARE
    hourly["soc_initial_MWh"] = cap_total * SOC_INITIAL_SHARE
    hourly["soc_max_MWh"] = cap_total * SOC_MAX_SHARE
    hourly["battery_kWh_per_vehicle_weighted_avg"] = cap_total * 1000 / total_fleet
    hourly["charger_power_kW_per_available_vehicle_avg"] = np.where(hourly["available_vehicles"] > 0, hourly["p_charge_max_MW"] * 1000 / hourly["available_vehicles"], 0)
    hourly["avg_daily_energy_kWh_per_vehicle"] = hourly.groupby("date")["driving_energy_MWh"].transform("sum") * 1000 / total_fleet
    hourly["available_vehicles"] = hourly["available_vehicles"].clip(upper=total_fleet)
    hourly["driving_vehicles"] = hourly["driving_vehicles"].clip(upper=total_fleet)

    by_class_frames = []
    for vc, arrays in class_arrays.items():
        temp = base.copy()
        temp["vehicle_class"] = vc
        for k, arr in arrays.items():
            temp[k] = arr
        target_count = int(ev_fleet.loc[ev_fleet["vehicle_class"] == vc, "target_count"].iloc[0])
        battery_kWh = float(ev_fleet.loc[ev_fleet["vehicle_class"] == vc, "battery_kWh"].iloc[0])
        temp["fleet_size_class"] = target_count
        temp["battery_capacity_MWh_class"] = target_count * battery_kWh / 1000
        by_class_frames.append(temp)
    hourly_by_class = pd.concat(by_class_frames, ignore_index=True)

    return hourly, hourly_by_class


# ============================================================
# 7. OUTPUTS
# ============================================================

def build_class_summary(events: pd.DataFrame, fleet: pd.DataFrame) -> pd.DataFrame:
    event_summary = events.groupby(["vehicle_class", "class_group"], as_index=False).agg(
        sampled_vehicle_days=("vehicle_id", "count"),
        mean_distance_km=("distance_km", "mean"),
        p50_distance_km=("distance_km", "median"),
        mean_driving_energy_kWh=("driving_energy_kWh", "mean"),
        mean_departure_hour=("departure_hour", "mean"),
        mean_return_hour=("return_hour", "mean"),
        mean_parking_hrs_after_return=("parking_hrs_after_return", "mean"),
    )
    keep_cols = [
        "vehicle_class", "raw_reference_count", "target_count", "sample_count", "target_weight",
        "battery_kWh", "charger_kW_used", "v2g_power_kW", "consumption_kWh_per_km", "range_km",
        "is_ev", "include_in_optimisation", "first_operation_year", "empirical_profile", "notes"
    ]
    summary = fleet[keep_cols].merge(event_summary, on="vehicle_class", how="left")
    summary["total_battery_capacity_MWh"] = summary["target_count"] * summary["battery_kWh"] / 1000
    summary["total_charging_capacity_MW"] = summary["target_count"] * summary["charger_kW_used"] / 1000
    summary["total_nameplate_v2g_capacity_MW"] = summary["target_count"] * summary["v2g_power_kW"] / 1000
    return summary


def save_metadata(fleet: pd.DataFrame, hourly: pd.DataFrame, tables: Dict[str, pd.DataFrame]) -> None:
    metadata = {
        "target_year": TARGET_YEAR,
        "fleet_composition_mode": FLEET_COMPOSITION_MODE,
        "target_total_ev_fleet": int(fleet[(fleet["is_ev"]) & (fleet["include_in_optimisation"])]["target_count"].sum()),
        "raw_reference_ev_total_before_scaling": int(fleet[(fleet["is_ev"]) & (fleet["include_in_optimisation"])]["raw_reference_count"].sum()),
        "sample_total_ev_fleet": SAMPLE_TOTAL_EV_FLEET,
        "seed": SEED,
        "emobpy_available_in_environment": EMOBPY_AVAILABLE,
        "method": "DT-CARGO empirical sampling + emobpy-style mobility, consumption, grid availability and charging aggregation",
        "important_note": "This is a synthetic academic dataset, not official DHL operational data.",
        "soc_assumptions": {
            "soc_min_share": SOC_MIN_SHARE,
            "soc_initial_share": SOC_INITIAL_SHARE,
            "soc_max_share": SOC_MAX_SHARE,
            "charging_efficiency": CHARGING_EFFICIENCY,
            "v2g_enabled_share": V2G_ENABLED_SHARE,
            "discharge_power_derate": DISCHARGE_POWER_DERATE,
        },
        "hourly_summary": {
            "rows": int(len(hourly)),
            "total_annual_driving_energy_MWh": float(hourly["driving_energy_MWh"].sum()),
            "mean_available_vehicles": float(hourly["available_vehicles"].mean()),
            "mean_driving_vehicles": float(hourly["driving_vehicles"].mean()),
            "max_charge_power_MW": float(hourly["p_charge_max_MW"].max()),
            "max_discharge_power_MW": float(hourly["p_discharge_max_MW"].max()),
            "total_battery_capacity_MWh": float(hourly["battery_capacity_MWh_total"].iloc[0]),
        },
        "vehicle_classes": fleet.to_dict(orient="records"),
        "dt_cargo_workbook_used": str(DT_CARGO_WORKBOOK),
        "dt_cargo_workbook_found": DT_CARGO_WORKBOOK.exists(),
        "dt_cargo_sheets_loaded": {k: int(len(v)) for k, v in tables.items()},
    }
    with open(OUTPUT_FOLDER / "metadata_assumptions_multiclass.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


def main() -> None:
    print("Building multiclass DHL EV fleet table...")
    fleet = build_target_fleet_table()

    print("Loading DT-CARGO empirical workbook, or fallback distributions if missing...")
    tables = load_dt_cargo_tables(DT_CARGO_WORKBOOK)

    print("Generating synthetic vehicle-day events...")
    events = generate_events(fleet, tables)

    print("Aggregating to hourly fleet and class-level time series...")
    hourly, hourly_by_class = aggregate_hourly(events, fleet)

    print("Preparing emobpy-compatible trip table and class summary...")
    class_summary = build_class_summary(events, fleet)
    emobpy_trips = events[[
        "vehicle_id", "vehicle_class", "class_group", "departure_time", "return_time", "active_hours",
        "distance_km", "driving_energy_kWh", "location_start", "location_end",
        "emobpy_state_while_driving", "emobpy_state_while_parked", "emobpy_charging_point",
        "battery_kWh", "charger_kW", "v2g_available_power_kW", "target_weight",
        "soc_initial_share", "soc_min_share", "soc_max_share", "first_operation_year"
    ]].copy()

    hourly_file = OUTPUT_FOLDER / f"dhl_mobility_timeseries_1h_{TARGET_YEAR}_synthetic_34000_multiclass.csv"
    class_hourly_file = OUTPUT_FOLDER / f"dhl_mobility_timeseries_1h_{TARGET_YEAR}_by_vehicle_class.csv"
    events_file = OUTPUT_FOLDER / f"dhl_vehicle_events_{TARGET_YEAR}_synthetic_sample_scaled_multiclass.csv"
    emobpy_file = OUTPUT_FOLDER / f"dhl_emobpy_compatible_trip_table_{TARGET_YEAR}_multiclass.csv"
    summary_file = OUTPUT_FOLDER / "dhl_vehicle_class_summary_multiclass.csv"
    fleet_file = OUTPUT_FOLDER / "dhl_fleet_composition_multiclass.csv"

    hourly.to_csv(hourly_file, index=False, encoding="utf-8-sig")
    hourly_by_class.to_csv(class_hourly_file, index=False, encoding="utf-8-sig")
    events.to_csv(events_file, index=False, encoding="utf-8-sig")
    emobpy_trips.to_csv(emobpy_file, index=False, encoding="utf-8-sig")
    class_summary.to_csv(summary_file, index=False, encoding="utf-8-sig")
    fleet.to_csv(fleet_file, index=False, encoding="utf-8-sig")
    save_metadata(fleet, hourly, tables)

    # Show the full DHL EV vehicle variety actually represented in this run,
    # so the heavy trucks are visible and confirmed present (not rounded out).
    print("\nDHL EV vehicle variety represented in this run:")
    variety = fleet[(fleet["is_ev"]) & (fleet["include_in_optimisation"])][
        ["vehicle_class", "class_group", "target_count", "sample_count",
         "target_weight", "battery_kWh", "v2g_power_kW"]
    ].sort_values("battery_kWh")
    print(variety.to_string(index=False))
    missing = variety[variety["sample_count"] < 1]["vehicle_class"].tolist()
    if missing:
        print("WARNING: classes with zero samples (raise SAMPLE_TOTAL_EV_FLEET):", missing)
    else:
        print("All EV classes represented, including heavy trucks.")

    print("\nDone. Files written to:", OUTPUT_FOLDER)
    print("Main hourly optimisation file:", hourly_file)
    print("Hourly by vehicle class:", class_hourly_file)
    print("Vehicle events sample-scaled file:", events_file)
    print("emobpy-compatible trip table:", emobpy_file)
    print("Vehicle class summary:", summary_file)
    print("Fleet composition file:", fleet_file)
    print("Metadata:", OUTPUT_FOLDER / "metadata_assumptions_multiclass.json")
    print("\nKey checks")
    print("EV fleet size:", int(hourly["fleet_size"].iloc[0]))
    print("Rows:", len(hourly))
    print("Annual driving energy MWh:", round(hourly["driving_energy_MWh"].sum(), 2))
    print("Battery capacity MWh:", round(hourly["battery_capacity_MWh_total"].iloc[0], 2))
    print("Max charging power MW:", round(hourly["p_charge_max_MW"].max(), 2))
    print("Max V2G discharge power MW:", round(hourly["p_discharge_max_MW"].max(), 2))
    print("emobpy import available:", EMOBPY_AVAILABLE)


if __name__ == "__main__":
    main()

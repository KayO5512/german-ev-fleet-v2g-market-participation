# Mobility Dataset Generator

Generates a synthetic, academic-use DHL-style multiclass EV fleet mobility
dataset for optimization: departure/return times, route distances,
parking/charging availability, battery sizes, charger ratings, SOC
assumptions, and V2G power availability, aggregated to hourly fleet-level
inputs for Pyomo. Author: Kweku Obo Aidoo.

## What it models

Seven EV classes plus one non-EV reference class:

- StreetScooter-type last-mile e-vans (~95% of fleet capacity)
- Bulky-goods e-vans (2-person handling)
- E-trikes and e-bikes (micro-mobility, no V2G)
- Volvo FL Electric 4x2 heavy trucks
- Other heavy-duty electric trucks
- Mercedes-Benz eActros 600 long-haul trucks
- Biogas regional trucks (non-EV, reference only, excluded from optimization)

Fleet counts scale to a 34,000-vehicle target. A stratified minimum-per-class
sampler guarantees every class, including the rare heavy trucks (13 Volvo,
16 other heavy-duty, 30 eActros), appears in the sample and is never rounded
to zero, which a plain proportional sampler would do.

## Method

Follows the emobpy methodology (mobility profile → driving consumption →
grid availability → grid demand) with a custom DHL-style commercial fleet
profile, since standard emobpy examples are private-car oriented. This is
an emobpy-style generator; it does not invoke the emobpy engine directly.
The exported trip table is emobpy-compatible if you want to map it into
native emobpy objects later.

## Setup

1. Place `dtcargo_results.xlsx` in a `data/` subfolder next to this script.
   If missing, the script falls back to transparent synthetic distributions
   and prints a note that it did so.
2. Install dependencies:
   ```
   pip install pandas numpy openpyxl emobpy
   ```
3. Run:
   ```
   python generate_dhl_emobpy_multiclass_fleet.py
   ```

Outputs are written to a local `outputs/` folder (gitignored):

| File | Contents |
|---|---|
| `dhl_mobility_timeseries_1h_2025_synthetic_34000_multiclass.csv` | Main hourly fleet-level optimization input |
| `dhl_mobility_timeseries_1h_2025_by_vehicle_class.csv` | Hourly, split by vehicle class |
| `dhl_vehicle_events_2025_synthetic_sample_scaled_multiclass.csv` | Sampled vehicle-day events |
| `dhl_emobpy_compatible_trip_table_2025_multiclass.csv` | Trip table in emobpy-compatible format |
| `dhl_vehicle_class_summary_multiclass.csv` | Per-class summary statistics |
| `dhl_fleet_composition_multiclass.csv` | Fleet composition table |
| `metadata_assumptions_multiclass.json` | All assumptions and run metadata |

## Confirmed aggregate outputs

- ~888 MWh nameplate virtual battery capacity
- ~248 MW charge headroom
- ~122 MW V2G discharge capacity
- Weekday mean availability: ~23,244 vehicles
- Weekend mean availability: ~29,859 vehicles

## Data note

`dtcargo_results.xlsx` (Balke and Adenaw, 2023, DT-CARGO dataset, NEFTON
project grant 01MV21004A) is not tracked in this repository. It empirically
grounds only the heavy truck classes, roughly 5% of V2G capacity;
StreetScooter vans carrying about 95% of capacity use synthetic
distributions. Correct framing for the report: a representative DHL-style
scenario, not an empirical DHL fleet model.

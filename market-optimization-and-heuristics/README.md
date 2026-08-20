# DHL EV Fleet Market Optimization

This repository contains the reproducible models, heuristics, market data, Excel outputs, figures, and written analysis for a DHL electric-vehicle fleet market-participation study.

## Scope

The project evaluates:

- Perfect-foresight linear-programming optimization for day-ahead (DA), intraday (ID), FCR, and combined market participation
- A 2-hour original greedy intraday heuristic
- An enhanced greedy intraday heuristic with driving-energy forecasting, longer lookahead, SOC reservation, and peak-hour protection
- FCR market participation, policy comparison, availability constraints, and vehicle-class analysis
- Battery degradation, throughput, unmet driving energy, profitability, and computational performance

## Main Scripts

### `fcr_scenario_market_model.py`

Integrated reference model containing fleet data loading, DA/ID/FCR alignment, the Naive EDF baseline, original greedy intraday dispatch, Pyomo optimization scenarios, FCR policy analysis, Excel exports, and comparison figures.

### `dhl_market_optimization_realistic_fcr.py`

Standalone perfect-foresight LP runner for DA, ID, FCR, and Combined scenarios with and without linear battery-degradation costs.

### `enhanced_greedy_vs_id_optimization.py`

Real-data comparison of the enhanced greedy ID heuristic against the normal ID optimization baseline. It produces comparison figures and an Excel summary.

### `enhanced_greedy_intraday_optimization.py`

Strategy-development framework comparing original greedy, enhanced greedy, hybrid rolling control, and perfect-foresight reference results.

### `advanced_fcr_figures_accurate.py`

FCR-specific figure generator using actual optimization results and vehicle-class data.

### `dhl_data_processor.py`

Reusable data-processing and analysis helpers.

### `trucks_hourly_heatmap.py`

Hourly fleet-availability heatmap for FCR and V2G analysis.

## Analysis Documents

- `GREEDY_VS_OPTIMIZATION_ANALYSIS.md`: detailed empirical and literature-based comparison
- `GREEDY_VS_OPTIMIZATION_ANALYSIS.md`: focused greedy-analysis reference

## Excel Files

### `excel_inputs/`

Source workbooks required by the model:

- Fleet mobility and battery data
- Day-ahead prices
- 15-minute intraday IDC prices
- German FCR capacity-market prices

### `excel_model_outputs/`

Representative DA, ID, FCR, Combined, Naive EDF, and original-greedy result workbooks, plus the processed analysis summary.

### `excel_enhanced_greedy/`

Standalone enhanced-greedy strategy comparison and the latest enhanced-greedy-versus-ID workbook.

### `excel_fcr_outputs/`

FCR policy-comparison results with realistic FCR constraints.

The included workbooks provide traceability for the figures and written analysis. Historical duplicate exports are intentionally excluded.

## Figures

`sample_outputs/` contains representative PNG and SVG figures covering:

- `01a` Net profit comparison
- `01b` Cost and revenue decomposition
- `01c` Throughput and operational reliability
- `01d` Performance summary table
- All-strategy profit ranking
- Degradation impact
- Value capture and throughput efficiency
- Trading profiles
- `06a`-`06f` standalone KPI plots for profit, value capture, degradation, charging, discharge revenue, and charging cost
- `07a`-`07d` standalone vehicle-class profitability, reliability, activity, and decision-summary table
- FCR and algorithm-flow visualizations

Composite images with four or more plots are intentionally excluded to reduce cognitive load. Every former four-panel or six-panel figure is represented by individual plot or table files with matching PNG and SVG versions. The two-panel value-capture figure and three-panel trading-profile figure remain compact enough to review as single figures.

## Environment

Recommended environment:

- Python 3.12+
- pandas
- numpy
- matplotlib
- seaborn
- openpyxl
- Pyomo
- HiGHS or the configured Pyomo solver

The scripts were syntax-checked with `python -m py_compile` in the project environment.

## Reproduction

The current scripts contain absolute Windows paths matching the original workspace. To reproduce on another system, update the input and output path constants or place the workbooks at the configured locations.

From the project directory:

```powershell
python fcr_scenario_market_model.py
python dhl_market_optimization_realistic_fcr.py
python enhanced_greedy_vs_id_optimization.py
python advanced_fcr_figures_accurate.py
```

The full-year Pyomo optimization can take substantially longer than the heuristic scripts. The enhanced greedy benchmark is linear in the number of timesteps and is intended for fast online comparison.

## Interpretation

The original greedy heuristic is retained as a diagnostic reference case demonstrating the effects of insufficient foresight and missing driving-energy awareness. The enhanced greedy heuristic is the constraint-aware online approach under evaluation.

Results should distinguish between:

- **ID Optimization**: offline perfect-foresight LP benchmark
- **Original Greedy ID**: short-lookahead diagnostic reference
- **Enhanced Greedy ID**: forecasting- and constraint-aware online heuristic
- **Combined / FCR scenarios**: multi-market and ancillary-service benchmarks

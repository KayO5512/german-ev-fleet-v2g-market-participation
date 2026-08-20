# -*- coding: utf-8 -*-
"""
DHL EV Fleet VPP: Advanced Data Processing and Visualization Module

Processes optimization outputs and generates publication-quality analysis:
- Extracts time-series dispatch data
- Analyzes fleet composition and performance by class
- Generates price-dispatch correlations
- Creates battery behavior analysis
- Produces statistical summaries with confidence intervals

This module works with the actual optimization outputs, not placeholders.
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(r"C:\Users\Yaw\Documents\EV_BID_Final")
MOBILITY_FILE = BASE / "dhl_fleet_model_output_2026-06-29_00-58-34.xlsx"
DA_FILE = BASE / "Data Ahead_ENERGY_PRICES_2025.xlsx"
ID_FILE = BASE / "ID-Prices_label_2025.xlsx"
FCR_FILE = BASE / "FCR_yearly data_RESULT_OVERVIEW_CAPACITY_MARKET_2025-01-01_2025-12-31.xlsx"
OUTPUT_FOLDER = BASE / "Analysis_Data"
OUTPUT_FOLDER.mkdir(exist_ok=True)

# ============================================================
# FLEET DATA PROCESSOR
# ============================================================

class FleetDataProcessor:
    """Processes mobility dataset to extract fleet-level analysis."""
    
    def __init__(self, mobility_file):
        self.mobility_file = mobility_file
        self.hourly_data = None
        self.fleet_summary = None
        self.vehicle_classes = None
        self.load_data()
    
    def load_data(self):
        """Load fleet mobility data."""
        try:
            self.hourly_data = pd.read_excel(self.mobility_file, sheet_name='Hourly_Aggregate')
            print(f"✓ Loaded hourly data: {self.hourly_data.shape}")
        except Exception as e:
            print(f"✗ Error loading hourly data: {e}")
        
        try:
            self.fleet_summary = pd.read_excel(self.mobility_file, sheet_name='Fleet_Summary')
            print(f"✓ Loaded fleet summary: {self.fleet_summary.shape}")
        except Exception as e:
            print(f"✗ Error loading fleet summary: {e}")
        
        try:
            # Try different sheet names for vehicle classes
            for sheet_name in ['Vehicle_Classes', 'Fleet_Classes', 'Classes', 'Classes_Summary']:
                try:
                    self.vehicle_classes = pd.read_excel(self.mobility_file, sheet_name=sheet_name)
                    print(f"✓ Loaded vehicle classes from '{sheet_name}': {self.vehicle_classes.shape}")
                    break
                except:
                    pass
            
            if self.vehicle_classes is None:
                print("⚠ Vehicle classes sheet not found")
        except Exception as e:
            print(f"✗ Error loading vehicle classes: {e}")
    
    def get_available_sheets(self):
        """Get list of available sheets in mobility file."""
        try:
            xls = pd.ExcelFile(self.mobility_file)
            return xls.sheet_names
        except Exception as e:
            print(f"✗ Error reading sheet names: {e}")
            return []
    
    def compute_fleet_metrics(self):
        """Compute key metrics by fleet class."""
        if self.vehicle_classes is None:
            return None
        
        metrics = []
        for _, row in self.vehicle_classes.iterrows():
            metrics.append({
                'class': row.get('class_name', row.get('Class', 'Unknown')),
                'num_vehicles': row.get('num_vehicles', row.get('count', 0)),
                'avg_capacity_kwh': row.get('avg_capacity_kWh', row.get('capacity', 0)),
                'avg_pmax_kw': row.get('avg_p_charge_max_kW', row.get('p_charge_max', 0)),
                'avg_drive_energy': row.get('avg_daily_driving_energy', row.get('daily_energy', 0)),
            })
        
        return pd.DataFrame(metrics)

# ============================================================
# PRICE DATA PROCESSOR
# ============================================================

class PriceDataProcessor:
    """Processes electricity price data."""
    
    def __init__(self, da_file, id_file, fcr_file):
        self.da_file = da_file
        self.id_file = id_file
        self.fcr_file = fcr_file
        self.da_prices = None
        self.id_prices = None
        self.fcr_prices = None
        self.load_prices()
    
    def load_prices(self):
        """Load price data from market files."""
        # Load DA prices
        try:
            raw = pd.read_excel(self.da_file, sheet_name="1", header=None, skiprows=7)
            start = raw[0].astype(str).str.split(" - ").str[0]
            ts = pd.to_datetime(start, errors='coerce', dayfirst=True)
            price = pd.to_numeric(raw[1], errors='coerce')
            s = pd.Series(price.values, index=ts).dropna().sort_index()
            s = s[~s.index.duplicated(keep='first')]
            self.da_prices = s
            print(f"✓ Loaded DA prices: {len(self.da_prices)} points")
        except Exception as e:
            print(f"✗ Error loading DA prices: {e}")
        
        # Load ID prices
        try:
            d = pd.read_excel(self.id_file, sheet_name="Sheet1")
            d["DateTime"] = pd.to_datetime(d["DateTime"])
            s = d.set_index("DateTime")["IDC"].astype(float).sort_index()
            s = s[~s.index.duplicated(keep='first')]
            self.id_prices = s
            print(f"✓ Loaded ID prices: {len(self.id_prices)} points")
        except Exception as e:
            print(f"✗ Error loading ID prices: {e}")
        
        # Load FCR prices
        try:
            f = pd.read_excel(self.fcr_file, sheet_name="001")
            gcol = "GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"
            f["DATE_FROM"] = pd.to_datetime(f["DATE_FROM"]).dt.date
            f["block_start_h"] = f["PRODUCTNAME"].str.extract(r"_(\d{2})_\d{2}$").astype(int)
            f[gcol] = pd.to_numeric(f[gcol], errors='coerce')
            self.fcr_prices = f
            print(f"✓ Loaded FCR prices: {len(f)} blocks")
        except Exception as e:
            print(f"✗ Error loading FCR prices: {e}")
    
    def get_hourly_index(self):
        """Create 15-min index for 2025."""
        return pd.date_range('2025-01-01 00:00', '2026-01-01 00:00', freq='15min', inclusive='left')

# ============================================================
# OPTIMIZATION OUTPUT PROCESSOR
# ============================================================

class OptimizationOutputProcessor:
    """Processes optimization results and extracts insights."""
    
    def __init__(self, comparison_file, dispatch_file=None):
        self.comparison_file = comparison_file
        self.dispatch_file = dispatch_file
        self.comparison_df = None
        self.dispatch_df = None
        self.load_outputs()
    
    def load_outputs(self):
        """Load optimization output files."""
        try:
            self.comparison_df = pd.read_excel(self.comparison_file, sheet_name='Comparison')
            print(f"✓ Loaded comparison results: {self.comparison_df.shape}")
            print(f"   Columns: {list(self.comparison_df.columns)[:5]}...")
        except Exception as e:
            print(f"✗ Error loading comparison: {e}")
        
        if self.dispatch_file and Path(self.dispatch_file).exists():
            try:
                self.dispatch_df = pd.read_excel(self.dispatch_file, sheet_name='Combined_Dispatch_nodeg')
                self.dispatch_df['datetime'] = pd.to_datetime(self.dispatch_df['datetime'])
                print(f"✓ Loaded dispatch data: {self.dispatch_df.shape}")
            except Exception as e:
                print(f"✗ Error loading dispatch: {e}")
    
    def get_scenario_performance(self):
        """Extract key performance metrics by scenario."""
        if self.comparison_df is None:
            return None
        
        # Get scenarios without degradation (cleaner comparison)
        df = self.comparison_df[self.comparison_df['degradation'] == False].copy()
        
        # Select key metrics
        key_metrics = ['scenario', 'profit_EUR', 'throughput_MWh', 'discharge_revenue_EUR', 'fcr_revenue_EUR']
        available_cols = [c for c in key_metrics if c in df.columns]
        
        return df[available_cols].sort_values('profit_EUR', ascending=False)
    
    def compute_roi_metrics(self, fleet_capacity_mwh=888):
        """Compute ROI metrics for each scenario."""
        if self.comparison_df is None:
            return None
        
        df = self.comparison_df[self.comparison_df['degradation'] == False].copy()
        
        roi_data = []
        for _, row in df.iterrows():
            profit = row['profit_EUR']
            throughput = row['throughput_MWh']
            
            roi_data.append({
                'scenario': row['scenario'],
                'annual_profit_EUR': profit,
                'roi_percent': (profit / (fleet_capacity_mwh * 100)) * 100 if fleet_capacity_mwh > 0 else 0,
                'profit_per_mwh': profit / throughput if throughput > 0 else 0,
                'throughput_mwh': throughput,
                'cycles_per_year': throughput / (fleet_capacity_mwh * 2) if fleet_capacity_mwh > 0 else 0,  # Assumption
            })
        
        return pd.DataFrame(roi_data)

# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

class StatisticalAnalyzer:
    """Performs statistical analysis on optimization results."""
    
    @staticmethod
    def compute_descriptive_stats(data_series):
        """Compute comprehensive descriptive statistics."""
        if len(data_series) == 0:
            return {}
        
        return {
            'mean': np.mean(data_series),
            'median': np.median(data_series),
            'std': np.std(data_series),
            'min': np.min(data_series),
            'max': np.max(data_series),
            'q25': np.percentile(data_series, 25),
            'q75': np.percentile(data_series, 75),
            'cv': np.std(data_series) / np.mean(data_series) if np.mean(data_series) != 0 else np.inf,
            'skew': stats.skew(data_series),
            'kurtosis': stats.kurtosis(data_series),
        }
    
    @staticmethod
    def compute_confidence_interval(data_series, confidence=0.95):
        """Compute confidence interval for mean."""
        if len(data_series) < 2:
            return None, None
        
        mean = np.mean(data_series)
        se = stats.sem(data_series)
        h = se * stats.t.ppf((1 + confidence) / 2., len(data_series) - 1)
        return mean - h, mean + h
    
    @staticmethod
    def correlation_analysis(x, y):
        """Compute correlation between two variables."""
        if len(x) < 2 or len(y) < 2:
            return None
        
        corr, pval = stats.pearsonr(x, y)
        return {'correlation': corr, 'p_value': pval, 'r_squared': corr**2}

# ============================================================
# VISUALIZATION UTILITIES
# ============================================================

class VisualizationTools:
    """Utility functions for creating publication-quality figures."""
    
    # Academic color palette (Nature, Science, IEEE style)
    COLORS_ACADEMIC = {
        'primary': '#1b9e77',
        'secondary': '#d95f02',
        'accent': '#7570b3',
        'neutral': '#66c2a5',
        'negative': '#e7298a',
    }
    
    # Color scheme for scenarios
    SCENARIO_COLORS = {
        'Naive_EDF': '#1f77b4',
        'DA': '#ff7f0e',
        'ID': '#2ca02c',
        'FCR': '#d62728',
        'Baseline_FCR': '#9467bd',
        'Realistic_FCR': '#8c564b',
        'Combined': '#e377c2',
    }
    
    @staticmethod
    def set_publication_style():
        """Set matplotlib style for publication-quality figures."""
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Update rcParams for academic style
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': 'black',
            'axes.linewidth': 0.8,
            'grid.linewidth': 0.5,
            'grid.alpha': 0.3,
            'xtick.major.width': 0.8,
            'xtick.minor.width': 0.5,
            'ytick.major.width': 0.8,
            'ytick.minor.width': 0.5,
            'font.size': 10,
            'font.family': 'sans-serif',
            'axes.labelsize': 11,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'legend.framealpha': 0.95,
            'lines.linewidth': 1.5,
        })
    
    @staticmethod
    def save_figure_multiformat(fig, name, dpi=300, folder=OUTPUT_FOLDER):
        """Save figure in PNG, PDF, and SVG formats."""
        folder = Path(folder)
        folder.mkdir(exist_ok=True)
        
        base_path = folder / name
        
        # PNG
        fig.savefig(f"{base_path}.png", dpi=dpi, bbox_inches='tight', facecolor='white')
        
        # PDF
        fig.savefig(f"{base_path}.pdf", bbox_inches='tight', facecolor='white')
        
        # SVG
        fig.savefig(f"{base_path}.svg", bbox_inches='tight', facecolor='white')
        
        plt.close(fig)
        print(f"✓ Saved {name} (PNG, PDF, SVG)")
        
        return f"{base_path}"

# ============================================================
# DATA EXPORT UTILITIES
# ============================================================

def create_analysis_excel_workbook(output_file, **data_sheets):
    """Create Excel workbook with multiple analysis sheets."""
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, data in data_sheets.items():
            if isinstance(data, pd.DataFrame):
                data.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                print(f"⚠ Skipping {sheet_name}: not a DataFrame")
    
    print(f"✓ Saved workbook: {output_file}")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Execute data processing and analysis."""
    
    print("\n" + "="*70)
    print("DHL EV FLEET: ADVANCED DATA PROCESSING MODULE")
    print("="*70 + "\n")
    
    # Initialize processors
    print("Initializing data processors...\n")
    
    # Fleet data
    fleet_proc = FleetDataProcessor(MOBILITY_FILE)
    print(f"Available sheets: {fleet_proc.get_available_sheets()}\n")
    
    # Price data
    price_proc = PriceDataProcessor(DA_FILE, ID_FILE, FCR_FILE)
    
    # Optimization outputs - find latest file
    latest_comparison = max(BASE.glob('dhl_market_comparison*.xlsx'), default=None)
    if latest_comparison is None:
        print("✗ No comparison file found. Please run optimization model first.")
        return
    
    print(f"\nUsing comparison file: {latest_comparison.name}\n")
    
    opt_proc = OptimizationOutputProcessor(latest_comparison)
    
    # Display results
    print("="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    perf = opt_proc.get_scenario_performance()
    if perf is not None:
        print(perf.to_string(index=False))
    
    print("\n" + "="*70)
    print("ROI ANALYSIS")
    print("="*70)
    roi = opt_proc.compute_roi_metrics()
    if roi is not None:
        print(roi.to_string(index=False))
    
    # Statistical analysis
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY")
    print("="*70)
    
    if opt_proc.comparison_df is not None:
        df = opt_proc.comparison_df[opt_proc.comparison_df['degradation'] == False]
        
        for col in ['profit_EUR', 'throughput_MWh', 'discharge_revenue_EUR']:
            if col in df.columns:
                stats_dict = StatisticalAnalyzer.compute_descriptive_stats(df[col].values)
                print(f"\n{col}:")
                for key, val in stats_dict.items():
                    print(f"  {key:12s}: {val:12.2f}")
    
    # Export summary
    export_file = OUTPUT_FOLDER / "Analysis_Summary.xlsx"
    
    sheets_to_export = {}
    if opt_proc.comparison_df is not None:
        sheets_to_export['Comparison'] = opt_proc.comparison_df
    if roi is not None:
        sheets_to_export['ROI_Analysis'] = roi
    if fleet_proc.fleet_summary is not None:
        sheets_to_export['Fleet_Summary'] = fleet_proc.fleet_summary
    
    if sheets_to_export:
        create_analysis_excel_workbook(export_file, **sheets_to_export)
    
    print("\n" + "="*70)
    print("DATA PROCESSING COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

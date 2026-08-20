# -*- coding: utf-8 -*-
"""
Advanced FCR Analysis: Accurate Figures with Real Data
Using actual vehicle class data from Class_Summary sheet
Baseline FCR revenue: €7.94M (actual optimization output)
Realistic FCR revenue: Applied German market constraints
Blue color scheme for PowerPoint integration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Colors: Blue-family palette for PowerPoint integration
COLOR_PRIMARY_BLUE = '#0052CC'      # Strong primary blue
COLOR_LIGHT_BLUE = '#4A90E2'        # Light blue for secondary
COLOR_DARK_BLUE = '#003D99'         # Dark blue for emphasis
COLOR_ACCENT_ORANGE = '#FF6B35'     # Orange for contrast (negative values)
COLOR_ACCENT_TEAL = '#00897B'       # Teal for tertiary dimension
COLOR_ACCENT_PURPLE = '#6C5CE7'     # Purple-blue for diversity

BASE = Path(r"C:\Users\Yaw\Documents\EV_BID_Final")
OUTPUT_FOLDER = BASE / "Advanced_FCR_Figures"
OUTPUT_FOLDER.mkdir(exist_ok=True)

# ==============================================================================
# 1. LOAD ACTUAL DATA
# ==============================================================================

def load_actual_fcr_data():
    """Load ACTUAL FCR revenue from optimization output and class summary from mobility data."""
    
    # Load optimization output - THIS IS ACTUAL, NOT ESTIMATED
    # Use the file with detailed FCR breakdown (gross/net/constraints)
    latest_file = BASE / "dhl_market_comparison_2026-06-29_16-15-58.xlsx"
    
    if not latest_file.exists():
        # Fallback to most recent file
        import glob
        comparison_files = sorted(glob.glob(str(BASE / "dhl_market_comparison_*.xlsx")))
        latest_file = Path(comparison_files[-1])
    
    opt_df = pd.read_excel(latest_file, sheet_name="Comparison")
    
    # Extract baseline FCR (no degradation) - ACTUAL NUMBERS FROM OPTIMIZATION
    fcr_baseline_row = opt_df[(opt_df['scenario'] == 'FCR') & (opt_df['degradation'] == False)].iloc[0]
    fcr_gross_baseline = fcr_baseline_row['fcr_gross_revenue_EUR']
    fcr_net_realistic = fcr_baseline_row['fcr_revenue_EUR']
    fcr_acceptance_loss = fcr_baseline_row.get('fcr_acceptance_haircut_EUR', 0)
    fcr_aggregator_fee = fcr_baseline_row.get('fcr_aggregator_fee_EUR', 0)
    fcr_activation_cost = fcr_baseline_row.get('fcr_activation_cost_EUR', 0)
    fcr_ramp_penalty = fcr_baseline_row.get('ramp_delay_penalty_EUR', 0)
    
    print(f"✓ Baseline FCR gross revenue: €{fcr_gross_baseline:,.0f}")
    print(f"✓ Realistic FCR net revenue: €{fcr_net_realistic:,.0f}")
    print(f"✓ Acceptance haircut loss: €{fcr_acceptance_loss:,.0f}")
    print(f"✓ Aggregator fee loss: €{fcr_aggregator_fee:,.0f}")
    print(f"✓ Activation cost: €{fcr_activation_cost:,.0f}")
    print(f"✓ Ramp delay penalty: €{fcr_ramp_penalty:,.0f}")
    
    # Load mobility data - CLASS SUMMARY
    mobility_wb = BASE / "dhl_fleet_model_output_2026-06-29_00-58-34.xlsx"
    class_summary = pd.read_excel(mobility_wb, sheet_name="Class_Summary")
    
    # Filter to V2G-capable classes only
    v2g_classes = class_summary[class_summary['v2g_power_kW'] > 0].copy()
    
    return {
        'fcr_gross_baseline_eur': fcr_gross_baseline,
        'fcr_net_realistic_eur': fcr_net_realistic,
        'fcr_acceptance_loss_eur': fcr_acceptance_loss,
        'fcr_aggregator_fee_eur': fcr_aggregator_fee,
        'fcr_activation_cost_eur': fcr_activation_cost,
        'fcr_ramp_penalty_eur': fcr_ramp_penalty,
        'class_summary': v2g_classes,
    }

def calculate_realistic_fcr(baseline_gross, acceptance_loss, aggregator_fee, activation_cost, ramp_penalty):
    """Calculate realistic FCR with actual constraint values from optimization output."""
    
    # Use ACTUAL values from optimization (not estimates)
    fcr_realistic_net = baseline_gross - acceptance_loss - aggregator_fee - activation_cost - ramp_penalty
    
    # Waterfall values
    loss_acceptance = acceptance_loss
    loss_aggregator_fee = aggregator_fee
    loss_activation = activation_cost
    loss_ramp_penalty = ramp_penalty
    total_loss = baseline_gross - fcr_realistic_net
    
    return {
        'baseline': baseline_gross,
        'realistic': max(fcr_realistic_net, 0),
        'loss_acceptance': loss_acceptance,
        'loss_aggregator_fee': loss_aggregator_fee,
        'loss_activation': loss_activation,
        'loss_ramp_penalty': loss_ramp_penalty,
        'total_loss': total_loss,
        'pct_reduction': (total_loss / baseline_gross) * 100 if baseline_gross > 0 else 0,
    }

# ==============================================================================
# 2. FIGURE 1: WATERFALL - BASELINE TO REALISTIC
# ==============================================================================

def figure1_waterfall(data):
    """Waterfall chart showing revenue reduction from baseline to realistic FCR with ACTUAL values."""
    
    baseline = data['fcr_gross_baseline_eur']
    realistic_net = data['fcr_net_realistic_eur']
    loss_acceptance = data['fcr_acceptance_loss_eur']
    loss_aggregator_fee = data['fcr_aggregator_fee_eur']
    loss_activation = data['fcr_activation_cost_eur']
    loss_ramp_penalty = data['fcr_ramp_penalty_eur']
    total_loss = baseline - realistic_net
    pct_reduction = (total_loss / baseline) * 100 if baseline > 0 else 0
    
    realistic_calc = {
        'baseline': baseline,
        'realistic': realistic_net,
        'loss_acceptance': loss_acceptance,
        'loss_aggregator_fee': loss_aggregator_fee,
        'loss_activation': loss_activation,
        'loss_ramp_penalty': loss_ramp_penalty,
        'total_loss': total_loss,
        'pct_reduction': pct_reduction,
    }
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Waterfall data
    categories = ['Baseline\nGross FCR', 'Auction\nAcceptance\nHaircut', 
                  'Aggregator/\nBSP Fee', 'Activation\nCost', 
                  'Ramp Delay\nPenalty', 'Realistic\nNet FCR']
    
    values = [
        baseline,
        -realistic_calc['loss_acceptance'],
        -realistic_calc['loss_aggregator_fee'],
        -realistic_calc['loss_activation'],
        -realistic_calc['loss_ramp_penalty'],
        realistic_calc['realistic']
    ]
    
    # Cumulative for positioning
    cumulative = [baseline]
    running = baseline
    for i in range(1, len(values) - 1):
        running += values[i]
        cumulative.append(running)
    cumulative.append(realistic_calc['realistic'])
    
    # Colors: Blue for positive/baseline, Orange for losses, Blue for result
    colors = [COLOR_PRIMARY_BLUE, COLOR_ACCENT_ORANGE, COLOR_ACCENT_ORANGE, 
              COLOR_ACCENT_ORANGE, COLOR_ACCENT_ORANGE, COLOR_LIGHT_BLUE]
    
    # Plot bars
    x = np.arange(len(categories))
    for i, (cat, val, cum, col) in enumerate(zip(categories, values, cumulative, colors)):
        if i == 0:
            ax.bar(i, val, color=col, edgecolor='black', linewidth=1.5, width=0.6)
        elif i == len(categories) - 1:
            ax.bar(i, val, color=col, edgecolor='black', linewidth=1.5, width=0.6)
        else:
            # Floating bar for intermediate steps
            bottom = cumulative[i-1] + val
            ax.bar(i, val, bottom=bottom if val < 0 else cumulative[i-1], 
                   color=col, edgecolor='black', linewidth=1.5, width=0.6)
    
    # Connector lines
    for i in range(len(categories) - 1):
        if i < len(categories) - 2:
            ax.plot([i + 0.3, i + 0.7], [cumulative[i+1], cumulative[i+1]], 
                   'k--', linewidth=1, alpha=0.7)
    
    # Labels and formatting
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylabel('Annual Revenue (EUR millions)', fontsize=12, fontweight='bold')
    ax.set_title('Baseline vs Realistic FCR Revenue: Constraint Impact Waterfall (ACTUAL DATA)\n' + 
                 f'Realistic constraints reduce FCR by {realistic_calc["pct_reduction"]:.0f}%', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x/1e6:.1f}M'))
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (val, cum) in enumerate(zip(values, cumulative)):
        if i == 0 or i == len(values) - 1:
            label_y = cum + baseline * 0.02
            ax.text(i, label_y, f'€{abs(cum)/1e6:.2f}M', ha='center', va='bottom', 
                   fontweight='bold', fontsize=10, color=COLOR_PRIMARY_BLUE)
        else:
            label_y = cumulative[i] + baseline * 0.02
            ax.text(i, label_y, f'-€{abs(val)/1e6:.2f}M', ha='center', va='bottom',
                   fontweight='bold', fontsize=9, color=COLOR_ACCENT_ORANGE)
    
    # Legend
    baseline_patch = mpatches.Patch(color=COLOR_PRIMARY_BLUE, label='Baseline / Result')
    loss_patch = mpatches.Patch(color=COLOR_ACCENT_ORANGE, label='Market Constraint Loss')
    ax.legend(handles=[baseline_patch, loss_patch], loc='upper right', fontsize=11)
    
    plt.tight_layout()
    
    # Save
    png_path = OUTPUT_FOLDER / "Fig_01_FCR_Waterfall_Accurate.png"
    svg_path = OUTPUT_FOLDER / "Fig_01_FCR_Waterfall_Accurate.svg"
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(svg_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Fig_01_FCR_Waterfall_Accurate (PNG 300dpi + SVG)")
    
    return realistic_calc
    
    return realistic_calc

# ==============================================================================
# 3. FIGURE 2: VEHICLE CLASS PROFITABILITY
# ==============================================================================

def figure2_class_profitability(class_summary, fcr_baseline_gross):
    """Bar chart comparing V2G-capable vehicle classes on profitability."""
    
    # Extract relevant classes
    classes_data = []
    for _, row in class_summary.iterrows():
        vehicle_class = row['vehicle_class'].replace('_', ' ').title()
        if vehicle_class == 'Mercedes Benz Eactros 600 Long Haul':
            vehicle_class = 'eActros 600 (Long-haul)'
        elif vehicle_class == 'Volvo Fl Electric 4x2 Berlin':
            vehicle_class = 'Volvo FL (Heavy-truck)'
        elif vehicle_class == 'Other Heavy Duty E Truck':
            vehicle_class = 'Heavy-duty e-Truck'
        elif vehicle_class == 'Large Bulky Goods E Van 2mh':
            vehicle_class = 'Bulky-goods Van'
        elif vehicle_class == 'Streetscooter Last Mile E Van':
            vehicle_class = 'StreetScooter (Last-mile)'
        
        capacity_mwh = row['total_battery_capacity_MWh']
        v2g_power_kw = row['total_nameplate_v2g_capacity_MW'] * 1000
        v2g_power_mw = row['total_nameplate_v2g_capacity_MW']
        n_vehicles = row['target_count']
        
        # Allocate profitability: Baseline FCR revenue allocated by V2G capacity share
        total_v2g_capacity = class_summary['total_nameplate_v2g_capacity_MW'].sum()
        fcr_revenue_estimate = fcr_baseline_gross * (v2g_power_mw / total_v2g_capacity)
        revenue_per_mwh = fcr_revenue_estimate / capacity_mwh if capacity_mwh > 0 else 0
        
        classes_data.append({
            'class': vehicle_class,
            'capacity_mwh': capacity_mwh,
            'v2g_power_mw': v2g_power_mw,
            'n_vehicles': n_vehicles,
            'fcr_revenue_eur': fcr_revenue_estimate,
            'revenue_per_mwh': revenue_per_mwh,
        })
    
    df_class = pd.DataFrame(classes_data).sort_values('fcr_revenue_eur', ascending=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Panel A: FCR Revenue by Class
    y_pos = np.arange(len(df_class))
    colors_a = [COLOR_PRIMARY_BLUE if i < 4 else COLOR_LIGHT_BLUE for i in range(len(df_class))]
    
    ax1.barh(y_pos, df_class['fcr_revenue_eur'] / 1e6, color=colors_a, edgecolor='black', linewidth=1.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_class['class'], fontsize=11, fontweight='bold')
    ax1.set_xlabel('Estimated FCR Revenue (EUR millions)', fontsize=12, fontweight='bold')
    ax1.set_title('Panel A: FCR Revenue by Vehicle Class\n(Baseline scenario)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3, linestyle=':')
    ax1.set_axisbelow(True)
    
    for i, (idx, row) in enumerate(df_class.iterrows()):
        ax1.text(row['fcr_revenue_eur']/1e6 + 0.1, i, f"€{row['fcr_revenue_eur']/1e6:.2f}M",
                va='center', fontweight='bold', fontsize=10)
    
    # Panel B: Available V2G Capacity
    colors_b = [COLOR_ACCENT_TEAL if i < 4 else COLOR_LIGHT_BLUE for i in range(len(df_class))]
    
    ax2.barh(y_pos, df_class['v2g_power_mw'], color=colors_b, edgecolor='black', linewidth=1.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(['' for _ in range(len(df_class))], fontsize=11)
    ax2.set_xlabel('Available V2G Capacity (MW)', fontsize=12, fontweight='bold')
    ax2.set_title('Panel B: Available V2G Capacity\n(for FCR participation)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3, linestyle=':')
    ax2.set_axisbelow(True)
    
    for i, (idx, row) in enumerate(df_class.iterrows()):
        ax2.text(row['v2g_power_mw'] + 0.2, i, f"{row['v2g_power_mw']:.2f} MW\n({int(row['n_vehicles'])} veh)",
                va='center', fontweight='bold', fontsize=9)
    
    fig.suptitle('V2G-Capable Vehicle Classes: Profitability and Capacity Ranking', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save
    png_path = OUTPUT_FOLDER / "Fig_02_Class_Profitability_Accurate.png"
    svg_path = OUTPUT_FOLDER / "Fig_02_Class_Profitability_Accurate.svg"
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(svg_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Fig_02_Class_Profitability_Accurate (PNG 300dpi + SVG)")
    
    return df_class

# ==============================================================================
# 4. FIGURE 3: HOURLY AVAILABILITY HEATMAP BY CLASS
# ==============================================================================

def figure3_hourly_heatmap():
    """Heatmap showing hourly V2G availability by vehicle class."""
    
    # Load hourly by class data
    mobility_wb = BASE / "dhl_fleet_model_output_2026-06-29_00-58-34.xlsx"
    hourly_class = pd.read_excel(mobility_wb, sheet_name="Hourly_By_Class")
    
    # Filter to V2G-capable classes only
    v2g_classes = ['streetscooter_last_mile_e_van', 'volvo_fl_electric_4x2_berlin', 
                   'other_heavy_duty_e_truck', 'mercedes_benz_eactros_600_long_haul',
                   'large_bulky_goods_e_van_2mh']
    
    hourly_v2g = hourly_class[hourly_class['vehicle_class'].isin(v2g_classes)].copy()
    
    # Map class names to shorter labels
    class_labels = {
        'streetscooter_last_mile_e_van': 'StreetScooter\n(Last-mile)',
        'volvo_fl_electric_4x2_berlin': 'Volvo FL\n(Heavy-truck)',
        'other_heavy_duty_e_truck': 'Heavy-duty\ne-Truck',
        'mercedes_benz_eactros_600_long_haul': 'eActros 600\n(Long-haul)',
        'large_bulky_goods_e_van_2mh': 'Bulky-goods\nVan',
    }
    
    hourly_v2g['class_short'] = hourly_v2g['vehicle_class'].map(class_labels)
    
    # Aggregate by hour and class (average across all days)
    hourly_agg = hourly_v2g.groupby(['hour', 'class_short'])['p_discharge_max_MW'].mean().reset_index()
    pivot_data = hourly_agg.pivot(index='class_short', columns='hour', values='p_discharge_max_MW')
    
    # Reorder to V2G classes order
    pivot_data = pivot_data.reindex([class_labels[c] for c in v2g_classes if c in class_labels])
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create heatmap
    im = ax.imshow(pivot_data.values, cmap='Blues', aspect='auto', interpolation='nearest')
    
    # Axes
    ax.set_xticks(np.arange(24))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], fontsize=10, rotation=45)
    ax.set_yticks(np.arange(len(pivot_data)))
    ax.set_yticklabels(pivot_data.index, fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Hour of Day (UTC)', fontsize=12, fontweight='bold')
    ax.set_title('Hourly V2G Discharge Availability by Vehicle Class\nDark blue = High availability for FCR participation', 
                fontsize=13, fontweight='bold', pad=15)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Available Discharge Power (MW)')
    cbar.set_label('Available Discharge Power (MW)', fontsize=11, fontweight='bold')
    
    # Add grid
    ax.set_xticks(np.arange(24) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(pivot_data)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    
    # Add text annotations (only for significant values)
    for i in range(len(pivot_data)):
        for j in range(24):
            val = pivot_data.values[i, j]
            if val > 1:  # Only label if > 1 MW
                text_color = 'white' if val > pivot_data.values.max() * 0.6 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                       color=text_color, fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    png_path = OUTPUT_FOLDER / "Fig_03_Hourly_Heatmap_Accurate.png"
    svg_path = OUTPUT_FOLDER / "Fig_03_Hourly_Heatmap_Accurate.svg"
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(svg_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Fig_03_Hourly_Heatmap_Accurate (PNG 300dpi + SVG)")

# ==============================================================================
# 5. FIGURE 4: REALISTIC vs BASELINE FCR COMPARISON
# ==============================================================================

def figure4_baseline_vs_realistic(realistic_calc, baseline_fcr):
    """Side-by-side comparison of baseline vs realistic FCR with all constraints visible."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    scenarios = ['Baseline\n(Ideal)', 'Realistic\n(German Market)']
    revenues = [realistic_calc['baseline'] / 1e6, realistic_calc['realistic'] / 1e6]
    colors_compare = [COLOR_PRIMARY_BLUE, COLOR_LIGHT_BLUE]
    
    x = np.arange(len(scenarios))
    bars = ax.bar(x, revenues, color=colors_compare, edgecolor='black', linewidth=2, width=0.5)
    
    # Add value labels
    for i, (bar, rev) in enumerate(zip(bars, revenues)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
               f'€{rev:.2f}M', ha='center', va='bottom', fontweight='bold', fontsize=13)
    
    # Add constraint breakdown for realistic
    constraint_text = (
        f"Constraints Applied:\n"
        f"• Auction acceptance: -{realistic_calc['loss_acceptance']/1e6:.2f}M ({realistic_calc['loss_acceptance']/realistic_calc['baseline']*100:.1f}%)\n"
        f"• Aggregator fee: -{realistic_calc['loss_aggregator_fee']/1e6:.2f}M\n"
        f"• Activation cost: -{realistic_calc['loss_activation']/1e6:.2f}M\n"
        f"• Ramp delay penalty: -{realistic_calc['loss_ramp_penalty']/1e6:.2f}M (LARGEST)\n"
        f"──────────────────────\n"
        f"Total reduction: -{realistic_calc['total_loss']/1e6:.2f}M (-{realistic_calc['pct_reduction']:.0f}%)"
    )
    
    ax.text(1, realistic_calc['realistic']/1e6 * 0.3, constraint_text, 
           ha='left', va='top', fontsize=10, family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=12, fontweight='bold')
    ax.set_ylabel('Annual FCR Revenue (EUR millions)', fontsize=12, fontweight='bold')
    ax.set_title('Baseline vs Realistic FCR: Market Constraint Impact\n' +
                f'German FCR markets reduce revenue by {realistic_calc["pct_reduction"]:.0f}% from ideal case',
                fontsize=14, fontweight='bold', pad=20)
    
    ax.set_ylim(0, realistic_calc['baseline'] / 1e6 * 1.2)
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
    
    # Add note about profitability
    ax.text(0.5, -0.15, 'Note: Even realistic FCR (€' + f'{realistic_calc["realistic"]/1e6:.2f}M) remains economically viable when combined with DA+ID arbitrage',
           ha='center', va='top', transform=ax.transAxes, fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    png_path = OUTPUT_FOLDER / "Fig_04_Baseline_vs_Realistic_Accurate.png"
    svg_path = OUTPUT_FOLDER / "Fig_04_Baseline_vs_Realistic_Accurate.svg"
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(svg_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Fig_04_Baseline_vs_Realistic_Accurate (PNG 300dpi + SVG)")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("="*80)
    print("ADVANCED FCR ANALYSIS - ACCURATE FIGURES WITH REAL DATA")
    print("="*80)
    print()
    
    # Load data
    data = load_actual_fcr_data()
    class_summary = data['class_summary']
    baseline_fcr = data['fcr_gross_baseline_eur']
    
    print(f"\nAggregated V2G capacity: {class_summary['total_nameplate_v2g_capacity_MW'].sum():.2f} MW")
    print(f"V2G-capable vehicles: {class_summary['target_count'].sum():.0f}")
    print()
    
    # Generate realistic FCR calculation using ACTUAL values
    realistic_calc = calculate_realistic_fcr(
        baseline_fcr,
        data['fcr_acceptance_loss_eur'],
        data['fcr_aggregator_fee_eur'],
        data['fcr_activation_cost_eur'],
        data['fcr_ramp_penalty_eur'],
    )
    print(f"\nRealistic FCR Calculation (ACTUAL from optimization):")
    print(f"  Baseline gross: €{realistic_calc['baseline']/1e6:.2f}M")
    print(f"  Realistic net: €{realistic_calc['realistic']/1e6:.2f}M")
    print(f"  Total loss: €{realistic_calc['total_loss']/1e6:.2f}M ({realistic_calc['pct_reduction']:.0f}%)")
    print()
    
    # Create figures
    print("Generating figures...")
    print()
    figure1_waterfall(data)
    df_class = figure2_class_profitability(class_summary, baseline_fcr)
    figure3_hourly_heatmap()
    figure4_baseline_vs_realistic(realistic_calc, baseline_fcr)
    
    print()
    print("="*80)
    print("✓ FIGURE GENERATION COMPLETE")
    print("="*80)
    print(f"\nOutput folder: {OUTPUT_FOLDER}")
    print("Formats: PNG (300 dpi) + SVG (editable)")
    print("\nColor scheme: Blue-family palette")
    print(f"  Primary: {COLOR_PRIMARY_BLUE}")
    print(f"  Light: {COLOR_LIGHT_BLUE}")
    print(f"  Dark: {COLOR_DARK_BLUE}")
    print(f"  Accent (Orange): {COLOR_ACCENT_ORANGE}")
    print(f"  Accent (Teal): {COLOR_ACCENT_TEAL}")
    print(f"  Accent (Purple): {COLOR_ACCENT_PURPLE}")

if __name__ == "__main__":
    main()

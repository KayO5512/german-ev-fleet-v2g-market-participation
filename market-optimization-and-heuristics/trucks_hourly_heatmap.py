# -*- coding: utf-8 -*-
"""
Hourly Heatmap: Trucks-Only Fleet
Aggregates all truck classes (Volvo FL, Heavy-duty e-Truck, eActros 600, Bulky-goods Van)
Shows daily discharge availability pattern for trucks as unified V2G resource
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(r"C:\Users\Yaw\Documents\EV_BID_Final")
OUTPUT_FOLDER = BASE / "Advanced_FCR_Figures"
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Blue color palette
COLOR_PRIMARY_BLUE = '#0052CC'
COLOR_LIGHT_BLUE = '#4A90E2'
COLOR_DARK_BLUE = '#003D99'

# Load hourly by class data
mobility_wb = BASE / "dhl_fleet_model_output_2026-06-29_00-58-34.xlsx"
hourly_class = pd.read_excel(mobility_wb, sheet_name="Hourly_By_Class")

# Define truck classes (all V2G-capable heavy vehicles)
truck_classes = [
    'volvo_fl_electric_4x2_berlin',
    'other_heavy_duty_e_truck',
    'mercedes_benz_eactros_600_long_haul',
    'large_bulky_goods_e_van_2mh'
]

# Filter to trucks only
hourly_trucks = hourly_class[hourly_class['vehicle_class'].isin(truck_classes)].copy()

# Aggregate all trucks by hour
hourly_trucks_agg = hourly_trucks.groupby('hour').agg({
    'p_discharge_max_MW': 'sum',
    'fleet_size_class': 'first'  # Just to have a value
}).reset_index()

# Create hourly profile (0-23)
hourly_profile = hourly_trucks_agg['p_discharge_max_MW'].values
if len(hourly_profile) < 24:
    # Pad if necessary
    hourly_profile = np.pad(hourly_profile, (0, 24 - len(hourly_profile)), mode='constant')

# Create the heatmap (single row for trucks)
fig, ax = plt.subplots(figsize=(16, 6))

# Reshape to 1 row × 24 hours for display
data_matrix = hourly_profile.reshape(1, 24)

# Create heatmap
im = ax.imshow(data_matrix, cmap='Blues', aspect='auto', interpolation='nearest', vmin=0)

# Axes labels
ax.set_xticks(np.arange(24))
ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], fontsize=11, fontweight='bold', rotation=45)
ax.set_yticks([0])
ax.set_yticklabels(['Combined Truck Fleet'], fontsize=12, fontweight='bold')

ax.set_xlabel('Hour of Day (UTC)', fontsize=13, fontweight='bold')
ax.set_title('Hourly V2G Discharge Availability: Combined Truck Fleet\nDark blue = High availability for FCR participation throughout the day', 
            fontsize=14, fontweight='bold', pad=20)

# Colorbar
cbar = plt.colorbar(im, ax=ax, label='Available Discharge Power (MW)', orientation='horizontal', pad=0.15)
cbar.set_label('Available Discharge Power (MW)', fontsize=12, fontweight='bold')

# Add grid
ax.set_xticks(np.arange(24) - 0.5, minor=True)
ax.set_yticks([0.5], minor=True)
ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

# Add value labels on each hour
for hour in range(24):
    val = hourly_profile[hour]
    text_color = 'white' if val > np.max(hourly_profile) * 0.5 else 'black'
    ax.text(hour, 0, f'{val:.1f}', ha='center', va='center', 
           color=text_color, fontsize=11, fontweight='bold')

# Add annotation for peak hours
peak_hour = np.argmax(hourly_profile)
peak_value = hourly_profile[peak_hour]
ax.annotate(f'Peak: {peak_value:.1f} MW\nat {peak_hour:02d}:00',
           xy=(peak_hour, 0), xytext=(peak_hour, -0.7),
           ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT_BLUE, alpha=0.7),
           arrowprops=dict(arrowstyle='->', color=COLOR_DARK_BLUE, lw=2))

# Add mean line info
mean_capacity = np.mean(hourly_profile)
ax.text(0.5, 1.15, f'Average capacity: {mean_capacity:.2f} MW | Peak: {peak_value:.2f} MW | Min: {np.min(hourly_profile):.2f} MW',
       transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

# Save
png_path = OUTPUT_FOLDER / "Fig_Trucks_Hourly_Heatmap.png"
svg_path = OUTPUT_FOLDER / "Fig_Trucks_Hourly_Heatmap.svg"
plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(svg_path, bbox_inches='tight', facecolor='white')
plt.close()

print("="*80)
print("TRUCKS-ONLY HOURLY HEATMAP")
print("="*80)
print()
print("✓ Fig_Trucks_Hourly_Heatmap.png (300 dpi)")
print("✓ Fig_Trucks_Hourly_Heatmap.svg (editable)")
print()
print(f"Output folder: {OUTPUT_FOLDER}")
print()
print("Truck Classes Aggregated:")
print(f"  • volvo_fl_electric_4x2_berlin (13 vehicles, 3.45 MWh)")
print(f"  • other_heavy_duty_e_truck (16 vehicles, 5.12 MWh)")
print(f"  • mercedes_benz_eactros_600_long_haul (29 vehicles, 18.01 MWh)")
print(f"  • large_bulky_goods_e_van_2mh (21 vehicles, 1.58 MWh)")
print()
print("Hourly Discharge Availability:")
print()
for hour in range(24):
    bar_length = int(hourly_profile[hour] / np.max(hourly_profile) * 40)
    bar = '█' * bar_length
    print(f"  {hour:02d}:00 - {hourly_profile[hour]:6.2f} MW {bar}")
print()
print(f"Peak capacity: {peak_value:.2f} MW at {peak_hour:02d}:00")
print(f"Average capacity: {mean_capacity:.2f} MW")
print(f"Minimum capacity: {np.min(hourly_profile):.2f} MW")
print()
print("="*80)

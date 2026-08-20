import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Prepare data matrix (Degradation = FALSE rows to match your layout reference)
plot_data = [
    {"Scenario": "Naive",    "Net profit": -9.13,  "Charging cost": -9.13,  "Discharging revenue": 0.0,  "FCR revenue": 0.0},
    {"Scenario": "DA",       "Net profit": 6.38,   "Charging cost": -23.22, "Discharging revenue": 29.60, "FCR revenue": 0.0},
    {"Scenario": "ID",       "Net profit": 9.99,   "Charging cost": -25.56, "Discharging revenue": 35.54, "FCR revenue": 0.0},
    {"Scenario": "FCR",      "Net profit": 4.25,   "Charging cost": -3.69,  "Discharging revenue": 0.0,  "FCR revenue": 7.94},
    {"Scenario": "Combined", "Net profit": 24.00,  "Charging cost": -33.98, "Discharging revenue": 50.05, "FCR revenue": 7.94}
]

df = pd.DataFrame(plot_data)

# 2. Plot configuration
scenarios = df["Scenario"].tolist()
y_indices = np.arange(len(scenarios))
bar_height = 0.18  # Spacing thickness for grouped look

# Explicit color theme mapping from your design layout
colors = {
    "Net profit": "#1A365D",          # Deep Navy Blue
    "Charging cost": "#5C6B84",       # Muted Slate Blue/Gray
    "Discharging revenue": "#2B6CB0", # Vibrant Blue
    "FCR revenue": "#71B1D9"          # Light Sky Blue
}

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

# 3. Plotting individual horizontal groups with specific offset alignment
# We invert the y-axis index tracking so 'Naive' sits at the top row naturally
rects1 = ax.barh(y_indices + bar_height,     df["Net profit"],          bar_height, label="Net profit",          color=colors["Net profit"])
rects2 = ax.barh(y_indices,                  df["Charging cost"],       bar_height, label="Charging cost",       color=colors["Charging cost"])
rects3 = ax.barh(y_indices - bar_height,     df["Discharging revenue"], bar_height, label="Discharging revenue", color=colors["Discharging revenue"])
rects4 = ax.barh(y_indices - (2*bar_height), df["FCR revenue"],         bar_height, label="FCR revenue",         color=colors["FCR revenue"])

# 4. Axis structuring and reference alignments
ax.axvline(0, color="#1A365D", linestyle="-", linewidth=1.2) # Dynamic center spine
ax.set_xlabel("Financial contribution (€ million)", fontsize=11, labelpad=10)

# Set group placement limits and ticks matching original reference
ax.set_yticks(y_indices - bar_height / 2)
ax.set_yticklabels(scenarios, fontsize=11)
ax.set_xlim(-62, 62)
ax.set_xticks([-60, -40, -20, 0, 20, 40, 60])
ax.invert_yaxis()  # Forces Naive to top, Combined to bottom

# 5. Add text value annotations next to the bars
def add_value_labels(rects):
    for rect in rects:
        width = rect.get_width()
        if abs(width) < 0.01: # Skip zero entries to keep rendering clean
            continue
        
        # Position logic: shift left for negative costs, right for positive revenue
        offset = 0.6 if width >= 0 else -0.6
        ha_align = "left" if width >= 0 else "right"
        
        ax.annotate(
            f"{width:+.2f}" if width != df["Charging cost"].values[0] else f"{width:.2f}",
            xy=(width + offset, rect.get_y() + rect.get_height() / 2),
            xytext=(0, 0),
            textcoords="offset points",
            ha=ha_align, va="center",
            fontsize=9, fontweight="bold", color="#2D3748"
        )

# Apply annotations across all sets
for r in [rects1, rects2, rects3, rects4]:
    add_value_labels(r)

# 6. Legends & Footnotes Placement
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False, fontsize=10)

# Baseline explanatory caption layout
ax.text(-62, -0.7, "Positive values represent profit or revenue; negative values represent costs.", 
        fontsize=9, style="italic", color="#4A5568", ha="left")

# Clean layout lines to emphasize horizontal bands
ax.grid(axis='y', linestyle='None') # Disable secondary gridlines on categories
ax.grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("horizontal_vpp_breakdown.png", bbox_inches="tight")
plt.show()
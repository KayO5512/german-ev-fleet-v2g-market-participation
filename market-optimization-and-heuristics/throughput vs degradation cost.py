import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

# ============================================================
# INPUT DATA
# ============================================================
data = [
    {"scenario": "Naive_EDF", "degradation": "FALSE", "total_charged_MWh": 106050.48, "total_discharged_MWh": 0.0,        "degradation_cost_EUR": 0.0},
    {"scenario": "DA",        "degradation": "FALSE", "total_charged_MWh": 371934.48, "total_discharged_MWh": 240255.66,  "degradation_cost_EUR": 0.0},
    {"scenario": "DA",        "degradation": "TRUE",  "total_charged_MWh": 193428.88, "total_discharged_MWh": 79154.36,   "degradation_cost_EUR": 5451664.75},
    {"scenario": "ID",        "degradation": "FALSE", "total_charged_MWh": 430578.53, "total_discharged_MWh": 293181.91,  "degradation_cost_EUR": 0.0},
    {"scenario": "ID",        "degradation": "TRUE",  "total_charged_MWh": 203457.75, "total_discharged_MWh": 88205.41,   "degradation_cost_EUR": 5833263.17},
    {"scenario": "FCR",       "degradation": "FALSE", "total_charged_MWh": 105723.22, "total_discharged_MWh": 0.0,        "degradation_cost_EUR": 0.0},
    {"scenario": "FCR",       "degradation": "TRUE",  "total_charged_MWh": 105723.22, "total_discharged_MWh": 0.0,        "degradation_cost_EUR": 2114464.43},
    {"scenario": "Combined",  "degradation": "FALSE", "total_charged_MWh": 562022.03, "total_discharged_MWh": 411809.67,  "degradation_cost_EUR": 0.0},
    {"scenario": "Combined",  "degradation": "TRUE",  "total_charged_MWh": 257177.48, "total_discharged_MWh": 136687.47,  "degradation_cost_EUR": 7877299.02},
]

df = pd.DataFrame(data)

# ============================================================
# PREPARE PLOTTING DATA
# ============================================================

# Keep only degradation-active scenarios
plot_df = df[df["degradation"] == "TRUE"].copy()

# Add Naive baseline manually
naive_row = df[df["scenario"] == "Naive_EDF"].iloc[0]
naive_plot = pd.DataFrame([{
    "scenario": "Naive",
    "degradation": "TRUE",
    "total_charged_MWh": naive_row["total_charged_MWh"],
    "total_discharged_MWh": naive_row["total_discharged_MWh"],
    "degradation_cost_EUR": 0.0
}])

plot_df = pd.concat([naive_plot, plot_df], ignore_index=True)

# Scenario order
scenario_order = ["Naive", "DA", "ID", "FCR", "Combined"]
scenario_labels = {
    "Naive": "Naive",
    "DA": "Day-Ahead Only",
    "ID": "Intraday Only",
    "FCR": "FCR Capacity",
    "Combined": "Combined Markets"
}

plot_df["scenario"] = pd.Categorical(
    plot_df["scenario"],
    categories=scenario_order,
    ordered=True
)
plot_df = plot_df.sort_values("scenario").reset_index(drop=True)

plot_df["label"] = plot_df["scenario"].map(scenario_labels)
plot_df["degradation_cost_MEUR"] = plot_df["degradation_cost_EUR"] / 1e6
plot_df["total_energy_MWh"] = (
    plot_df["total_charged_MWh"] + plot_df["total_discharged_MWh"]
)

# ============================================================
# STYLE
# ============================================================
charge_color = "#BFD0E0"     # light blue
discharge_color = "#2B6CB0"  # dark blue
line_color = "#021E3E"       # red
text_color = "#1A365D"
grid_color = "#D9E2F1"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10
})

# ============================================================
# PLOT
# ============================================================
fig, ax1 = plt.subplots(figsize=(11, 6), dpi=300)

x = np.arange(len(plot_df))
bar_width = 0.45

# Stacked bars
bars_charge = ax1.bar(
    x,
    plot_df["total_charged_MWh"],
    width=bar_width,
    color=charge_color,
    edgecolor=discharge_color,
    linewidth=1.0,
    label="Charged energy"
)

bars_discharge = ax1.bar(
    x,
    plot_df["total_discharged_MWh"],
    width=bar_width,
    bottom=plot_df["total_charged_MWh"],
    color=discharge_color,
    edgecolor=discharge_color,
    linewidth=1.0,
    label="Discharged energy"
)

# Left axis
ax1.set_ylabel(
    "Battery energy throughput (MWh)",
    color=discharge_color,
    fontsize=11,
    fontweight="bold"
)
ax1.set_xlabel(
    "Market Scenario (Degradation Activated)",
    fontsize=11,
    fontweight="bold",
    labelpad=10
)
ax1.set_xticks(x)
ax1.set_xticklabels(plot_df["label"], fontsize=10)
ax1.tick_params(axis="y", labelcolor=discharge_color)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:,.0f}"))
ax1.grid(axis="y", linestyle="--", alpha=0.30, color=grid_color)
ax1.set_axisbelow(True)

# Labels above full stacked bars
max_total = plot_df["total_energy_MWh"].max()
bar_label_offset = max_total * 0.02

for i, total in enumerate(plot_df["total_energy_MWh"]):
    ax1.text(
        x[i],
        total + bar_label_offset,
        f"{total:,.0f}\nMWh",
        ha="center",
        va="bottom",
        color=text_color,
        fontsize=9,
        fontweight="bold"
    )

# Right axis for degradation cost
ax2 = ax1.twinx()
ax2.plot(
    x,
    plot_df["degradation_cost_MEUR"],
    color=line_color,
    marker="o",
    linestyle="--",
    linewidth=2.4,
    markersize=6,
    label="Degradation cost"
)

ax2.set_ylabel(
    "Resulting degradation cost (Million EUR)",
    color=line_color,
    fontsize=11,
    fontweight="bold"
)
ax2.tick_params(axis="y", labelcolor=line_color)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:.1f} M€"))

# Line labels moved to the LEFT of markers only
for i, val in enumerate(plot_df["degradation_cost_MEUR"]):
    if val > 0:
        ax2.annotate(
            f"{val:.2f} M€",
            xy=(x[i], val),
            xytext=(-8, 0),           # shift label left only
            textcoords="offset points",
            ha="right",
            va="center",
            color=line_color,
            fontsize=9,
            fontweight="bold"
        )

# Align zero on both y-axes
max_energy = plot_df["total_energy_MWh"].max()
max_deg = plot_df["degradation_cost_MEUR"].max()

ax1.set_ylim(0, max_energy * 1.18)
ax2.set_ylim(0, max_deg * 1.18)

# Title
ax1.set_title(
    "Asset Utilization vs. Degradation Overhead",
    fontsize=13,
    fontweight="bold",
    color=text_color,
    pad=14
)

# Combined legend
handles_1, labels_1 = ax1.get_legend_handles_labels()
handles_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(
    handles_1 + handles_2,
    labels_1 + labels_2,
    loc="upper left",
    frameon=False,
    ncol=3,
    fontsize=10
)

# Clean spines
ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("throughput_vs_degradation_stacked.png", dpi=300, bbox_inches="tight")
plt.show()
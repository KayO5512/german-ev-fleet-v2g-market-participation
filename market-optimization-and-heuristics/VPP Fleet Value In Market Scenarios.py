import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Recreate the summary data from the image
data = {
    "scenario": [
        "Naive_EDF",
        "DA",
        "DA",
        "ID",
        "ID",
        "FCR",
        "FCR",
        "Combined",
        "Combined",
    ],
    "degradation_on": [
        "FALSE",
        "FALSE",
        "TRUE",
        "FALSE",
        "TRUE",
        "FALSE",
        "TRUE",
        "FALSE",
        "TRUE",
    ],
    "profit_EUR": [
        -9132061.44,
        6382331.93,
        -1861326.77,
        9988370.58,
        850775.61,
        4246259.56,
        2131795.13,
        24000789.06,
        11539302.44,
    ],
}

df = pd.DataFrame(data)

# Convert Profit to Millions of EUR for presentation clarity
df["profit_M_EUR"] = df["profit_EUR"] / 1e6

# 2. Set up professional plot styling
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

# Establish matching blue color theme from your UML diagrams
colors = {"FALSE": "#2B6CB0", "TRUE": "#A0AEC0"}  # Mid Blue vs Slate Gray

# 3. Plot the clustered bars
sns.barplot(
    data=df,
    x="scenario",
    y="profit_M_EUR",
    hue="degradation_on",
    palette=colors,
    edgecolor="#1A365D",
    linewidth=1.2,
    ax=ax,
)

# Add a distinct zero line to visually emphasize negative profit/losses
ax.axhline(0, color="#1A365D", linestyle="-", linewidth=1.2, alpha=0.7)

# 4. Enhance typography and labels
ax.set_title(
    "VPP Fleet Value In Market Scenarios: Net Profit Comparison",
    fontsize=14,
    fontweight="bold",
    pad=20,
    color="#1A365D",
)
ax.set_xlabel("Market Participation Scenario", fontsize=11, fontweight="bold", labelpad=12)
ax.set_ylabel("Net Profit / Loss (Million EUR)", fontsize=11, fontweight="bold", labelpad=12)

# Professionalize axis limits and category text formatting
ax.set_xticklabels(["Naive EDF Baseline", "Day-Ahead Only", "Intraday Only", "FCR Capacity", "Combined Markets"])
ax.tick_params(axis="both", which="major", labelsize=10)

# Format the Y-axis to clearly display financial formatting (e.g., +10M, -5M)
import matplotlib.ticker as ticker

ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%+g M€"))

# 5. Add value labels on top/bottom of each bar
for p in ax.patches:
    height = p.get_height()
    if np.isnan(height):
        continue
    # Adjust position slightly depending on if the value is a profit or a loss
    va_dir = "bottom" if height >= 0 else "top"
    offset = 0.4 if height >= 0 else -0.4

    ax.annotate(
        f"{height:+.2f} M€",
        (p.get_x() + p.get_width() / 2.0, height + offset),
        ha="center",
        va=va_dir,
        fontsize=9,
        fontweight="bold",
        color="#2D3748",
        xytext=(0, 0),
        textcoords="offset points",
    )

# 6. Polished Legend Layout
legend = ax.legend(
    title="Battery Degradation Model Included",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=True,
    shadow=False,
    facecolor="white",
    edgecolor="#A0AEC0",
)
legend.get_title().set_fontweight("bold")
legend.get_texts()[0].set_text("No (Costless Cycles)")
legend.get_texts()[1].set_text("Yes (Linear €20/MWh Throughput)")

plt.tight_layout()

# Save image for your PowerPoint
plt.savefig("vpp_market_comparison.png", bbox_inches="tight")
plt.show()
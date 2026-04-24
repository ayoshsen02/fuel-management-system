"""
ranking.py
----------
Model 4 : Vehicle Efficiency Ranking (standalone module)
Re-uses the ranking function from train.py and exports a detailed report.

Run: python ranking.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR  = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "fuel.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def rank_vehicles(df: pd.DataFrame) -> pd.DataFrame:
    """Return full ranked DataFrame with efficiency scores."""

    agg = df.groupby(["Vehicle_ID", "Vehicle_Type"]).agg(
        Total_Distance_KM     = ("Distance_KM",            "sum"),
        Total_Fuel_Liters     = ("Fuel_Quantity_Liters",    "sum"),
        Total_Fuel_Cost_INR   = ("Total_Fuel_Cost_INR",     "sum"),
        Total_Maint_Cost_INR  = ("Maintenance_Cost_INR",    "sum"),
        Avg_Mileage_KMPL      = ("Mileage_KMPL",            "mean"),
        Num_Trips             = ("Distance_KM",             "count"),
        Fraud_Flags           = ("Is_Fraud",                "sum") if "Is_Fraud" in df.columns else ("Distance_KM", "count"),
    ).reset_index()

    agg["Cost_Per_KM_INR"]   = agg["Total_Fuel_Cost_INR"] / agg["Total_Distance_KM"].replace(0, np.nan)
    agg["Total_Cost_INR"]    = agg["Total_Fuel_Cost_INR"] + agg["Total_Maint_Cost_INR"]
    agg["Overall_Cost_Per_KM"] = agg["Total_Cost_INR"] / agg["Total_Distance_KM"].replace(0, np.nan)

    def norm(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    agg["Score_Mileage"]   = norm(agg["Avg_Mileage_KMPL"])
    agg["Score_CostPerKM"] = 1 - norm(agg["Cost_Per_KM_INR"])
    agg["Score_Maint"]     = 1 - norm(agg["Total_Maint_Cost_INR"])
    agg["Score_Distance"]  = norm(agg["Total_Distance_KM"])

    agg["Efficiency_Score"] = (
        0.35 * agg["Score_Mileage"]   +
        0.35 * agg["Score_CostPerKM"] +
        0.15 * agg["Score_Maint"]     +
        0.15 * agg["Score_Distance"]
    ).round(4)

    agg = agg.sort_values("Efficiency_Score", ascending=False).reset_index(drop=True)
    agg["Rank"] = agg.index + 1
    return agg


def plot_rankings(agg: pd.DataFrame):
    """Generate top/bottom bar charts and a scatter efficiency plot."""

    top10  = agg.head(10)
    bot10  = agg.tail(10)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.barplot(data=top10, x="Efficiency_Score", y="Vehicle_ID",
                palette="YlGn_r", ax=axes[0])
    axes[0].set_title("🏆  Top 10 Most Efficient Vehicles")
    axes[0].set_xlabel("Efficiency Score")

    sns.barplot(data=bot10.sort_values("Efficiency_Score"),
                x="Efficiency_Score", y="Vehicle_ID",
                palette="OrRd", ax=axes[1])
    axes[1].set_title("⚠️   Top 10 Least Efficient Vehicles")
    axes[1].set_xlabel("Efficiency Score")

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "vehicle_ranking_bars.png", dpi=150)
    plt.close()

    # Scatter: mileage vs cost per KM coloured by score
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(agg["Avg_Mileage_KMPL"], agg["Cost_Per_KM_INR"],
                     c=agg["Efficiency_Score"], cmap="RdYlGn",
                     alpha=0.8, s=60, edgecolors="grey", linewidths=0.3)
    plt.colorbar(sc, label="Efficiency Score")
    plt.xlabel("Avg Mileage (KMPL)")
    plt.ylabel("Cost per KM (₹)")
    plt.title("Vehicle Efficiency Map – Mileage vs Cost/KM")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "vehicle_efficiency_map.png", dpi=150)
    plt.close()

    print("  📊  Charts saved.")


if __name__ == "__main__":
    print("\n📂  Loading dataset …")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

    print("\n" + "═"*60)
    print("  MODEL 4 : Vehicle Efficiency Ranking")
    print("═"*60)

    ranked = rank_vehicles(df)
    plot_rankings(ranked)
    ranked.to_csv(MODEL_DIR / "vehicle_rankings_full.csv", index=False)

    print(f"\n  Total vehicles ranked : {len(ranked)}")
    print("\n  🏆  TOP 10")
    print(ranked[["Rank", "Vehicle_ID", "Vehicle_Type",
                  "Avg_Mileage_KMPL", "Cost_Per_KM_INR",
                  "Efficiency_Score"]].head(10).to_string(index=False))

    print("\n  ⚠️   BOTTOM 10")
    print(ranked[["Rank", "Vehicle_ID", "Vehicle_Type",
                  "Avg_Mileage_KMPL", "Cost_Per_KM_INR",
                  "Efficiency_Score"]].tail(10).to_string(index=False))

    print("\n  ✅  Rankings saved → models/vehicle_rankings_full.csv")

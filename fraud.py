"""
fraud.py
--------
Model 2 : Fuel Fraud Detection using Isolation Forest

Detects:
  • High fuel bills (unusually high quantity/cost)
  • Low KM but high fuel consumption
  • Duplicate / suspicious entries

Run: python fraud.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble        import IsolationForest
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import classification_report, precision_score, recall_score

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "fuel.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def run_fraud_detection(df: pd.DataFrame):
    print("\n" + "═"*60)
    print("  MODEL 2 : Fraud Detection (Isolation Forest)")
    print("═"*60)

    # ── Feature engineering for anomaly detection ────────────────────────────
    df2 = df.copy()
    df2["Cost_Per_KM"]      = df2["Total_Fuel_Cost_INR"] / df2["Distance_KM"].replace(0, np.nan)
    df2["Fuel_Per_KM"]      = df2["Fuel_Quantity_Liters"] / df2["Distance_KM"].replace(0, np.nan)
    df2["Expected_Fuel"]    = df2["Distance_KM"] / df2["Mileage_KMPL"].replace(0, np.nan)
    df2["Fuel_Excess"]      = df2["Fuel_Quantity_Liters"] - df2["Expected_Fuel"]
    df2["Price_Deviation"]  = (
        df2["Fuel_Price_Per_Liter_INR"] -
        df2.groupby("Fuel_Type")["Fuel_Price_Per_Liter_INR"].transform("mean")
    )

    feature_cols = [
        "Fuel_Quantity_Liters",
        "Total_Fuel_Cost_INR",
        "Distance_KM",
        "Mileage_KMPL",
        "Cost_Per_KM",
        "Fuel_Per_KM",
        "Fuel_Excess",
        "Price_Deviation",
    ]

    X = df2[feature_cols].fillna(0)

    # ── Scale features ────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train Isolation Forest ────────────────────────────────────────────────
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,   # ~5% expected fraud
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    # ── Predict: -1 = anomaly, 1 = normal  →  remap to 1/0 ──────────────────
    raw_pred          = iso.predict(X_scaled)
    df2["Anomaly"]    = (raw_pred == -1).astype(int)
    df2["Anomaly_Score"] = -iso.score_samples(X_scaled)   # higher = more anomalous

    # ── Evaluation against synthetic Is_Fraud label ──────────────────────────
    if "Is_Fraud" in df2.columns:
        y_true = df2["Is_Fraud"]
        y_pred = df2["Anomaly"]
        print(f"\n  Precision : {precision_score(y_true, y_pred):.4f}")
        print(f"  Recall    : {recall_score(y_true, y_pred):.4f}")
        print("\n  Classification Report:")
        print(classification_report(y_true, y_pred,
                                    target_names=["Normal", "Fraud"]))

    # ── Fraud summary ─────────────────────────────────────────────────────────
    fraud_df = df2[df2["Anomaly"] == 1].copy()
    print(f"\n  🚨  Total Fraud Alerts : {len(fraud_df)}")
    print(f"  📅  Date Range         : {df2['Date'].min().date()} → {df2['Date'].max().date()}")

    fraud_cols = ["Date", "Vehicle_ID", "City", "Fuel_Type",
                  "Fuel_Quantity_Liters", "Total_Fuel_Cost_INR",
                  "Distance_KM", "Mileage_KMPL", "Anomaly_Score"]
    fraud_df_out = fraud_df[fraud_cols].sort_values("Anomaly_Score", ascending=False)
    fraud_df_out.to_csv(MODEL_DIR / "fraud_alerts.csv", index=False)
    print("  ✅  Fraud alerts saved → models/fraud_alerts.csv")

    # ── Scatter plot: cost vs distance (highlight fraud) ─────────────────────
    plt.figure(figsize=(10, 6))
    normal = df2[df2["Anomaly"] == 0]
    fraud  = df2[df2["Anomaly"] == 1]

    plt.scatter(normal["Distance_KM"], normal["Total_Fuel_Cost_INR"],
                c="steelblue", alpha=0.3, s=15, label="Normal")
    plt.scatter(fraud["Distance_KM"],  fraud["Total_Fuel_Cost_INR"],
                c="crimson", alpha=0.7, s=40, label="Fraud / Anomaly", zorder=3)

    plt.xlabel("Distance (KM)")
    plt.ylabel("Total Fuel Cost (₹ INR)")
    plt.title("Fraud Detection – Distance vs Cost\n(Red = Anomaly)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "fraud_detection_scatter.png", dpi=150)
    plt.close()

    # ── Anomaly score distribution ────────────────────────────────────────────
    plt.figure(figsize=(10, 4))
    sns.histplot(df2["Anomaly_Score"], bins=60, kde=True, color="darkorange")
    threshold = df2[df2["Anomaly"] == 1]["Anomaly_Score"].min()
    plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold ≈ {threshold:.3f}")
    plt.xlabel("Anomaly Score (higher = more suspicious)")
    plt.title("Anomaly Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "anomaly_score_dist.png", dpi=150)
    plt.close()
    print("  📊  Charts saved.")

    # ── Save models ───────────────────────────────────────────────────────────
    joblib.dump(iso,    MODEL_DIR / "fraud_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "fraud_scaler.pkl")
    return fraud_df_out


if __name__ == "__main__":
    print("\n📂  Loading dataset …")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    fraud_results = run_fraud_detection(df)
    print("\n  Sample Fraud Records:")
    print(fraud_results.head(5).to_string(index=False))

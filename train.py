"""
train.py
--------
Model 1 : Fuel Consumption Prediction  (XGBoost Regressor)
Model 4 : Vehicle Efficiency Ranking

Run: python train.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score
from xgboost                 import XGBRegressor

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR / "data" / "fuel.csv"
MODEL_DIR  = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ─── Helpers ─────────────────────────────────────────────────────────────────
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL 1 – FUEL CONSUMPTION PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def train_fuel_prediction(df: pd.DataFrame):
    print("\n" + "═"*60)
    print("  MODEL 1 : Fuel Consumption Prediction (XGBoost)")
    print("═"*60)

    # ── Feature engineering ──────────────────────────────────────────────────
    df2 = df.copy()
    df2["Month"]     = df2["Date"].dt.month
    df2["DayOfWeek"] = df2["Date"].dt.dayofweek
    df2["Quarter"]   = df2["Date"].dt.quarter

    cat_cols = ["Vehicle_Type", "Fuel_Type", "City", "State",
                "Department", "Route_Type"]
    le_dict  = {}
    for col in cat_cols:
        le = LabelEncoder()
        df2[col] = le.fit_transform(df2[col].astype(str))
        le_dict[col] = le

    feature_cols = ["Month", "DayOfWeek", "Quarter", "Vehicle_Type",
                    "Fuel_Type", "City", "State", "Fuel_Price_Per_Liter_INR",
                    "Distance_KM", "Mileage_KMPL", "Maintenance_Cost_INR",
                    "Department", "Route_Type"]
    target_col   = "Fuel_Quantity_Liters"

    X = df2[feature_cols]
    y = df2[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    print(f"  RMSE : {rmse(y_test, y_pred):.4f}  litres")
    print(f"  MAE  : {mean_absolute_error(y_test, y_pred):.4f}  litres")
    print(f"  R²   : {r2_score(y_test, y_pred):.4f}")

    # ── Feature importance plot ───────────────────────────────────────────────
    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=fi.values[:10], y=fi.index[:10], palette="viridis")
    plt.title("Top-10 Feature Importances – Fuel Consumption (XGBoost)")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "fuel_prediction_feature_importance.png", dpi=150)
    plt.close()
    print("  📊  Feature importance chart saved.")

    # ── Save model + encoders ─────────────────────────────────────────────────
    joblib.dump(model,   MODEL_DIR / "fuel_prediction_model.pkl")
    joblib.dump(le_dict, MODEL_DIR / "fuel_prediction_encoders.pkl")
    print("  ✅  Model saved → models/fuel_prediction_model.pkl")
    return model, le_dict


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL 4 – VEHICLE EFFICIENCY RANKING
# ══════════════════════════════════════════════════════════════════════════════
def vehicle_efficiency_ranking(df: pd.DataFrame):
    print("\n" + "═"*60)
    print("  MODEL 4 : Vehicle Efficiency Ranking")
    print("═"*60)

    # ── Aggregate per vehicle ─────────────────────────────────────────────────
    agg = df.groupby("Vehicle_ID").agg(
        Total_Distance_KM      = ("Distance_KM",           "sum"),
        Total_Fuel_Liters      = ("Fuel_Quantity_Liters",   "sum"),
        Total_Fuel_Cost_INR    = ("Total_Fuel_Cost_INR",    "sum"),
        Total_Maintenance_INR  = ("Maintenance_Cost_INR",   "sum"),
        Avg_Mileage_KMPL       = ("Mileage_KMPL",           "mean"),
        Num_Trips              = ("Distance_KM",            "count"),
    ).reset_index()

    # ── Derived efficiency metrics ────────────────────────────────────────────
    agg["Cost_Per_KM_INR"]     = (agg["Total_Fuel_Cost_INR"] /
                                   agg["Total_Distance_KM"].replace(0, np.nan))
    agg["Total_Cost_INR"]      = (agg["Total_Fuel_Cost_INR"] +
                                   agg["Total_Maintenance_INR"])
    agg["Overall_Cost_Per_KM"] = (agg["Total_Cost_INR"] /
                                   agg["Total_Distance_KM"].replace(0, np.nan))

    # ── Composite score (higher = more efficient) ─────────────────────────────
    # Normalize each metric to [0,1] and compute weighted score
    def norm(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    agg["Score_Mileage"]   = norm(agg["Avg_Mileage_KMPL"])
    agg["Score_CostPerKM"] = 1 - norm(agg["Cost_Per_KM_INR"])   # lower cost = higher score
    agg["Score_Maint"]     = 1 - norm(agg["Total_Maintenance_INR"])
    agg["Score_Distance"]  = norm(agg["Total_Distance_KM"])

    agg["Efficiency_Score"] = (
        0.35 * agg["Score_Mileage"]   +
        0.35 * agg["Score_CostPerKM"] +
        0.15 * agg["Score_Maint"]     +
        0.15 * agg["Score_Distance"]
    ).round(4)

    agg = agg.sort_values("Efficiency_Score", ascending=False).reset_index(drop=True)
    agg["Rank"] = agg.index + 1

    # ── Print top/bottom 10 ───────────────────────────────────────────────────
    display_cols = ["Rank", "Vehicle_ID", "Avg_Mileage_KMPL",
                    "Cost_Per_KM_INR", "Total_Distance_KM", "Efficiency_Score"]

    print("\n  🏆  TOP 10 Most Efficient Vehicles")
    print(agg[display_cols].head(10).to_string(index=False))
    print("\n  ⚠️   TOP 10 Least Efficient Vehicles")
    print(agg[display_cols].tail(10).to_string(index=False))

    # ── Save ranking CSV ──────────────────────────────────────────────────────
    agg.to_csv(MODEL_DIR / "vehicle_rankings.csv", index=False)

    # ── Bar chart – top 10 ────────────────────────────────────────────────────
    top10 = agg.head(10)
    plt.figure(figsize=(12, 5))
    sns.barplot(data=top10, x="Vehicle_ID", y="Efficiency_Score", palette="YlGn_r")
    plt.title("Top 10 Most Efficient Vehicles")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "vehicle_efficiency_ranking.png", dpi=150)
    plt.close()

    print("\n  ✅  Rankings saved → models/vehicle_rankings.csv")
    return agg


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n📂  Loading dataset …")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    print(f"   Shape: {df.shape}")

    train_fuel_prediction(df)
    vehicle_efficiency_ranking(df)

    print("\n✅  Training complete. All models saved in /models/")

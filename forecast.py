"""
forecast.py
-----------
Model 3 : Monthly Fuel Cost Forecasting using Meta Prophet

Forecasts next 3 / 6 / 12 months of total fleet fuel spending in ₹ INR.

Run: python forecast.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    from prophet import Prophet
except ImportError:
    from fbprophet import Prophet

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "fuel.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def run_forecasting(df: pd.DataFrame, forecast_months: int = 12):
    print("\n" + "═"*60)
    print(f"  MODEL 3 : Cost Forecasting (Prophet) – {forecast_months} months")
    print("═"*60)

    # ── Aggregate to daily total fuel cost ───────────────────────────────────
    daily = (
        df.groupby("Date")["Total_Fuel_Cost_INR"]
        .sum()
        .reset_index()
        .rename(columns={"Date": "ds", "Total_Fuel_Cost_INR": "y"})
    )
    daily = daily.sort_values("ds").reset_index(drop=True)
    print(f"  Daily records for training : {len(daily)}")

    # ── Train Prophet ─────────────────────────────────────────────────────────
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    model.fit(daily)

    # ── Forecast ──────────────────────────────────────────────────────────────
    future    = model.make_future_dataframe(periods=forecast_months * 30, freq="D")
    forecast  = model.predict(future)

    # ── Monthly aggregation ───────────────────────────────────────────────────
    forecast["Month"] = forecast["ds"].dt.to_period("M")
    monthly = forecast.groupby("Month").agg(
        Forecasted_Cost_INR = ("yhat",  "sum"),
        Lower_Bound_INR     = ("yhat_lower", "sum"),
        Upper_Bound_INR     = ("yhat_upper", "sum"),
    ).reset_index()
    monthly["Month"] = monthly["Month"].astype(str)

    # Keep only future months
    last_actual_month = daily["ds"].max().to_period("M").strftime("%Y-%m")
    future_monthly = monthly[monthly["Month"] > last_actual_month].copy()

    print(f"\n  📅  Forecast period : next {forecast_months} months")
    print(future_monthly.head(forecast_months).to_string(index=False))

    future_monthly.to_csv(MODEL_DIR / "monthly_forecast.csv", index=False)
    print("\n  ✅  Forecast saved → models/monthly_forecast.csv")

    # ── Plot 1 : full forecast ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(forecast["ds"], forecast["yhat_lower"],
                    forecast["yhat_upper"], alpha=0.2, color="steelblue",
                    label="Confidence Interval")
    ax.plot(forecast["ds"], forecast["yhat"], color="steelblue", linewidth=1.5,
            label="Forecast")
    ax.plot(daily["ds"], daily["y"], "ko", markersize=2, alpha=0.5,
            label="Actual")
    ax.axvline(daily["ds"].max(), color="red", linestyle="--", linewidth=1,
               label="Forecast Start")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x/1e6:.1f}M" if x >= 1e6 else f"₹{x/1e3:.0f}K"
    ))
    ax.set_title("Fleet Fuel Cost Forecast – Prophet")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Daily Fuel Cost (₹ INR)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "cost_forecast.png", dpi=150)
    plt.close()

    # ── Plot 2 : components ────────────────────────────────────────────────────
    fig2 = model.plot_components(forecast)
    fig2.suptitle("Forecast Components (Trend + Seasonality)", y=1.01)
    plt.tight_layout()
    fig2.savefig(MODEL_DIR / "forecast_components.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  📊  Forecast charts saved.")

    # ── Save model ─────────────────────────────────────────────────────────────
    joblib.dump(model, MODEL_DIR / "forecast_model.pkl")
    return future_monthly


if __name__ == "__main__":
    print("\n📂  Loading dataset …")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

    for months in [3, 6, 12]:
        run_forecasting(df, forecast_months=months)

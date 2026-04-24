"""
setup.py
--------
Auto-runs all 4 models if their output files are missing.
Called by app.py on startup — works on Streamlit Cloud.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

BASE_DIR  = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "fuel.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR  = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


# ─── Step 0: Generate dataset if missing ─────────────────────────────────────
def ensure_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH, parse_dates=["Date"])

    print("⚙️  Generating dataset...")
    np.random.seed(42)

    CITIES_STATES = {
        "Delhi":"Delhi","Mumbai":"Maharashtra","Bengaluru":"Karnataka",
        "Chennai":"Tamil Nadu","Hyderabad":"Telangana","Kolkata":"West Bengal",
        "Pune":"Maharashtra","Ahmedabad":"Gujarat"
    }
    VEHICLE_TYPES  = ["Truck","Bus","Car","Van","SUV","Mini-Truck"]
    FUEL_TYPES     = ["Petrol","Diesel","CNG"]
    DEPARTMENTS    = ["Logistics","Sales","Operations","Maintenance","Administration"]
    ROUTE_TYPES    = ["Urban","Highway","Mixed"]
    FUEL_PRICES    = {"Petrol":(94,105),"Diesel":(85,95),"CNG":(75,85)}
    MILEAGE_RANGE  = {"Truck":(5,12),"Bus":(4,8),"Car":(12,22),"Van":(10,16),"SUV":(8,14),"Mini-Truck":(8,14)}

    dates  = pd.date_range("2024-01-01","2025-12-31",freq="D")
    cities = list(CITIES_STATES.keys())
    rows   = []

    for i in range(1, 5001):
        city  = np.random.choice(cities)
        vtype = np.random.choice(VEHICLE_TYPES)
        ftype = np.random.choice(FUEL_TYPES, p=[0.35,0.50,0.15])
        route = np.random.choice(ROUTE_TYPES)
        date  = pd.Timestamp(np.random.choice(dates))
        price = round(np.random.uniform(*FUEL_PRICES[ftype]), 2)
        mil   = round(np.random.uniform(*MILEAGE_RANGE[vtype]), 2)
        base  = 200 if route=="Highway" else (80 if route=="Urban" else 130)
        dist  = round(abs(np.random.normal(base, 40)), 1)
        qty   = round(dist / mil, 2)
        cost  = round(qty * price, 2)
        maint = round(abs(np.random.normal(1200, 500)), 2)
        fraud = 0
        if np.random.rand() < 0.05:
            fraud = 1
            ft = np.random.choice(["high_bill","low_km_high_fuel"])
            if ft == "high_bill":
                qty = round(qty * 3, 2); cost = round(qty * price, 2)
            else:
                dist = round(dist * 0.2, 1); qty = round(qty * 2, 2)
                cost = round(qty * price, 2); mil = round(dist / max(qty, 0.1), 2)
        rows.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Vehicle_ID": f"VH{1000+(i%200):04d}",
            "Vehicle_Type": vtype, "Driver_ID": f"DR{500+(i%100):04d}",
            "City": city, "State": CITIES_STATES[city], "Fuel_Type": ftype,
            "Fuel_Quantity_Liters": qty, "Fuel_Price_Per_Liter_INR": price,
            "Total_Fuel_Cost_INR": cost, "Distance_KM": dist, "Mileage_KMPL": mil,
            "Maintenance_Cost_INR": maint, "Department": np.random.choice(DEPARTMENTS),
            "Route_Type": route, "Is_Fraud": fraud
        })

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df.to_csv(DATA_PATH, index=False)
    print(f"✅ Dataset generated: {df.shape}")
    return df


# ─── Step 1: Train XGBoost model ─────────────────────────────────────────────
def ensure_prediction_model(df):
    if (MODEL_DIR / "fuel_prediction_model.pkl").exists():
        return

    print("⚙️  Training XGBoost model...")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing   import LabelEncoder
    from xgboost                 import XGBRegressor

    df2 = df.copy()
    df2["Month"]     = df2["Date"].dt.month
    df2["DayOfWeek"] = df2["Date"].dt.dayofweek
    df2["Quarter"]   = df2["Date"].dt.quarter

    cat_cols = ["Vehicle_Type","Fuel_Type","City","State","Department","Route_Type"]
    le_dict  = {}
    for col in cat_cols:
        le = LabelEncoder()
        df2[col] = le.fit_transform(df2[col].astype(str))
        le_dict[col] = le

    features = ["Month","DayOfWeek","Quarter","Vehicle_Type","Fuel_Type","City","State",
                "Fuel_Price_Per_Liter_INR","Distance_KM","Mileage_KMPL",
                "Maintenance_Cost_INR","Department","Route_Type"]
    X = df2[features]; y = df2["Fuel_Quantity_Liters"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6,
                         random_state=42, verbosity=0)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    joblib.dump(model,   MODEL_DIR / "fuel_prediction_model.pkl")
    joblib.dump(le_dict, MODEL_DIR / "fuel_prediction_encoders.pkl")
    print("✅ XGBoost model saved")


# ─── Step 2: Fraud detection ─────────────────────────────────────────────────
def ensure_fraud_model(df):
    if (MODEL_DIR / "fraud_alerts.csv").exists():
        return

    print("⚙️  Running fraud detection...")
    from sklearn.ensemble      import IsolationForest
    from sklearn.preprocessing import StandardScaler

    df2 = df.copy()
    df2["Cost_Per_KM"]   = df2["Total_Fuel_Cost_INR"] / df2["Distance_KM"].replace(0, np.nan)
    df2["Fuel_Per_KM"]   = df2["Fuel_Quantity_Liters"] / df2["Distance_KM"].replace(0, np.nan)
    df2["Expected_Fuel"] = df2["Distance_KM"] / df2["Mileage_KMPL"].replace(0, np.nan)
    df2["Fuel_Excess"]   = df2["Fuel_Quantity_Liters"] - df2["Expected_Fuel"]
    df2["Price_Dev"]     = (df2["Fuel_Price_Per_Liter_INR"] -
                            df2.groupby("Fuel_Type")["Fuel_Price_Per_Liter_INR"].transform("mean"))

    feat = ["Fuel_Quantity_Liters","Total_Fuel_Cost_INR","Distance_KM",
            "Mileage_KMPL","Cost_Per_KM","Fuel_Per_KM","Fuel_Excess","Price_Dev"]
    X = df2[feat].fillna(0)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    iso.fit(Xs)
    df2["Anomaly"]       = (iso.predict(Xs) == -1).astype(int)
    df2["Anomaly_Score"] = -iso.score_samples(Xs)

    fraud_df = df2[df2["Anomaly"] == 1][
        ["Date","Vehicle_ID","City","Fuel_Type",
         "Fuel_Quantity_Liters","Total_Fuel_Cost_INR",
         "Distance_KM","Mileage_KMPL","Anomaly_Score"]
    ].sort_values("Anomaly_Score", ascending=False)

    fraud_df.to_csv(MODEL_DIR / "fraud_alerts.csv", index=False)
    joblib.dump(iso, MODEL_DIR / "fraud_model.pkl")
    joblib.dump(sc,  MODEL_DIR / "fraud_scaler.pkl")
    print("✅ Fraud alerts saved")


# ─── Step 3: Forecast ────────────────────────────────────────────────────────
def ensure_forecast(df):
    if (MODEL_DIR / "monthly_forecast.csv").exists():
        return

    print("⚙️  Running Prophet forecast...")
    try:
        from prophet import Prophet
    except ImportError:
        from fbprophet import Prophet

    daily = (df.groupby("Date")["Total_Fuel_Cost_INR"]
               .sum().reset_index()
               .rename(columns={"Date":"ds","Total_Fuel_Cost_INR":"y"}))

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                seasonality_mode="multiplicative")
    m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    m.fit(daily)

    future   = m.make_future_dataframe(periods=365, freq="D")
    forecast = m.predict(future)

    forecast["Month"] = forecast["ds"].dt.to_period("M")
    monthly = forecast.groupby("Month").agg(
        Forecasted_Cost_INR=("yhat","sum"),
        Lower_Bound_INR=("yhat_lower","sum"),
        Upper_Bound_INR=("yhat_upper","sum"),
    ).reset_index()
    monthly["Month"] = monthly["Month"].astype(str)

    last_m   = daily["ds"].max().to_period("M").strftime("%Y-%m")
    future_m = monthly[monthly["Month"] > last_m].head(12)
    future_m.to_csv(MODEL_DIR / "monthly_forecast.csv", index=False)
    print("✅ Forecast saved")


# ─── Step 4: Vehicle rankings ─────────────────────────────────────────────────
def ensure_rankings(df):
    if (MODEL_DIR / "vehicle_rankings_full.csv").exists():
        return

    print("⚙️  Computing vehicle rankings...")
    agg = df.groupby("Vehicle_ID").agg(
        Total_Distance_KM    =("Distance_KM","sum"),
        Total_Fuel_Liters    =("Fuel_Quantity_Liters","sum"),
        Total_Fuel_Cost_INR  =("Total_Fuel_Cost_INR","sum"),
        Total_Maint_Cost_INR =("Maintenance_Cost_INR","sum"),
        Avg_Mileage_KMPL     =("Mileage_KMPL","mean"),
        Num_Trips            =("Distance_KM","count"),
    ).reset_index()

    agg["Cost_Per_KM_INR"] = agg["Total_Fuel_Cost_INR"] / agg["Total_Distance_KM"].replace(0, np.nan)

    def norm(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)

    agg["Efficiency_Score"] = (
        0.35 * norm(agg["Avg_Mileage_KMPL"]) +
        0.35 * (1 - norm(agg["Cost_Per_KM_INR"])) +
        0.15 * (1 - norm(agg["Total_Maint_Cost_INR"])) +
        0.15 * norm(agg["Total_Distance_KM"])
    ).round(4)

    agg = agg.sort_values("Efficiency_Score", ascending=False).reset_index(drop=True)
    agg["Rank"] = agg.index + 1
    agg.to_csv(MODEL_DIR / "vehicle_rankings_full.csv", index=False)
    print("✅ Rankings saved")


# ─── Master setup function (called by app.py) ────────────────────────────────
def run_all_setup():
    df = ensure_dataset()
    ensure_prediction_model(df)
    ensure_fraud_model(df)
    ensure_forecast(df)
    ensure_rankings(df)
    return df
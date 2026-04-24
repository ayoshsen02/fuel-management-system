"""
generate_data.py
----------------
Generates a realistic synthetic Indian fuel dataset (2024-2025) and saves it to data/fuel.csv
Run: python src/generate_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─── Seed for reproducibility ───────────────────────────────────────────────
np.random.seed(42)

# ─── Reference tables ────────────────────────────────────────────────────────
CITIES_STATES = {
    "Delhi":     "Delhi",
    "Mumbai":    "Maharashtra",
    "Bengaluru": "Karnataka",
    "Chennai":   "Tamil Nadu",
    "Hyderabad": "Telangana",
    "Kolkata":   "West Bengal",
    "Pune":      "Maharashtra",
    "Ahmedabad": "Gujarat",
}

VEHICLE_TYPES  = ["Truck", "Bus", "Car", "Van", "SUV", "Mini-Truck"]
FUEL_TYPES     = ["Petrol", "Diesel", "CNG"]
DEPARTMENTS    = ["Logistics", "Sales", "Operations", "Maintenance", "Administration"]
ROUTE_TYPES    = ["Urban", "Highway", "Mixed"]

# Approximate 2024-25 Indian fuel prices (₹/litre)
FUEL_PRICES = {
    "Petrol": (94, 105),   # min, max range
    "Diesel": (85,  95),
    "CNG":    (75,  85),
}

# Mileage range (km/l) by vehicle type
MILEAGE_RANGE = {
    "Truck":      (5, 12),
    "Bus":        (4,  8),
    "Car":        (12, 22),
    "Van":        (10, 16),
    "SUV":        (8,  14),
    "Mini-Truck": (8,  14),
}

# ─── Generate data ───────────────────────────────────────────────────────────
def generate_dataset(n_records: int = 5000) -> pd.DataFrame:
    dates        = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    cities       = list(CITIES_STATES.keys())

    rows = []
    for i in range(1, n_records + 1):
        city         = np.random.choice(cities)
        state        = CITIES_STATES[city]
        vehicle_type = np.random.choice(VEHICLE_TYPES)
        fuel_type    = np.random.choice(FUEL_TYPES, p=[0.35, 0.50, 0.15])
        dept         = np.random.choice(DEPARTMENTS)
        route        = np.random.choice(ROUTE_TYPES)

        date         = pd.Timestamp(np.random.choice(dates))

        # Price with seasonal noise
        p_lo, p_hi   = FUEL_PRICES[fuel_type]
        price        = round(np.random.uniform(p_lo, p_hi), 2)

        # Mileage
        m_lo, m_hi   = MILEAGE_RANGE[vehicle_type]
        mileage      = round(np.random.uniform(m_lo, m_hi), 2)

        # Distance (highway trips are longer)
        base_dist    = 200 if route == "Highway" else (80 if route == "Urban" else 130)
        distance     = round(abs(np.random.normal(base_dist, 40)), 1)

        # Fuel quantity derived from distance & mileage
        quantity     = round(distance / mileage, 2)
        total_cost   = round(quantity * price, 2)
        maint_cost   = round(abs(np.random.normal(1200, 500)), 2)

        # ── Inject ~5% fraud anomalies ──────────────────────────────────────
        is_fraud = False
        if np.random.rand() < 0.05:
            is_fraud = True
            fraud_type = np.random.choice(["high_bill", "low_km_high_fuel", "duplicate"])
            if fraud_type == "high_bill":
                quantity   = round(quantity * np.random.uniform(2.5, 4.0), 2)
                total_cost = round(quantity * price, 2)
            elif fraud_type == "low_km_high_fuel":
                distance   = round(distance * 0.2, 1)
                quantity   = round(quantity * 2.0, 2)
                total_cost = round(quantity * price, 2)
                mileage    = round(distance / max(quantity, 0.1), 2)
            else:  # duplicate – same values as previous row (if any)
                if rows:
                    prev       = rows[-1].copy()
                    prev["id"] = i
                    rows.append(prev)
                    continue

        rows.append({
            "id":                     i,
            "Date":                   date.strftime("%Y-%m-%d"),
            "Vehicle_ID":             f"VH{1000 + (i % 200):04d}",
            "Vehicle_Type":           vehicle_type,
            "Driver_ID":              f"DR{500 + (i % 100):04d}",
            "City":                   city,
            "State":                  state,
            "Fuel_Type":              fuel_type,
            "Fuel_Quantity_Liters":   quantity,
            "Fuel_Price_Per_Liter_INR": price,
            "Total_Fuel_Cost_INR":    total_cost,
            "Distance_KM":            distance,
            "Mileage_KMPL":           mileage,
            "Maintenance_Cost_INR":   maint_cost,
            "Department":             dept,
            "Route_Type":             route,
            "Is_Fraud":               int(is_fraud),   # label (for evaluation only)
        })

    df = pd.DataFrame(rows).drop(columns=["id"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "data" / "fuel.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(5000)
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset saved → {output_path}  |  Shape: {df.shape}")
    print(df.head(3).to_string())

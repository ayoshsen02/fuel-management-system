# Smart Fuel Management System

> A complete, production-grade Data Science project for Indian fleet fuel analytics.
> Built with Python · XGBoost · Prophet · Isolation Forest · Streamlit

---

## 📊 What This Project Does

| # | Model | Algorithm | Output |
|---|-------|-----------|--------|
| 1 | **Fuel Consumption Prediction** | XGBoost Regressor | Predict litres needed per trip |
| 2 | **Fraud Detection** | Isolation Forest | Flag suspicious fuel entries |
| 3 | **Cost Forecasting** | Meta Prophet | 3/6/12-month ₹ INR forecast |
| 4 | **Vehicle Efficiency Ranking** | Weighted Composite Score | Best & worst vehicles |

---

## 📁 Project Structure

```
fuel_project/
├── data/
│   └── fuel.csv               ← Indian fuel dataset 2024–2025
├── notebooks/
│   └── fuel_analysis.ipynb    ← Google Colab notebook (all steps)
├── models/                    ← Saved ML models + CSV outputs
├── src/
│   └── generate_data.py       ← Synthetic dataset generator
├── app.py                     ← Streamlit dashboard
├── train.py                   ← Model 1 + Model 4
├── fraud.py                   ← Model 2 (Fraud Detection)
├── forecast.py                ← Model 3 (Prophet)
├── ranking.py                 ← Model 4 standalone
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start (VS Code)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the Indian dataset
```bash
python src/generate_data.py
```
This creates `data/fuel.csv` with 5,000 records for Indian cities (2024–25).

### 3. Train models
```bash
python train.py        # XGBoost prediction + Vehicle ranking
python fraud.py        # Isolation Forest fraud detection
python forecast.py     # Prophet cost forecasting
python ranking.py      # Full vehicle ranking report
```

### 4. Launch the Streamlit dashboard
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

---

## 🌐 Google Colab

1. Open `notebooks/fuel_analysis.ipynb` in Google Colab
2. Run **all cells top to bottom** — the notebook installs its own dependencies
3. All 4 models run inline with charts rendered in the notebook

---

## 📋 Dataset Columns (`fuel.csv`)

| Column | Description |
|--------|-------------|
| `Date` | Date of fuel entry (2024-01-01 to 2025-12-31) |
| `Vehicle_ID` | Unique vehicle identifier |
| `Vehicle_Type` | Truck / Bus / Car / Van / SUV / Mini-Truck |
| `Driver_ID` | Unique driver identifier |
| `City` | Delhi, Mumbai, Bengaluru, Chennai, Hyderabad, Kolkata, Pune, Ahmedabad |
| `State` | Indian state |
| `Fuel_Type` | Petrol / Diesel / CNG |
| `Fuel_Quantity_Liters` | Fuel filled (litres) |
| `Fuel_Price_Per_Liter_INR` | Fuel price (₹/litre) |
| `Total_Fuel_Cost_INR` | Total fuel cost (₹) |
| `Distance_KM` | Distance traveled (km) |
| `Mileage_KMPL` | Fuel efficiency (km/litre) |
| `Maintenance_Cost_INR` | Maintenance cost (₹) |
| `Department` | Logistics / Sales / Operations / Maintenance / Administration |
| `Route_Type` | Urban / Highway / Mixed |

---

## 📈 Model Details

### Model 1 – XGBoost Fuel Prediction
- **Input**: Vehicle type, fuel type, city, route, distance, price, mileage
- **Output**: Predicted fuel quantity (litres)
- **Metrics**: RMSE, MAE, R²

### Model 2 – Isolation Forest Fraud Detection
- **Detects**: High fuel bills · Low KM but high fuel · Price anomalies
- **Output**: `fraud_alerts.csv` with anomaly scores
- **Metrics**: Precision, Recall

### Model 3 – Prophet Cost Forecasting
- **Input**: Historical daily fuel costs
- **Output**: 3/6/12-month monthly forecasts with confidence bounds
- **Saved**: `monthly_forecast.csv`

### Model 4 – Vehicle Efficiency Ranking
- **Score components**: Mileage (35%) + Cost/KM (35%) + Maintenance (15%) + Distance (15%)
- **Output**: Full ranked table, top 10 / bottom 10 charts
- **Saved**: `vehicle_rankings_full.csv`

---

## 🛠️ Technologies Used

```
Python 3.10+  |  Pandas  |  NumPy  |  Matplotlib  |  Seaborn  |  Plotly
Scikit-learn  |  XGBoost  |  Prophet  |  Streamlit  |  Joblib
```

---

## 🚀 Dashboard Features

- ✅ Upload your own `fuel.csv`
- ✅ KPI cards: Total Cost ₹, Fuel Used, Distance, Avg Mileage
- ✅ Monthly cost trend · State-wise spending · Vehicle type breakdown
- ✅ Live fuel consumption prediction form
- ✅ Fraud alerts table with download
- ✅ 3/6/12-month forecast chart
- ✅ Vehicle efficiency rankings (top 10 / bottom 10)
- ✅ Download all reports as CSV

---

*Made for Indian fleet management | All amounts in ₹ INR*

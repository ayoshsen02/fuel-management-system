"""
app.py
------
AI-Powered Smart Fuel Management Dashboard (Streamlit)

Run: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Fuel Management | India",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .kpi-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 12px;
    padding: 20px;
    border-left: 4px solid #e94560;
    margin-bottom: 10px;
  }
  .kpi-title { color: #a0aec0; font-size: 0.85rem; margin-bottom: 4px; }
  .kpi-value { color: #ffffff; font-size: 1.7rem; font-weight: 700; }
  .kpi-delta { color: #48bb78; font-size: 0.8rem; margin-top: 4px; }
  .section-header {
    color: #e94560;
    font-size: 1.2rem;
    font-weight: 700;
    border-bottom: 2px solid #e94560;
    padding-bottom: 8px;
    margin: 20px 0 14px 0;
  }
  .fraud-badge {
    background: #e94560; color: white; border-radius: 4px;
    padding: 2px 8px; font-size: 0.75rem; font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)

BASE_DIR  = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"

# ─── Auto-setup: run all models on first launch (Streamlit Cloud) ─────────────
from setup import run_all_setup
with st.spinner("⚙️ First-time setup: training models, please wait ~1 min..."):
    run_all_setup()


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file, parse_dates=["Date"])
    return df


def kpi_card(title: str, value: str, delta: str = ""):
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-title">{title}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-delta">{delta}</div>
    </div>
    """, unsafe_allow_html=True)


def fmt_inr(v: float) -> str:
    if v >= 1e7:
        return f"₹{v/1e7:.2f} Cr"
    if v >= 1e5:
        return f"₹{v/1e5:.1f} L"
    return f"₹{v:,.0f}"


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/fuel-pump.png", width=64)
    st.title("⛽ Fuel Management")
    st.markdown("**AI-Powered Analytics Dashboard**")
    st.divider()

    uploaded = st.file_uploader("Upload fuel.csv", type=["csv"])
    if uploaded:
        df_raw = load_data(uploaded)
        st.success(f"✅ {len(df_raw):,} records loaded")
    elif (BASE_DIR / "data" / "fuel.csv").exists():
        df_raw = load_data(BASE_DIR / "data" / "fuel.csv")
        st.info(f"📂 Using local dataset ({len(df_raw):,} records)")
    else:
        st.warning("⚠️ No dataset found. Please upload fuel.csv")
        st.stop()

    st.divider()
    st.markdown("**Filters**")
    years = sorted(df_raw["Date"].dt.year.unique())
    sel_year = st.multiselect("Year", years, default=years)

    cities = sorted(df_raw["City"].unique())
    sel_city = st.multiselect("City", cities, default=cities)

    fuel_types = sorted(df_raw["Fuel_Type"].unique())
    sel_fuel = st.multiselect("Fuel Type", fuel_types, default=fuel_types)

    forecast_months = st.selectbox("Forecast Horizon", [3, 6, 12], index=2)

# ── Filter ────────────────────────────────────────────────────────────────────
df = df_raw[
    df_raw["Date"].dt.year.isin(sel_year) &
    df_raw["City"].isin(sel_city) &
    df_raw["Fuel_Type"].isin(sel_fuel)
].copy()

if df.empty:
    st.error("No data matches the selected filters.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🇮🇳 AI-Powered Smart Fuel Management System")
st.caption(f"Data: {df['Date'].min().date()} → {df['Date'].max().date()}  |  "
           f"{len(df):,} records  |  {df['Vehicle_ID'].nunique()} vehicles")
st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
#  KPI CARDS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 Key Performance Indicators</div>', unsafe_allow_html=True)

total_cost    = df["Total_Fuel_Cost_INR"].sum()
total_fuel    = df["Fuel_Quantity_Liters"].sum()
total_dist    = df["Distance_KM"].sum()
avg_mileage   = df["Mileage_KMPL"].mean()
total_maint   = df["Maintenance_Cost_INR"].sum()
fraud_count   = int(df["Is_Fraud"].sum()) if "Is_Fraud" in df.columns else "N/A"
cost_per_km   = total_cost / total_dist if total_dist else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: kpi_card("Total Fuel Cost",     fmt_inr(total_cost),           "Fleet-wide spend")
with c2: kpi_card("Total Fuel Used",     f"{total_fuel:,.0f} L",        "Litres consumed")
with c3: kpi_card("Total Distance",      f"{total_dist:,.0f} KM",       "Fleet kilometres")
with c4: kpi_card("Avg Mileage",         f"{avg_mileage:.2f} KMPL",     "Fleet average")
with c5: kpi_card("Maintenance Cost",    fmt_inr(total_maint),          "Total spend")
with c6: kpi_card("Fraud Alerts 🚨",    str(fraud_count),               "Anomalies detected")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHARTS – EDA
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📈 Exploratory Analytics</div>', unsafe_allow_html=True)

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    monthly = (
        df.set_index("Date")
        .resample("ME")["Total_Fuel_Cost_INR"]
        .sum()
        .reset_index()
    )
    monthly.columns = ["Month", "Cost_INR"]
    fig = px.line(monthly, x="Month", y="Cost_INR",
                  title="📅 Monthly Fuel Cost Trend (₹ INR)",
                  markers=True,
                  color_discrete_sequence=["#e94560"])
    fig.update_yaxes(tickprefix="₹", tickformat=",")
    st.plotly_chart(fig, use_container_width=True)

with row1_col2:
    state_cost = df.groupby("State")["Total_Fuel_Cost_INR"].sum().reset_index()
    fig2 = px.bar(state_cost.sort_values("Total_Fuel_Cost_INR", ascending=False),
                  x="State", y="Total_Fuel_Cost_INR",
                  title="🗺️ State-wise Fuel Spending (₹ INR)",
                  color="Total_Fuel_Cost_INR",
                  color_continuous_scale="OrRd")
    fig2.update_yaxes(tickprefix="₹", tickformat=",")
    st.plotly_chart(fig2, use_container_width=True)

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    vtype_cost = df.groupby("Vehicle_Type")["Fuel_Quantity_Liters"].sum().reset_index()
    fig3 = px.pie(vtype_cost, names="Vehicle_Type", values="Fuel_Quantity_Liters",
                  title="🚗 Fuel Usage by Vehicle Type",
                  hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig3, use_container_width=True)

with row2_col2:
    # Correlation heatmap
    num_cols = ["Fuel_Quantity_Liters", "Fuel_Price_Per_Liter_INR",
                "Total_Fuel_Cost_INR", "Distance_KM",
                "Mileage_KMPL", "Maintenance_Cost_INR"]
    corr = df[num_cols].corr().round(2)
    fig4 = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns, y=corr.index,
        colorscale="RdBu", zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text}",
    ))
    fig4.update_layout(title="🔗 Correlation Heatmap", height=360)
    st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL 1 – FUEL CONSUMPTION PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🤖 Model 1 – Fuel Consumption Prediction</div>',
            unsafe_allow_html=True)

model_path = MODEL_DIR / "fuel_prediction_model.pkl"
if model_path.exists():
    model   = joblib.load(model_path)
    le_dict = joblib.load(MODEL_DIR / "fuel_prediction_encoders.pkl")

    st.subheader("Predict Fuel Needed for a Trip")
    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        p_vtype   = st.selectbox("Vehicle Type", df["Vehicle_Type"].unique())
        p_ftype   = st.selectbox("Fuel Type", df["Fuel_Type"].unique())
    with pc2:
        p_city    = st.selectbox("City", df["City"].unique())
        p_route   = st.selectbox("Route Type", df["Route_Type"].unique())
    with pc3:
        p_dist    = st.number_input("Distance (KM)", 10, 1000, 150)
        p_price   = st.number_input("Fuel Price/Litre (₹)", 70.0, 120.0, 95.0)
    with pc4:
        p_mileage = st.number_input("Expected Mileage (KMPL)", 3.0, 30.0, 12.0)
        p_maint   = st.number_input("Maintenance Cost (₹)", 0, 10000, 1200)

    if st.button("🔮 Predict Fuel Consumption"):
        enc_vtype = le_dict["Vehicle_Type"].transform([p_vtype])[0]
        enc_ftype = le_dict["Fuel_Type"].transform([p_ftype])[0]
        enc_city  = le_dict["City"].transform([p_city])[0]
        enc_state = le_dict["State"].transform([df[df["City"] == p_city]["State"].iloc[0]])[0]
        enc_dept  = le_dict["Department"].transform([df["Department"].iloc[0]])[0]
        enc_route = le_dict["Route_Type"].transform([p_route])[0]

        X_pred = np.array([[1, 1, 1,                 # month, dow, quarter
                            enc_vtype, enc_ftype, enc_city, enc_state,
                            p_price, p_dist, p_mileage, p_maint,
                            enc_dept, enc_route]])

        pred = model.predict(X_pred)[0]
        expected = p_dist / p_mileage
        cost_est  = pred * p_price
        st.success(f"🛢️ Predicted Fuel: **{pred:.2f} L**  |  "
                   f"Expected: **{expected:.2f} L**  |  "
                   f"Estimated Cost: **₹{cost_est:,.2f}**")
else:
    st.info("Run `python train.py` first to generate the prediction model.")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL 2 – FRAUD ALERTS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🚨 Model 2 – Fraud Detection Alerts</div>',
            unsafe_allow_html=True)

fraud_csv = MODEL_DIR / "fraud_alerts.csv"
if fraud_csv.exists():
    fraud_df = pd.read_csv(fraud_csv, parse_dates=["Date"])
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        st.dataframe(
            fraud_df.head(20).style.background_gradient(
                subset=["Anomaly_Score"], cmap="Reds"
            ),
            use_container_width=True, height=300
        )
    with col_f2:
        fig_f = px.scatter(fraud_df, x="Distance_KM", y="Total_Fuel_Cost_INR",
                           color="Anomaly_Score", color_continuous_scale="Reds",
                           title="Fraud – Distance vs Cost",
                           hover_data=["Vehicle_ID", "City"])
        fig_f.update_yaxes(tickprefix="₹", tickformat=",")
        st.plotly_chart(fig_f, use_container_width=True)

    # Download
    st.download_button("⬇️ Download Fraud Report CSV",
                       fraud_df.to_csv(index=False).encode(),
                       "fraud_alerts.csv", "text/csv")
else:
    st.info("Run `python fraud.py` first to generate fraud alerts.")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL 3 – COST FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📉 Model 3 – Monthly Cost Forecast (Prophet)</div>',
            unsafe_allow_html=True)

forecast_csv = MODEL_DIR / "monthly_forecast.csv"
if forecast_csv.exists():
    fc_df = pd.read_csv(forecast_csv)
    fc_df = fc_df.head(forecast_months)

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=fc_df["Month"], y=fc_df["Upper_Bound_INR"],
        fill=None, mode="lines", line_color="rgba(233,69,96,0.2)",
        name="Upper Bound"
    ))
    fig_fc.add_trace(go.Scatter(
        x=fc_df["Month"], y=fc_df["Lower_Bound_INR"],
        fill="tonexty", mode="lines", line_color="rgba(233,69,96,0.2)",
        name="Confidence Band", fillcolor="rgba(233,69,96,0.1)"
    ))
    fig_fc.add_trace(go.Scatter(
        x=fc_df["Month"], y=fc_df["Forecasted_Cost_INR"],
        mode="lines+markers", line=dict(color="#e94560", width=3),
        name="Forecast"
    ))
    fig_fc.update_layout(
        title=f"📅 {forecast_months}-Month Fuel Cost Forecast (₹ INR)",
        xaxis_title="Month", yaxis_title="Fuel Cost (₹)",
        yaxis=dict(tickprefix="₹", tickformat=","),
        height=420,
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.dataframe(fc_df, use_container_width=True)
    with col_t2:
        st.download_button("⬇️ Download Forecast CSV",
                           fc_df.to_csv(index=False).encode(),
                           "monthly_forecast.csv", "text/csv")
else:
    st.info("Run `python forecast.py` first to generate forecasts.")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL 4 – VEHICLE RANKINGS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🏆 Model 4 – Vehicle Efficiency Rankings</div>',
            unsafe_allow_html=True)

ranking_csv = MODEL_DIR / "vehicle_rankings_full.csv"
if not ranking_csv.exists():
    ranking_csv = MODEL_DIR / "vehicle_rankings.csv"

if ranking_csv.exists():
    rank_df = pd.read_csv(ranking_csv)
    top10   = rank_df.head(10)
    bot10   = rank_df.tail(10)

    r1, r2 = st.columns(2)
    with r1:
        st.subheader("🏆 Top 10 Most Efficient")
        fig_r1 = px.bar(top10, x="Efficiency_Score", y="Vehicle_ID",
                        orientation="h", color="Efficiency_Score",
                        color_continuous_scale="Greens",
                        title="Best Performing Vehicles")
        fig_r1.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig_r1, use_container_width=True)

    with r2:
        st.subheader("⚠️ Bottom 10 – Needs Attention")
        fig_r2 = px.bar(bot10.sort_values("Efficiency_Score"),
                        x="Efficiency_Score", y="Vehicle_ID",
                        orientation="h", color="Efficiency_Score",
                        color_continuous_scale="Reds_r",
                        title="Least Efficient Vehicles")
        fig_r2.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig_r2, use_container_width=True)

    display_cols = [c for c in ["Rank", "Vehicle_ID", "Vehicle_Type",
                                "Avg_Mileage_KMPL", "Cost_Per_KM_INR",
                                "Total_Distance_KM", "Efficiency_Score"]
                    if c in rank_df.columns]
    st.dataframe(rank_df[display_cols].head(30), use_container_width=True)

    st.download_button("⬇️ Download Full Rankings CSV",
                       rank_df.to_csv(index=False).encode(),
                       "vehicle_rankings.csv", "text/csv")
else:
    st.info("Run `python ranking.py` first to generate vehicle rankings.")


# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption("🇮🇳 AI-Powered Smart Fuel Management System  |  Built with Python, XGBoost, Prophet & Streamlit")

import streamlit as st
import pandas as pd
import numpy as np
import tempfile
from io import BytesIO
from lightgbm import LGBMRegressor
from lightgbm.callback import early_stopping, log_evaluation

# ================= DATA LOADER ================= #
def load_sme_data_auto(file_path):
    df = pd.read_excel(file_path)
    df = df.dropna(axis=1, how="all")

    sales_candidates = [c for c in df.columns if any(k in c.lower() for k in ["sale","qty","quantity","amount"])]
    sku_candidates = [c for c in df.columns if any(k in c.lower() for k in ["sku","id","product","item","name"])]
    date_candidates = [c for c in df.columns if any(k in c.lower() for k in ["date","month","time"])]

    if not sales_candidates or not sku_candidates or not date_candidates:
        raise ValueError("Required columns not detected")

    df = df.rename(columns={
        sku_candidates[0]:"SKU",
        sales_candidates[0]:"Sales",
        date_candidates[0]:"Date"
    })

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["SKU","Date","Sales"])
    return df

# ================= CLEANING ================= #
def clean_data(df):
    df = df.copy()
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
    df = df.drop_duplicates(["SKU","Date"])
    df["Sales"] = df["Sales"].clip(upper=df["Sales"].quantile(0.99))
    return df.sort_values(["SKU","Date"]).reset_index(drop=True)

# ================= FEATURES ================= #
def feature_engineering(df):
    df = df.copy()
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    df["Month_sin"] = np.sin(2*np.pi*df["Month"]/12)
    df["Month_cos"] = np.cos(2*np.pi*df["Month"]/12)

    for lag in range(1,7):
        df[f"Sales_Lag_{lag}"] = df.groupby("SKU")["Sales"].shift(lag)

    for w in [3,6]:
        df[f"Sales_RollMean_{w}"] = df.groupby("SKU")["Sales"].shift(1).rolling(w).mean()
        df[f"Sales_RollStd_{w}"] = df.groupby("SKU")["Sales"].shift(1).rolling(w).std()

    sku_stats = df.groupby("SKU")["Sales"].agg(
        SKU_Mean="mean",
        SKU_Std="std",
        Zero_Ratio=lambda x: (x==0).mean()
    ).reset_index()

    df = df.merge(sku_stats, on="SKU", how="left")
    return df.fillna(0)

# ================= TIME-SERIES SPLIT (PER-SKU) ================= #
def time_series_split(df, horizon=30):
    train_list, val_list = [], []
    for sku, g in df.groupby("SKU"):
        g = g.sort_values("Date")
        if len(g) > horizon:
            train_list.append(g.iloc[:-horizon])
            val_list.append(g.iloc[-horizon:])
        else:
            train_list.append(g)
    train_df = pd.concat(train_list)
    val_df = pd.concat(val_list) if val_list else train_df
    return train_df, val_df

# ================= QUANTILE MODEL TRAINING ================= #
def train_quantile_model(df, features, quantile=0.5):
    X = df[features]
    y = df["Sales"]
    model = LGBMRegressor(objective="quantile", alpha=quantile, n_estimators=1500, learning_rate=0.03, num_leaves=64)
    model.fit(X, y)
    return model

# ================= INVENTORY OPTIMIZATION ================= #
SERVICE_Z = {0.90:1.28, 0.95:1.65, 0.975:1.96, 0.99:2.33}

def safety_stock(df, lead_time, service_level):
    z = SERVICE_Z[service_level]
    return {
        sku:int(np.ceil(z * g["Sales"].std() * np.sqrt(lead_time)))
        for sku,g in df.groupby("SKU")
    }

def calculate_eoq(daily_demand, order_cost, holding_rate, unit_cost):
    annual_demand = daily_demand * 365
    holding_cost = holding_rate * unit_cost
    if annual_demand <= 0:
        return 0
    return int(np.ceil(np.sqrt((2*annual_demand*order_cost)/holding_cost)))

def reorder_point(daily_demand, lead_time, safety_stock):
    return int(np.ceil(daily_demand*lead_time + safety_stock))

def forecast_and_optimize(df, features, forecast_days, lead_time, service_level, order_cost, holding_rate, unit_cost):
    # Train quantile models
    models = {q: train_quantile_model(df, features, quantile=q) for q in [0.1,0.5,0.9]}

    ss_map = safety_stock(df, lead_time, service_level)
    output = []

    for sku in df["SKU"].unique():
        last = df[df["SKU"]==sku].tail(1)
        preds = {q: models[q].predict(last[features])[0] for q in [0.1,0.5,0.9]}
        daily_pred = preds[0.5]  # P50 as default forecast

        ss = ss_map[sku]
        eoq = calculate_eoq(daily_pred, order_cost, holding_rate, unit_cost)
        rop = reorder_point(daily_pred, lead_time, ss)

        output.append({
            "SKU": sku,
            "P10_Daily": round(preds[0.1],2),
            "P50_Daily": round(preds[0.5],2),
            "P90_Daily": round(preds[0.9],2),
            "Monthly_Forecast": round(preds[0.5]*forecast_days,2),
            "Safety_Stock": ss,
            "EOQ": eoq,
            "Reorder_Point": rop,
            "Recommended_Order_Qty": max(eoq, rop),
            "Service_Level_%": int(service_level*100)
        })
    return pd.DataFrame(output)

# ================= STREAMLIT UI ================= #
st.set_page_config("Demand Forecasting & Inventory Optimizer", layout="wide")
st.title("📦 Demand Forecasting & Inventory Optimization Engine")

file = st.file_uploader("Upload Sales Excel", type=["xlsx"])

forecast_days = st.slider("Forecast Horizon (days)", 7, 30, 30)
lead_time = st.slider("Supplier Lead Time (days)", 1, 60, 14)
service_level = st.selectbox("Target Service Level", [0.90,0.95,0.975,0.99], index=1)

order_cost = st.number_input("Order Cost ($)", 50, 500, 100)
holding_rate = st.slider("Holding Cost Rate", 0.05, 0.5, 0.2)
unit_cost = st.number_input("Unit Cost ($)", 1, 100, 10)

if file and st.button("Run Optimization"):
    with tempfile.NamedTemporaryFile(delete=False,suffix=".xlsx") as tmp:
        tmp.write(file.getbuffer())
        path = tmp.name

    df = load_sme_data_auto(path)
    df = clean_data(df)
    df = feature_engineering(df)

    # Proper per-SKU train/validation split
    train_df, val_df = time_series_split(df, horizon=forecast_days)

    features = [c for c in df.columns if c not in ["SKU","Date","Sales"]]

    result = forecast_and_optimize(
        df, features, forecast_days, lead_time,
        service_level, order_cost, holding_rate, unit_cost
    )

    st.success("Optimization Complete ✅")
    st.dataframe(result)

    buf = BytesIO()
    result.to_excel(buf,index=False)
    buf.seek(0)

    st.download_button(
        "Download Optimized Plan",
        buf,
        "Inventory_Optimization.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

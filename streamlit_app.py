import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Distributor AI Dashboard", layout="wide")
st.title("📦 AI-Driven Distributor Management System")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('demand_forecast_model.pkl')

model = load_model()

# Load or recreate sample data (same synthetic data as your Kaggle notebook)
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 5000
    dates = pd.date_range('2023-01-01', periods=365)
    df = pd.DataFrame({
        'date': np.random.choice(dates, n),
        'retailer_id': np.random.randint(1, 50, n),
        'product_id': np.random.randint(1, 30, n),
        'quantity_ordered': np.random.poisson(20, n),
        'unit_price': np.round(np.random.uniform(5, 100, n), 2),
    })
    df['revenue'] = df['quantity_ordered'] * df['unit_price']
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

    daily = df.groupby(['date', 'product_id']).agg(
        quantity_ordered=('quantity_ordered', 'sum'),
        day_of_week=('day_of_week', 'first'),
        month=('month', 'first'),
        is_weekend=('is_weekend', 'first')
    ).reset_index()
    daily = daily.sort_values(['product_id', 'date'])
    daily['lag_7'] = daily.groupby('product_id')['quantity_ordered'].shift(7)
    daily['rolling_mean_7'] = daily.groupby('product_id')['quantity_ordered'].shift(1).rolling(7).mean()
    daily = daily.dropna()
    return df, daily

df, daily = load_data()
features = ['day_of_week', 'month', 'is_weekend', 'lag_7', 'rolling_mean_7', 'product_id']

# --- Sidebar controls ---
st.sidebar.header("Reorder Calculator")
selected_product = st.sidebar.selectbox("Select Product ID", sorted(daily['product_id'].unique()))
current_stock = st.sidebar.number_input("Current Stock", min_value=0, value=50)
days_of_cover = st.sidebar.slider("Days of Cover", 1, 30, 7)

def recommend_reorder(product_id, current_stock, days_of_cover):
    recent = daily[daily['product_id'] == product_id].tail(1)
    if recent.empty:
        return None
    predicted_daily_demand = model.predict(recent[features])[0]
    reorder_qty = max(0, (predicted_daily_demand * days_of_cover) - current_stock)
    return round(reorder_qty), round(predicted_daily_demand, 1)

# --- Main dashboard ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Daily Order Volume (All Products)")
    daily_sales = df.groupby('date')['quantity_ordered'].sum()
    fig, ax = plt.subplots(figsize=(8,4))
    daily_sales.plot(ax=ax)
    ax.set_xlabel("Date"); ax.set_ylabel("Quantity Ordered")
    st.pyplot(fig)

with col2:
    st.subheader("🏆 Top 10 Products by Volume")
    top_products = df.groupby('product_id')['quantity_ordered'].sum().sort_values(ascending=False).head(10)
    fig2, ax2 = plt.subplots(figsize=(8,4))
    top_products.plot(kind='bar', ax=ax2)
    st.pyplot(fig2)

st.divider()
st.subheader("🔮 Reorder Recommendation")

result = recommend_reorder(selected_product, current_stock, days_of_cover)
if result:
    reorder_qty, predicted_demand = result
    m1, m2 = st.columns(2)
    m1.metric("Predicted Daily Demand", f"{predicted_demand} units")
    m2.metric("Recommended Reorder Quantity", f"{reorder_qty} units")
else:
    st.warning("No data available for this product.")

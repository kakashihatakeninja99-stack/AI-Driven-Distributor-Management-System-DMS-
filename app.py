"""
AI-Driven Distributor Management System (DMS)
Production-ready version: works locally with SQLite and on the cloud with
Postgres (just set DB_URL in secrets), with login, caching, editable
records, and a real demand forecast built from each product's own sales
history.

Run locally with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
from datetime import date
from sklearn.linear_model import LinearRegression
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, Float, String, ForeignKey
)
from sqlalchemy.exc import IntegrityError

# ---------------------------------------------------------------------
# Config — reads secrets if present, otherwise falls back to local SQLite
# ---------------------------------------------------------------------
DB_URL = st.secrets.get("DB_URL", "sqlite:///dms.db")
APP_PASSWORD = st.secrets.get("APP_PASSWORD", None)


# ---------------------------------------------------------------------
# Login gate — set APP_PASSWORD in secrets to turn this on
# ---------------------------------------------------------------------
def check_password():
    if not APP_PASSWORD:
        return True
    if st.session_state.get("authenticated"):
        return True
    st.title("🔒 Distributor Management System")
    pwd = st.text_input("Enter password", type="password")
    if st.button("Login"):
        if pwd == APP_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


# ---------------------------------------------------------------------
# Database setup (SQLAlchemy Core — same code works on SQLite & Postgres)
# ---------------------------------------------------------------------
@st.cache_resource
def get_engine():
    return create_engine(DB_URL)


engine = get_engine()
metadata = MetaData()

companies = Table(
    "companies", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String, unique=True, nullable=False),
)
products = Table(
    "products", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    Column("company_id", Integer, ForeignKey("companies.id"), nullable=False),
)
purchases = Table(
    "purchases", metadata,
    Column("id", Integer, primary_key=True),
    Column("product_id", Integer, ForeignKey("products.id"), nullable=False),
    Column("quantity", Float, nullable=False),
    Column("unit_price", Float, nullable=False),
    Column("purchase_date", String, nullable=False),
)
sales = Table(
    "sales", metadata,
    Column("id", Integer, primary_key=True),
    Column("product_id", Integer, ForeignKey("products.id"), nullable=False),
    Column("quantity", Float, nullable=False),
    Column("unit_price", Float, nullable=False),
    Column("sale_date", String, nullable=False),
)
try:
    metadata.create_all(engine)
except Exception as e:
    st.error("Could not connect to the database. Real error below:")
    st.code(str(e))
    st.stop()


# ---------------------------------------------------------------------
# Data access — cached reads, cache-clearing writes, basic validation
# ---------------------------------------------------------------------
@st.cache_data(ttl=30)
def get_companies():
    return pd.read_sql(companies.select().order_by(companies.c.name), engine)


def add_company(name):
    with engine.begin() as conn:
        try:
            conn.execute(companies.insert().values(name=name))
        except IntegrityError:
            return False, "That company already exists."
    get_companies.clear()
    return True, "Company added."


@st.cache_data(ttl=30)
def get_products():
    query = """
        SELECT p.id, p.name, c.name AS company
        FROM products p JOIN companies c ON p.company_id = c.id
        ORDER BY p.name
    """
    return pd.read_sql(query, engine)


def add_product(name, company_id):
    with engine.begin() as conn:
        try:
            conn.execute(products.insert().values(name=name, company_id=company_id))
        except IntegrityError:
            return False, "That product already exists for this company."
    get_products.clear()
    return True, "Product added."


def record_purchase(product_id, quantity, unit_price, purchase_date):
    if quantity <= 0 or unit_price <= 0:
        return False, "Quantity and price must be greater than zero."
    with engine.begin() as conn:
        conn.execute(purchases.insert().values(
            product_id=product_id, quantity=quantity, unit_price=unit_price,
            purchase_date=purchase_date.isoformat(),
        ))
    get_report.clear()
    get_raw_purchases.clear()
    return True, "Purchase recorded."


def record_sale(product_id, quantity, unit_price, sale_date):
    if quantity <= 0 or unit_price <= 0:
        return False, "Quantity and price must be greater than zero."
    with engine.begin() as conn:
        conn.execute(sales.insert().values(
            product_id=product_id, quantity=quantity, unit_price=unit_price,
            sale_date=sale_date.isoformat(),
        ))
    get_report.clear()
    get_raw_sales.clear()
    return True, "Sale recorded."


@st.cache_data(ttl=30)
def get_raw_purchases():
    query = """
        SELECT pu.id, p.name AS product, c.name AS company,
               pu.quantity, pu.unit_price, pu.purchase_date
        FROM purchases pu
        JOIN products p ON pu.product_id = p.id
        JOIN companies c ON p.company_id = c.id
        ORDER BY pu.purchase_date DESC
    """
    return pd.read_sql(query, engine)


@st.cache_data(ttl=30)
def get_raw_sales():
    query = """
        SELECT s.id, p.name AS product, c.name AS company,
               s.quantity, s.unit_price, s.sale_date
        FROM sales s
        JOIN products p ON s.product_id = p.id
        JOIN companies c ON p.company_id = c.id
        ORDER BY s.sale_date DESC
    """
    return pd.read_sql(query, engine)


def delete_purchase(purchase_id):
    with engine.begin() as conn:
        conn.execute(purchases.delete().where(purchases.c.id == purchase_id))
    get_report.clear()
    get_raw_purchases.clear()


def delete_sale(sale_id):
    with engine.begin() as conn:
        conn.execute(sales.delete().where(sales.c.id == sale_id))
    get_report.clear()
    get_raw_sales.clear()


@st.cache_data(ttl=30)
def get_report():
    p = pd.read_sql("""
        SELECT p.name AS product, c.name AS company,
               SUM(pu.quantity) AS qty_purchased,
               SUM(pu.quantity * pu.unit_price) AS total_purchase_cost
        FROM purchases pu
        JOIN products p ON pu.product_id = p.id
        JOIN companies c ON p.company_id = c.id
        GROUP BY p.id, p.name, c.name
    """, engine)
    s = pd.read_sql("""
        SELECT p.name AS product, c.name AS company,
               SUM(s.quantity) AS qty_sold,
               SUM(s.quantity * s.unit_price) AS total_sale_revenue
        FROM sales s
        JOIN products p ON s.product_id = p.id
        JOIN companies c ON p.company_id = c.id
        GROUP BY p.id, p.name, c.name
    """, engine)
    report = pd.merge(p, s, on=["product", "company"], how="outer").fillna(0)
    report["stock_remaining"] = report["qty_purchased"] - report["qty_sold"]
    report["profit"] = report["total_sale_revenue"] - (
        report["qty_sold"] * (report["total_purchase_cost"] / report["qty_purchased"].replace(0, 1))
    )
    return report


@st.cache_data(ttl=30)
def get_sales_history(product_id):
    df = pd.read_sql(
        sales.select().where(sales.c.product_id == product_id),
        engine,
    )
    if df.empty:
        return df
    daily = df.groupby("sale_date")["quantity"].sum().reset_index()
    daily.columns = ["date", "quantity_sold"]
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="Distributor Management System", layout="wide")

if not check_password():
    st.stop()

st.title("📦 Distributor Management System")

tabs = st.tabs([
    "Add Company", "Add Product", "Record Purchase", "Record Sale",
    "Demand Forecast", "Report & Manage",
])

with tabs[0]:
    st.header("Add a Company")
    name = st.text_input("Company name", key="company_name")
    if st.button("Add Company"):
        if name.strip():
            ok, msg = add_company(name.strip())
            st.success(msg) if ok else st.error(msg)
        else:
            st.warning("Enter a company name.")
    st.subheader("Existing Companies")
    st.dataframe(get_companies(), use_container_width=True)

with tabs[1]:
    st.header("Add a Product")
    companies_df = get_companies()
    if companies_df.empty:
        st.warning("Add a company first.")
    else:
        company_name = st.selectbox("Company", companies_df["name"])
        company_id = int(companies_df[companies_df["name"] == company_name]["id"].iloc[0])
        product_name = st.text_input("Product name", key="product_name")
        if st.button("Add Product"):
            if product_name.strip():
                ok, msg = add_product(product_name.strip(), company_id)
                st.success(msg) if ok else st.error(msg)
            else:
                st.warning("Enter a product name.")
    st.subheader("Existing Products")
    companies_df_filter = get_companies()
    if companies_df_filter.empty:
        st.dataframe(get_products(), use_container_width=True)
    else:
        filter_choice = st.selectbox(
            "Filter by company",
            ["All Companies"] + list(companies_df_filter["name"]),
            key="product_company_filter",
        )
        if filter_choice == "All Companies":
            st.dataframe(get_products(), use_container_width=True)
        else:
            filter_company_id = int(
                companies_df_filter[companies_df_filter["name"] == filter_choice]["id"].iloc[0]
            )
            st.dataframe(get_products(company_id=filter_company_id), use_container_width=True)

with tabs[2]:
    st.header("Record a Purchase (buying stock from a company)")
    products_df = get_products()
    if products_df.empty:
        st.warning("Add a product first.")
    else:
        label = products_df["name"] + " (" + products_df["company"] + ")"
        choice = st.selectbox("Product", label, key="purchase_product")
        product_id = int(products_df.iloc[list(label).index(choice)]["id"])
        quantity = st.number_input("Quantity", min_value=0.0, step=1.0, key="purchase_qty")
        unit_price = st.number_input("Purchase price per unit", min_value=0.0, step=0.01, key="purchase_price")
        purchase_date = st.date_input("Purchase date", value=date.today(), key="purchase_date")
        if st.button("Save Purchase"):
            ok, msg = record_purchase(product_id, quantity, unit_price, purchase_date)
            st.success(msg) if ok else st.error(msg)

with tabs[3]:
    st.header("Record a Sale (selling to a customer)")
    products_df = get_products()
    if products_df.empty:
        st.warning("Add a product first.")
    else:
        label = products_df["name"] + " (" + products_df["company"] + ")"
        choice = st.selectbox("Product", label, key="sale_product")
        product_id = int(products_df.iloc[list(label).index(choice)]["id"])
        quantity = st.number_input("Quantity", min_value=0.0, step=1.0, key="sale_qty")
        unit_price = st.number_input("Sale price per unit", min_value=0.0, step=0.01, key="sale_price")
        sale_date = st.date_input("Sale date", value=date.today(), key="sale_date")
        if st.button("Save Sale"):
            ok, msg = record_sale(product_id, quantity, unit_price, sale_date)
            st.success(msg) if ok else st.error(msg)

with tabs[4]:
    st.header("📈 Demand Forecast (next 7 days)")
    st.caption("Trained live on each product's own sales history — not the sample notebook data.")
    products_df = get_products()
    if products_df.empty:
        st.warning("Add a product first.")
    else:
        label = products_df["name"] + " (" + products_df["company"] + ")"
        choice = st.selectbox("Product", label, key="forecast_product")
        product_id = int(products_df.iloc[list(label).index(choice)]["id"])
        history = get_sales_history(product_id)
        if len(history) < 10:
            st.info("Need at least 10 days of recorded sales for this product before forecasting.")
        else:
            history = history.copy()
            history["day_index"] = np.arange(len(history))
            model = LinearRegression().fit(history[["day_index"]], history["quantity_sold"])
            future_idx = np.arange(len(history), len(history) + 7).reshape(-1, 1)
            forecast = model.predict(future_idx).clip(min=0)
            future_dates = pd.date_range(history["date"].iloc[-1] + pd.Timedelta(days=1), periods=7)
            forecast_df = pd.DataFrame({"date": future_dates, "forecast_qty": forecast.round(1)})

            chart_df = pd.concat([
                history[["date", "quantity_sold"]].rename(columns={"quantity_sold": "actual"}).set_index("date"),
                forecast_df.rename(columns={"forecast_qty": "forecast"}).set_index("date"),
            ])
            st.line_chart(chart_df)
            st.dataframe(forecast_df, use_container_width=True)

with tabs[5]:
    st.header("Total Record: Purchases, Sales & Profit")
    report = get_report()
    if report.empty:
        st.info("No data yet. Add purchases and sales first.")
    else:
        report_filter = st.selectbox(
            "Filter by company",
            ["All Companies"] + sorted(report["company"].unique().tolist()),
            key="report_company_filter",
        )
        if report_filter != "All Companies":
            report = report[report["company"] == report_filter]
        st.dataframe(report, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Purchase Cost", f"{report['total_purchase_cost'].sum():,.2f}")
        col2.metric("Total Sale Revenue", f"{report['total_sale_revenue'].sum():,.2f}")
        col3.metric("Total Profit", f"{report['profit'].sum():,.2f}")
        csv = report.to_csv(index=False).encode("utf-8")
        st.download_button("Download report as CSV", csv, "dms_report.csv", "text/csv")

    st.subheader("Manage Purchases")
    raw_purchases = get_raw_purchases()
    st.dataframe(raw_purchases, use_container_width=True)
    if not raw_purchases.empty:
        del_id = st.selectbox("Purchase ID to delete", raw_purchases["id"], key="del_purchase")
        if st.button("Delete Purchase"):
            delete_purchase(int(del_id))
            st.success(f"Purchase #{del_id} deleted.")
            st.rerun()

    st.subheader("Manage Sales")
    raw_sales = get_raw_sales()
    st.dataframe(raw_sales, use_container_width=True)
    if not raw_sales.empty:
        del_id = st.selectbox("Sale ID to delete", raw_sales["id"], key="del_sale")
        if st.button("Delete Sale"):
            delete_sale(int(del_id))
            st.success(f"Sale #{del_id} deleted.")
            st.rerun()

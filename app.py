"""
AI-Driven Distributor Management System (DMS)
A Streamlit app for a wholesale distributor to record purchases (buying
from companies) and sales (selling to customers), and view total records.

Run with:
    streamlit run app.py
"""

import sqlite3
from datetime import date

import pandas as pd
import streamlit as st

DB_PATH = "dms.db"


# ---------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------
def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            company_id INTEGER NOT NULL,
            FOREIGN KEY (company_id) REFERENCES companies(id),
            UNIQUE(name, company_id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS purchases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            quantity REAL NOT NULL,
            unit_price REAL NOT NULL,
            purchase_date TEXT NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            quantity REAL NOT NULL,
            unit_price REAL NOT NULL,
            sale_date TEXT NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------
# Data access helpers
# ---------------------------------------------------------------------
def get_companies():
    conn = get_connection()
    df = pd.read_sql_query("SELECT id, name FROM companies ORDER BY name", conn)
    conn.close()
    return df


def add_company(name):
    conn = get_connection()
    try:
        conn.execute("INSERT INTO companies (name) VALUES (?)", (name,))
        conn.commit()
        return True, "Company added."
    except sqlite3.IntegrityError:
        return False, "That company already exists."
    finally:
        conn.close()


def get_products(company_id=None):
    conn = get_connection()
    if company_id:
        df = pd.read_sql_query(
            "SELECT p.id, p.name, c.name AS company FROM products p "
            "JOIN companies c ON p.company_id = c.id WHERE p.company_id = ? "
            "ORDER BY p.name",
            conn, params=(company_id,),
        )
    else:
        df = pd.read_sql_query(
            "SELECT p.id, p.name, c.name AS company FROM products p "
            "JOIN companies c ON p.company_id = c.id ORDER BY p.name",
            conn,
        )
    conn.close()
    return df


def add_product(name, company_id):
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO products (name, company_id) VALUES (?, ?)",
            (name, company_id),
        )
        conn.commit()
        return True, "Product added."
    except sqlite3.IntegrityError:
        return False, "That product already exists for this company."
    finally:
        conn.close()


def record_purchase(product_id, quantity, unit_price, purchase_date):
    conn = get_connection()
    conn.execute(
        "INSERT INTO purchases (product_id, quantity, unit_price, purchase_date) "
        "VALUES (?, ?, ?, ?)",
        (product_id, quantity, unit_price, purchase_date.isoformat()),
    )
    conn.commit()
    conn.close()


def record_sale(product_id, quantity, unit_price, sale_date):
    conn = get_connection()
    conn.execute(
        "INSERT INTO sales (product_id, quantity, unit_price, sale_date) "
        "VALUES (?, ?, ?, ?)",
        (product_id, quantity, unit_price, sale_date.isoformat()),
    )
    conn.commit()
    conn.close()


def get_report():
    conn = get_connection()
    purchases = pd.read_sql_query("""
        SELECT p.name AS product, c.name AS company,
               SUM(pu.quantity) AS qty_purchased,
               SUM(pu.quantity * pu.unit_price) AS total_purchase_cost
        FROM purchases pu
        JOIN products p ON pu.product_id = p.id
        JOIN companies c ON p.company_id = c.id
        GROUP BY p.id
    """, conn)
    sales = pd.read_sql_query("""
        SELECT p.name AS product, c.name AS company,
               SUM(s.quantity) AS qty_sold,
               SUM(s.quantity * s.unit_price) AS total_sale_revenue
        FROM sales s
        JOIN products p ON s.product_id = p.id
        JOIN companies c ON p.company_id = c.id
        GROUP BY p.id
    """, conn)
    conn.close()

    report = pd.merge(purchases, sales, on=["product", "company"], how="outer").fillna(0)
    report["stock_remaining"] = report["qty_purchased"] - report["qty_sold"]
    report["profit"] = report["total_sale_revenue"] - (
        report["qty_sold"] * (report["total_purchase_cost"] / report["qty_purchased"].replace(0, 1))
    )
    return report


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="Distributor Management System", layout="wide")
init_db()

st.title("📦 Distributor Management System")

menu = st.sidebar.radio(
    "Menu",
    ["Add Company", "Add Product", "Record Purchase", "Record Sale", "View Report"],
)

if menu == "Add Company":
    st.header("Add a Company")
    name = st.text_input("Company name")
    if st.button("Add Company"):
        if name.strip():
            ok, msg = add_company(name.strip())
            st.success(msg) if ok else st.error(msg)
        else:
            st.warning("Enter a company name.")
    st.subheader("Existing Companies")
    st.dataframe(get_companies(), use_container_width=True)

elif menu == "Add Product":
    st.header("Add a Product")
    companies = get_companies()
    if companies.empty:
        st.warning("Add a company first.")
    else:
        company_name = st.selectbox("Company", companies["name"])
        company_id = int(companies[companies["name"] == company_name]["id"].iloc[0])
        product_name = st.text_input("Product name")
        if st.button("Add Product"):
            if product_name.strip():
                ok, msg = add_product(product_name.strip(), company_id)
                st.success(msg) if ok else st.error(msg)
            else:
                st.warning("Enter a product name.")
    st.subheader("Existing Products")
    st.dataframe(get_products(), use_container_width=True)

elif menu == "Record Purchase":
    st.header("Record a Purchase (buying stock from a company)")
    products = get_products()
    if products.empty:
        st.warning("Add a product first.")
    else:
        label = products["name"] + " (" + products["company"] + ")"
        choice = st.selectbox("Product", label)
        product_id = int(products.iloc[list(label).index(choice)]["id"])
        quantity = st.number_input("Quantity", min_value=0.0, step=1.0)
        unit_price = st.number_input("Purchase price per unit", min_value=0.0, step=0.01)
        purchase_date = st.date_input("Purchase date", value=date.today())
        if st.button("Save Purchase"):
            record_purchase(product_id, quantity, unit_price, purchase_date)
            st.success("Purchase recorded.")

elif menu == "Record Sale":
    st.header("Record a Sale (selling to a customer)")
    products = get_products()
    if products.empty:
        st.warning("Add a product first.")
    else:
        label = products["name"] + " (" + products["company"] + ")"
        choice = st.selectbox("Product", label)
        product_id = int(products.iloc[list(label).index(choice)]["id"])
        quantity = st.number_input("Quantity", min_value=0.0, step=1.0)
        unit_price = st.number_input("Sale price per unit", min_value=0.0, step=0.01)
        sale_date = st.date_input("Sale date", value=date.today())
        if st.button("Save Sale"):
            record_sale(product_id, quantity, unit_price, sale_date)
            st.success("Sale recorded.")

elif menu == "View Report":
    st.header("Total Record: Purchases, Sales & Profit")
    report = get_report()
    if report.empty:
        st.info("No data yet. Add purchases and sales first.")
    else:
        st.dataframe(report, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Purchase Cost", f"{report['total_purchase_cost'].sum():,.2f}")
        col2.metric("Total Sale Revenue", f"{report['total_sale_revenue'].sum():,.2f}")
        col3.metric("Total Profit", f"{report['profit'].sum():,.2f}")

        csv = report.to_csv(index=False).encode("utf-8")
        st.download_button("Download report as CSV", csv, "dms_report.csv", "text/csv")

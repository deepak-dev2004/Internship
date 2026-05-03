"""Pizza Sales Dashboard project


Business Problem: The primary objective of this project is to analyze pizza sales data to support business decision-making.
Data Collection: pizza_sales_updated.csv (from kaggle)
Data Preparation (Data Cleaning & Transformation)
Exploratory Data Analysis (EDA): EDA is where insights are extracted using visualizations"""



import streamlit as st
import pandas as pd
import plotly.express as px

# ---------- LOAD DATA ----------
df = pd.read_csv("pizza_sales_updated.csv")
df['order_date'] = pd.to_datetime(df['order_date'])

# ---------- KPI CALCULATIONS ----------
total_revenue = df['total_price'].sum()
total_orders = df['order_id'].nunique()
total_pizzas = df['quantity'].sum()
avg_order_value = total_revenue / total_orders
avg_pizza_per_order = total_pizzas / total_orders

# ---------- PAGE CONFIG ----------
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align:center;'>🍕 Pizza Sales Dashboard</h1>", unsafe_allow_html=True)

# ---------- KPI CARDS ----------
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("💰 Revenue", f"{total_revenue:,.0f}")
col2.metric("📦 Orders", total_orders)
col3.metric("🍕 Pizzas Sold", total_pizzas)
col4.metric("💳 Avg Order Value", f"{avg_order_value:.2f}")
col5.metric("📊 Avg Pizzas/Order", f"{avg_pizza_per_order:.2f}")

st.markdown("---")

# ---------- FILTER ----------
category = st.selectbox("Filter by Category", ["All"] + list(df['pizza_category'].unique()))

if category != "All":
    df = df[df['pizza_category'] == category]

# ---------- ROW 1 ----------
col1, col2 = st.columns(2)

# 📊 Daily Trend (Bar)
daily = df.groupby(df['order_date'].dt.day_name())['order_id'].nunique().reset_index()
fig1 = px.bar(daily, x='order_date', y='order_id', title="Daily Orders")
col1.plotly_chart(fig1, use_container_width=True)

# 📈 Monthly Trend (Line)
monthly = df.groupby(df['order_date'].dt.month_name())['order_id'].nunique().reset_index()
fig2 = px.line(monthly, x='order_date', y='order_id', markers=True, title="Monthly Orders")
col2.plotly_chart(fig2, use_container_width=True)

# ---------- ROW 2 ----------
col1, col2, col3 = st.columns(3)

# 🍕 Category Sales (Donut)
cat = df.groupby('pizza_category')['total_price'].sum().reset_index()
fig3 = px.pie(cat, names='pizza_category', values='total_price', hole=0.5, title="Revenue by Category")
col1.plotly_chart(fig3, use_container_width=True)

# 📦 Size Sales (Pie)
size = df.groupby('pizza_size')['total_price'].sum().reset_index()
fig4 = px.pie(size, names='pizza_size', values='total_price', title="Revenue by Size")
col2.plotly_chart(fig4, use_container_width=True)

# 📊 Quantity by Category (Bar)
qty = df.groupby('pizza_category')['quantity'].sum().reset_index()
fig5 = px.bar(qty, x='pizza_category', y='quantity', title="Pizzas Sold by Category")
col3.plotly_chart(fig5, use_container_width=True)

# ---------- ROW 3 ----------
col1, col2 = st.columns(2)

# 🏆 Top 5 (Horizontal Bar)
top5 = df.groupby('pizza_name')['total_price'].sum().nlargest(5).reset_index()
fig6 = px.bar(top5, x='total_price', y='pizza_name', orientation='h', title="Top 5 Pizzas")
col1.plotly_chart(fig6, use_container_width=True)

# 📉 Bottom 5 (Horizontal Bar)
bottom5 = df.groupby('pizza_name')['total_price'].sum().nsmallest(5).reset_index()
fig7 = px.bar(bottom5, x='total_price', y='pizza_name', orientation='h', title="Bottom 5 Pizzas")
col2.plotly_chart(fig7, use_container_width=True)

# ---------- ROW 4 ----------
# 📊 Heatmap-style (Orders per Month)
heat = df.groupby([df['order_date'].dt.month_name(), df['pizza_category']])['order_id'].nunique().reset_index()
fig8 = px.density_heatmap(heat, x='order_date', y='pizza_category', z='order_id',
                         title="Orders Heatmap (Month vs Category)")
st.plotly_chart(fig8, use_container_width=True)
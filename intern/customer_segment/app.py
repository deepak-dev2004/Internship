import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("customers.csv")

st.title("🧑‍💼 Customer Segmentation Dashboard")

st.subheader("Dataset Preview")
st.write(df.head())

# Feature selection
features = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Elbow method
st.subheader("Elbow Method")

inertia = []
K = range(1, len(df)+1)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(K, inertia, marker='o')
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("Inertia")
st.pyplot(fig)

# Choose clusters
k = st.slider(
    "Select number of clusters",
    min_value=2,
    max_value=len(df),   # 🔥 key fix
    value=min(5, len(df))
)

# KMeans model
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

st.subheader("Clustered Data")
st.write(df.head())

# Visualization
st.subheader("Customer Segments")

fig2, ax2 = plt.subplots()
scatter = ax2.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster']
)

ax2.set_xlabel("Annual Income")
ax2.set_ylabel("Spending Score")
st.pyplot(fig2)
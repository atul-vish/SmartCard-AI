import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# =========================
# 🎨 PAGE CONFIG
# =========================
st.set_page_config(page_title="SmartCard-AI Dashboard", page_icon="🧠", layout="wide")

# =========================
# 🎨 CUSTOM UI
# =========================
st.markdown(
    """
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
.card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}
h1, h2, h3 {
    color: #f8fafc;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# 🏠 HEADER
# =========================
st.title("🧠 SmartCard-AI (E-Commerce Clustering System)")
st.caption("Turn raw customer data into actionable insights")

# =========================
# 📂 DATA SOURCE
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📂 Data Source")

option = st.radio(
    "Choose Data Source:", ["Default Dataset", "Upload CSV"], horizontal=True
)

if option == "Default Dataset":
    df = pd.read_csv("smartcart_customers.csv")
    st.success("✅ Using default dataset")
else:
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Custom dataset loaded")
    else:
        st.warning("⚠️ Please upload a CSV file")
        st.stop()

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# 📊 DATA PREVIEW
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Dataset Preview")

col1, col2 = st.columns(2)

st.dataframe(df.head(), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# =========================
# 🔧 PREPROCESSING (IMPORTANT FIX)
# =========================
df_encoded = df.copy()
le = LabelEncoder()

for col in df_encoded.columns:
    if df_encoded[col].dtype == "object":
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

X = df_encoded.select_dtypes(include=["int64", "float64"])
X = X.fillna(X.mean())

# =========================
# 📉 ELBOW METHOD
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📉 Optimal Cluster Detection")

wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    wcss.append(km.inertia_)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(range(1, 11), wcss, marker="o")
    ax1.set_title("Elbow Method", fontsize=10)
    ax1.set_xlabel("K", fontsize=8)
    ax1.set_ylabel("WCSS", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig1)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# 🤖 MODEL
# =========================
k = 3
model = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = model.fit_predict(X)

# =========================
# 🧠 SEGMENT NAMING
# =========================
cluster_mean = df.groupby("Cluster").mean(numeric_only=True)
sorted_clusters = cluster_mean.mean(axis=1).sort_values()

labels = {}
for i, cluster in enumerate(sorted_clusters.index):
    if i == 0:
        labels[cluster] = "Low Value"
    elif i == len(sorted_clusters) - 1:
        labels[cluster] = "High Value"
    else:
        labels[cluster] = "Medium Value"

df["Segment"] = df["Cluster"].map(labels)

# =========================
# 📊 KPI
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", len(df))
col2.metric("High Value", (df["Segment"] == "High Value").sum())
col3.metric("Low Value", (df["Segment"] == "Low Value").sum())
st.markdown("</div>", unsafe_allow_html=True)

# =========================
# 📍 PCA VISUALIZATION
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📍 Customer Segments Map")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    fig2, ax2 = plt.subplots(figsize=(5, 3))

    for segment in df["Segment"].unique():
        idx = df["Segment"] == segment
        ax2.scatter(X_pca[idx, 0], X_pca[idx, 1], label=segment, s=30)

    ax2.legend(fontsize=8)
    ax2.set_title("Segmentation Map", fontsize=10)
    ax2.set_xlabel("PCA 1", fontsize=8)
    ax2.set_ylabel("PCA 2", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig2)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# 📈 INSIGHTS
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📈 Business Insights")

col1, col2 = st.columns(2)

with col1:
    st.bar_chart(df["Segment"].value_counts())

with col2:
    st.dataframe(cluster_mean)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# 📥 DOWNLOAD
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)

csv = df.to_csv(index=False)
st.download_button("📥 Download Segmented Data", csv, "segmented_customers.csv")

st.markdown("</div>", unsafe_allow_html=True)

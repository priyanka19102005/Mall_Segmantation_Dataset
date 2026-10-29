import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# -------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="Mall Customer Segmentation & K-Means Clustering",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #BE185D;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #FDF2F8;
        border-left: 4px solid #DB2777;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Data Loading
# -------------------------------------------------------------
@st.cache_data
def load_data():
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "Mall_Customers.csv"),
        "Mall_Customers.csv",
        os.path.join(os.getcwd(), "Mall_Customers.csv")
    ]
    file_path = None
    for p in possible_paths:
        if os.path.exists(p):
            file_path = p
            break

    if file_path is None:
        raise FileNotFoundError("Mall_Customers.csv not found in the repository.")

    df = pd.read_csv(file_path)
    # Ensure standard column names
    col_map = {
        "Annual Income (k$)": "Annual_Income_k$",
        "Spending Score (1-100)": "Spending_Score"
    }
    df = df.rename(columns=lambda c: col_map.get(c, c.strip()))
    return df


try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# -------------------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=70)
st.sidebar.title("Clustering Controls")

k_clusters = st.sidebar.slider(
    "Select Number of Clusters (K)",
    min_value=2,
    max_value=10,
    value=5,
    help="Default is K=5 based on the optimal Elbow Method."
)

features_selected = st.sidebar.multiselect(
    "Select Features for K-Means Clustering",
    options=["Annual_Income_k$", "Spending_Score", "Age"],
    default=["Annual_Income_k$", "Spending_Score"],
    help="Default uses Annual Income and Spending Score."
)

if len(features_selected) < 2:
    st.sidebar.warning("Please select at least 2 features for clustering.")
    features_selected = ["Annual_Income_k$", "Spending_Score"]

# -------------------------------------------------------------
# K-Means Model Execution
# -------------------------------------------------------------
X_cluster = df_raw[features_selected].copy()
kmeans = KMeans(n_clusters=k_clusters, init="k-means++", random_state=42, n_init=10)
df = df_raw.copy()
df["Cluster"] = kmeans.fit_predict(X_cluster)
centroids = kmeans.cluster_centers_

# Persona helper for standard 2-feature clustering
def get_persona(income_center, spend_center):
    if income_center > 70 and spend_center > 65:
        return "💎 VIP / High Spenders (Target for premium offers & luxury campaigns)"
    elif income_center > 70 and spend_center < 40:
        return "💰 Careful / High Income, Low Spenders (Target with high-value promotions & incentives)"
    elif income_center < 40 and spend_center > 65:
        return "⚡ Impulsive / Low Income, High Spenders (Target with discounts, loyalty rewards & flash sales)"
    elif income_center < 40 and spend_center < 40:
        return "🏷️ Budget / Low Income, Low Spenders (Target with budget-friendly deals)"
    else:
        return "⚖️ Standard / Balanced (Consistent shoppers with balanced spending)"

# -------------------------------------------------------------
# Main Header
# -------------------------------------------------------------
st.markdown('<div class="main-header">🛍️ Mall Customer Segmentation Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Unsupervised Machine Learning with K-Means to identify distinct customer groups and optimize marketing campaigns.</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Interactive Clusters",
    "📉 Elbow Method Analysis",
    "🔮 Real-Time Customer Predictor",
    "📋 Dataset Explorer"
])

# =============================================================
# TAB 1: Visualizations & Clusters
# =============================================================
with tab1:
    st.subheader(f"Customer Segments (K = {k_clusters})")

    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("##### 2D Customer Clusters & Centroids")
        fig_2d = px.scatter(
            df,
            x=features_selected[0],
            y=features_selected[1],
            color="Cluster",
            hover_data=["CustomerID", "Gender", "Age"],
            color_continuous_scale=px.colors.qualitative.Bold,
            title=f"{features_selected[0]} vs {features_selected[1]}",
            template="plotly_white"
        )
        # Add centroids
        fig_2d.add_trace(
            go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode="markers",
                marker=dict(size=14, color="black", symbol="x", line=dict(width=2, color="white")),
                name="Centroids"
            )
        )
        fig_2d.update_layout(height=450)
        st.plotly_chart(fig_2d, use_container_width=True)

    with c2:
        st.markdown("##### Cluster Distribution & Summary")
        cluster_summary = df.groupby("Cluster")[features_selected].mean().round(1)
        cluster_counts = df["Cluster"].value_counts().sort_index()
        cluster_summary["Customer Count"] = cluster_counts
        cluster_summary["Percentage"] = (cluster_counts / len(df) * 100).round(1).astype(str) + "%"

        st.dataframe(cluster_summary, use_container_width=True)

        st.markdown("##### Marketing Personas")
        for i, center in enumerate(centroids):
            if "Annual_Income_k$" in features_selected and "Spending_Score" in features_selected:
                inc_idx = features_selected.index("Annual_Income_k$")
                spd_idx = features_selected.index("Spending_Score")
                persona = get_persona(center[inc_idx], center[spd_idx])
            else:
                persona = f"Cluster {i} segment"
            st.markdown(f"- **Cluster {i}** ({cluster_counts.get(i, 0)} customers): {persona}")

    if len(features_selected) == 3:
        st.markdown("---")
        st.markdown("##### 3D Interactive Cluster Projection")
        fig_3d = px.scatter_3d(
            df,
            x=features_selected[0],
            y=features_selected[1],
            z=features_selected[2],
            color="Cluster",
            hover_data=["CustomerID", "Gender"],
            color_continuous_scale=px.colors.qualitative.Prism,
            title="3D Cluster Scatter Plot",
            template="plotly_white"
        )
        fig_3d.update_layout(height=550)
        st.plotly_chart(fig_3d, use_container_width=True)


# =============================================================
# TAB 2: Elbow Method
# =============================================================
with tab2:
    st.subheader("Optimal K Determination using the Elbow Method")
    st.write("The Elbow Method computes the Within-Cluster Sum of Squares (WCSS) for varying cluster values of $K$. The 'elbow' point signifies the optimal number of clusters.")

    wcss = []
    k_range = list(range(1, 11))
    for k in k_range:
        km_test = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        km_test.fit(X_cluster)
        wcss.append(km_test.inertia_)

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.markdown("##### Dynamic Computed Elbow Curve")
        fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4))
        ax_elbow.plot(k_range, wcss, marker="o", color="#BE185D", linewidth=2, markersize=7)
        ax_elbow.set_xlabel("Number of Clusters (K)")
        ax_elbow.set_ylabel("WCSS (Inertia)")
        ax_elbow.set_title("WCSS vs Number of Clusters")
        ax_elbow.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig_elbow)
        plt.close(fig_elbow)

    with col_e2:
        st.markdown("##### Pre-Generated Analysis in Repository")
        if os.path.exists("Elbow.png"):
            st.image("Elbow.png", caption="Repository Elbow.png", use_container_width=True)
        elif os.path.exists(os.path.join(os.path.dirname(__file__), "Elbow.png")):
            st.image(os.path.join(os.path.dirname(__file__), "Elbow.png"), caption="Repository Elbow.png", use_container_width=True)
        else:
            st.info("Elbow.png loaded dynamically from dataset.")


# =============================================================
# TAB 3: Predict New Customer
# =============================================================
with tab3:
    st.subheader("Assign a New Customer to a Segment")
    st.write("Enter details for a prospective customer to identify their segment and ideal marketing strategy:")

    with st.form("new_customer_form"):
        col_in1, col_in2, col_in3 = st.columns(3)

        with col_in1:
            in_gender = st.selectbox("Gender", options=["Female", "Male"])
            in_age = st.number_input("Age", min_value=15, max_value=100, value=30)

        with col_in2:
            in_income = st.number_input("Annual Income (k$)", min_value=5, max_value=200, value=60)

        with col_in3:
            in_spend = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

        predict_btn = st.form_submit_button("🔮 Segment Customer", use_container_width=True)

    if predict_btn:
        input_dict = {
            "Annual_Income_k$": in_income,
            "Spending_Score": in_spend,
            "Age": in_age
        }
        input_features = [input_dict[f] for f in features_selected]
        predicted_cluster = kmeans.predict([input_features])[0]

        st.markdown("### 🏷️ Segmentation Result")
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.success(f"### Assigned Segment: **Cluster {predicted_cluster}**")
            center = centroids[predicted_cluster]
            if "Annual_Income_k$" in features_selected and "Spending_Score" in features_selected:
                inc_idx = features_selected.index("Annual_Income_k$")
                spd_idx = features_selected.index("Spending_Score")
                st.write(get_persona(center[inc_idx], center[spd_idx]))

        with res_col2:
            st.metric(
                label="Segment Size in Database",
                value=f"{cluster_counts.get(predicted_cluster, 0)} customers ({cluster_counts.get(predicted_cluster, 0)/len(df)*100:.1f}%)"
            )


# =============================================================
# TAB 4: Dataset Explorer
# =============================================================
with tab4:
    st.subheader("Mall Customers Dataset")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Customers", len(df))
    col_m2.metric("Average Age", f"{df['Age'].mean():.1f} yrs")
    col_m3.metric("Avg Annual Income", f"${df['Annual_Income_k$'].mean():.1f}k")
    col_m4.metric("Avg Spending Score", f"{df['Spending_Score'].mean():.1f}/100")

    st.dataframe(df, use_container_width=True)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Segmented Customers CSV",
        data=csv_data,
        file_name="mall_customers_segmented.csv",
        mime="text/csv",
        use_container_width=True
    )

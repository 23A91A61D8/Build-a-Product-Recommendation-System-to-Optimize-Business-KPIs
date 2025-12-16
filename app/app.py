# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Business-Optimized Recommendation Dashboard",
    layout="wide"
)

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

INTERACTIONS_PATH = os.path.join(DATA_DIR, "interactions.csv")
PRODUCTS_PATH = os.path.join(DATA_DIR, "products.csv")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    interactions = pd.read_csv(INTERACTIONS_PATH)
    products = pd.read_csv(PRODUCTS_PATH)
    return interactions, products

interactions, products = load_data()

# =========================
# PREPROCESSING
# =========================
interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])

rating_map = {
    "click": 1,
    "add_to_cart": 3,
    "purchase": 5
}
interactions["rating"] = interactions["event_type"].map(rating_map)

# =========================
# BASELINE MODEL (POPULARITY)
# =========================
popular_products = (
    interactions[interactions["event_type"] == "purchase"]
    .groupby("product_id")
    .size()
    .sort_values(ascending=False)
)

def recommend_popular(top_n=10):
    return popular_products.head(top_n).index.tolist()

# =========================
# COLLABORATIVE FILTERING
# =========================
user_item_matrix = interactions.pivot_table(
    index="user_id",
    columns="product_id",
    values="rating",
    aggfunc="mean"
).fillna(0)

user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

def recommend_collaborative(user_id, top_n=10):
    if user_id not in user_item_matrix.index:
        return recommend_popular(top_n)

    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]
    scores = user_item_matrix.loc[similar_users.index].T.dot(similar_users)
    return scores.sort_values(ascending=False).head(top_n).index.tolist()

# =========================
# BUSINESS RE-RANKING
# =========================
product_margin = products.set_index("product_id")["profit_margin"]

def rerank_with_business(product_list, alpha=0.7):
    scored = []
    for pid in product_list:
        margin = product_margin.get(pid, 0)
        final_score = alpha * 1 + (1 - alpha) * margin
        scored.append((pid, final_score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in scored]

# =========================
# FINAL RECOMMENDATION
# =========================
def recommend_final(user_id, top_n=5):
    if user_id not in interactions["user_id"].values:
        return recommend_popular(top_n)
    candidates = recommend_collaborative(user_id, top_n=20)
    return rerank_with_business(candidates)[:top_n]

# =========================
# DASHBOARD UI
# =========================
st.title("🛒 Business-Optimized Product Recommendation Dashboard")
st.caption("Dynamic personalization with measurable business impact")

# ---------- PLATFORM KPIs ----------
st.subheader("📊 Platform KPIs")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Total Users", interactions["user_id"].nunique())
with c2:
    st.metric("Total Products", products["product_id"].nunique())
with c3:
    st.metric("Total Interactions", len(interactions))
with c4:
    st.metric("Avg Profit Margin", f"{products['profit_margin'].mean():.2f}")

st.divider()

# ---------- USER SELECTION ----------
st.subheader("🎯 Personalized User Analysis")

user_id = st.number_input(
    "Select User ID",
    min_value=int(interactions["user_id"].min()),
    max_value=int(interactions["user_id"].max()),
    step=1
)

user_data = interactions[interactions["user_id"] == user_id]

# ---------- USER SEGMENT ----------
if user_data.empty:
    user_type = "🆕 New User (Cold Start)"
elif len(user_data) < 10:
    user_type = "🙂 Active User"
else:
    user_type = "🔥 Power User"

st.info(f"User Segment: **{user_type}**")

# ---------- USER ACTIVITY OVER TIME ----------
st.subheader("📈 User Activity Over Time")

if not user_data.empty:
    user_ts = (
        user_data
        .set_index("timestamp")
        .resample("D")
        .size()
    )
    st.line_chart(user_ts)
else:
    st.warning("No activity history for this user.")

# ---------- USER EVENT DISTRIBUTION ----------
st.subheader("📊 User Interaction Type Distribution")

if not user_data.empty:
    st.bar_chart(user_data["event_type"].value_counts())

# ---------- USER CATEGORY DISTRIBUTION ----------
st.subheader("📊 Categories Interacted by Selected User")

if not user_data.empty:
    user_categories = (
        user_data
        .merge(products, on="product_id")
        .groupby("category")
        .size()
    )
    st.bar_chart(user_categories)

st.divider()

# ---------- RECOMMENDATIONS ----------
if st.button("🚀 Generate Recommendations"):
    recs = recommend_final(user_id, top_n=5)

    rec_df = products[
        products["product_id"].isin(recs)
    ][["product_id", "category", "price", "profit_margin"]]

    rec_df = rec_df.sort_values("profit_margin", ascending=False)

    st.subheader("🛍️ Recommended Products")
    st.dataframe(rec_df, use_container_width=True)

    # ---------- BUSINESS METRICS ----------
    st.success(
        f"📈 Avg Profit Margin of Recommendations: **{rec_df['profit_margin'].mean():.2f}**"
    )

    # Profit trend
    st.subheader("📉 Profit Margin Trend (Recommendations)")
    st.line_chart(rec_df["profit_margin"].reset_index(drop=True))

    # Comparison chart
    st.subheader("📊 User vs Platform Profit Comparison")
    comparison_df = pd.DataFrame({
        "User Recommendations": [rec_df["profit_margin"].mean()],
        "Platform Average": [products["profit_margin"].mean()]
    }).T
    st.bar_chart(comparison_df)

st.divider()

# ---------- BUSINESS INSIGHTS ----------
st.subheader("💼 Business Insights")
st.markdown("""
- ✔ User behavior captured using **collaborative filtering**
- ✔ Business re-ranking prioritizes **high-margin products**
- ✔ Cold-start users handled via **popularity fallback**
- ✔ All charts dynamically update with **selected User ID**
""")

st.success("✅ This dashboard demonstrates real-world data science + business value.")

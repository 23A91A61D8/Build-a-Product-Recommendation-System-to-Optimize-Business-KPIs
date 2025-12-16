import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Product Recommendation System",
    layout="centered"
)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    interactions = pd.read_csv("../data/interactions.csv")
    products = pd.read_csv("../data/products.csv")
    return interactions, products

interactions, products = load_data()

# -------------------------------------------------
# PREPROCESSING
# -------------------------------------------------
# Map events to implicit ratings
rating_map = {
    "click": 1,
    "add_to_cart": 3,
    "purchase": 5
}
interactions["rating"] = interactions["event_type"].map(rating_map)

# -------------------------------------------------
# BASELINE: POPULARITY MODEL
# -------------------------------------------------
popular_products = (
    interactions[interactions["event_type"] == "purchase"]
    .groupby("product_id")
    .size()
    .sort_values(ascending=False)
)

def recommend_popular(top_n=10):
    return popular_products.head(top_n).index.tolist()

# -------------------------------------------------
# COLLABORATIVE FILTERING (USER-BASED)
# -------------------------------------------------
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

# -------------------------------------------------
# BUSINESS-AWARE RE-RANKING
# -------------------------------------------------
product_margin = products.set_index("product_id")["profit_margin"]

def rerank_with_business(product_list, alpha=0.7):
    scored = []
    for pid in product_list:
        margin = product_margin.get(pid, 0)
        final_score = alpha * 1 + (1 - alpha) * margin
        scored.append((pid, final_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in scored]

# -------------------------------------------------
# FINAL RECOMMENDATION PIPELINE
# -------------------------------------------------
def recommend_final(user_id, top_n=5):
    if user_id not in interactions["user_id"].values:
        return recommend_popular(top_n)

    candidates = recommend_collaborative(user_id, top_n=20)
    final_recs = rerank_with_business(candidates)
    return final_recs[:top_n]

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.title("🛒 Product Recommendation System")
st.write("Personalized & Business-Optimized Recommendations")

user_id = st.number_input(
    "Enter User ID",
    min_value=int(interactions["user_id"].min()),
    max_value=int(interactions["user_id"].max()),
    step=1
)

if st.button("Get Recommendations"):
    recommendations = recommend_final(user_id, top_n=5)

    st.subheader("Recommended Products")

    result_df = products[
        products["product_id"].isin(recommendations)
    ][["product_id", "category", "price", "profit_margin"]]

    st.dataframe(result_df)

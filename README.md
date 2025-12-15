# Build-a-Product-Recommendation-System-to-Optimize-Business-KPIs


# Business-Optimized Product Recommendation System

## Project Overview
This project implements an end-to-end **product recommendation system** for an e-commerce platform.  
The system personalizes product recommendations for users while explicitly optimizing **business KPIs such as profit margin**, rather than focusing only on recommendation accuracy.

The solution combines collaborative filtering with a business-aware re-ranking strategy and presents results through an interactive dashboard.

---

## Project Objectives
- Build a personalized recommendation engine for e-commerce users  
- Optimize recommendations for **business value (profit margin)**  
- Handle both **warm-start** and **cold-start** users  
- Evaluate recommendation quality using standard ranking metrics  
- Provide insights through an interactive Streamlit dashboard  

---

## Methodology

### 1. Data Integration & Analysis
- Integrated three data sources:
  - User interactions (clicks, add-to-cart, purchases)
  - Product catalog (category, price, profit margin)
  - User profiles
- Performed exploratory data analysis to study:
  - Long-tail user behavior
  - Interaction distributions
  - Temporal activity patterns
  - User segmentation (new, active, power users)

### 2. Feature Engineering
- Created a user–item interaction matrix
- Engineered user activity features
- Used product-level business attributes (profit margin)
- Implemented a popularity-based fallback for cold-start users

### 3. Modeling & Optimization
- Baseline popularity-based recommendation model
- User-based collaborative filtering using cosine similarity
- Multi-stage recommendation pipeline:
  1. Candidate generation using collaborative filtering
  2. Business-aware re-ranking as a post-processing step

### 4. Business Optimization
A re-ranking strategy was applied using:


- RelevanceScore: collaborative filtering relevance
- BusinessScore: normalized product profit margin
- α controls the balance between personalization and profitability

This strategy increased the average profit margin of recommended products compared to a relevance-only baseline.

### 5. Evaluation
- Used a temporal train–test split to avoid data leakage
- Evaluated using:
  - Precision@K
  - Recall@K
  - NDCG@K
- Simulated business impact by comparing profit margins of baseline vs optimized recommendations

### 6. Visualization & Reporting
- Developed an interactive Streamlit dashboard
- Dashboard features:
  - Platform-level KPIs
  - User-specific dynamic charts
  - Personalized recommendations
  - Business impact comparisons

## Run the Dashboard
streamlit run app.py

### 1. Install dependencies
```bash
pip install -r requirements.txt




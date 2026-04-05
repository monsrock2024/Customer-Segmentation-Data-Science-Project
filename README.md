# 🛒 Customer Segmentation — End-to-End Data Science Project

An end-to-end machine learning pipeline that discovers natural customer segments from marketing campaign data using unsupervised clustering, then converts them into a production-ready classification model deployed as an interactive web app.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-segmentation-data-science-project-anx7pxdpqp8kdzrn8n6.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?logo=streamlit&logoColor=white)

---

## 🎯 Objective

Segment customers based on demographics, spending patterns, and campaign responses — then deploy a real-time classifier that predicts the segment for any new customer.

**Why not just deploy K-Means directly?** Cluster centroids are computed from the full dataset, so re-running on new data can shift assignments entirely. A supervised classifier trained on the discovered segments gives stable predictions, probability scores, and feature importance — none of which K-Means provides out of the box.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | Marketing Campaign Dataset |
| **Records** | 2,240 (2,236 after cleaning) |
| **Raw Features** | 29 |
| **Engineered Features** | 12 |
| **Key Domains** | Demographics, Spending, Campaign Responses, Purchase Channels |

---

## 🔬 Project Pipeline

```
Phase 1: EDA                    Phase 2: Clustering              Phase 3: Classification
┌──────────────────────┐       ┌──────────────────────┐         ┌──────────────────────┐
│ marketing_campaign   │       │ model_ready.csv      │         │ clustered.csv        │
│ .xlsx (2240 × 29)    │──────▶│ (2236 × 12)          │────────▶│ (2236 × 14)          │
│                      │       │                      │         │                      │
│ • Missing values     │       │ • K-Means (chosen)   │         │ • Logistic Regression│
│ • Outlier removal    │       │ • Agglomerative      │         │ • Random Forest      │
│ • Feature engineering│       │ • DBSCAN             │         │ • XGBoost            │
│   (29 → 12 features)│       │ • K=4 (business fit) │         │ • SVM                │
└──────────────────────┘       └──────────────────────┘         └──────────────────────┘
                                                                         │
                                                                         ▼
                                                                ┌──────────────────────┐
                                                                │ Streamlit App        │
                                                                │ classifier.pkl       │
                                                                │ scaler.pkl           │
                                                                │ encoder.pkl          │
                                                                └──────────────────────┘
```

---

## 👥 Discovered Segments

| Segment | Share | Profile |
|---|---|---|
| **💎 Premium Loyal** | ~10% | High income, high spend, responsive to campaigns |
| **⭐ High-Value** | ~28% | Above-average income and spend, moderate engagement |
| **🏷️ Deal-Seeking Parents** | ~18% | Families with dependents, drawn to deals and discounts |
| **💰 Budget-Conscious** | ~44% | Lower income/spend, largest group, price-sensitive |

> **Labeling note:** The "Deal-Seeking Parents" segment was initially auto-labeled as "Average Mainstream" but was relabeled after analysis revealed it had the highest average dependents (>1.2× overall mean) and the highest deal purchases (>1.5× overall mean).

---

## 🏆 Model Performance

**Final Model:** Logistic Regression — **99.55% test accuracy**

Four models were compared in a model-agnostic pipeline (any model can win; all downstream steps adapt automatically):

| Model | Test Accuracy |
|---|---|
| Logistic Regression | **99.55%** |
| Random Forest | 99.33% |
| XGBoost / Gradient Boosting | 99.11% |
| SVM | 99.33% |

**Why Logistic Regression?** Despite all models performing similarly, Logistic Regression was selected for its interpretability, faster inference, and lower deployment overhead — ideal for a production Streamlit app.

---

## 🚀 Live Demo

👉 **[Launch the Streamlit App](https://customer-segmentation-data-science-project-anx7pxdpqp8kdzrn8n6.streamlit.app/)**

Enter customer attributes (income, spending, dependents, etc.) and get an instant segment prediction with confidence scores.

---

## 📁 Repository Structure

```
├── Customer_Segmentation_EDA.ipynb              # Phase 1: Cleaning & feature engineering
├── Customer_Segmentation_Model_Building.ipynb    # Phase 2: Clustering comparison
├── Customer_Segmentation_Classification.ipynb    # Phase 3: Supervised classification
├── app.py                                        # Streamlit deployment app
├── classification_model.pkl                      # Trained Logistic Regression model
├── classification_scaler.pkl                     # StandardScaler (fit on training split only)
├── label_encoder.pkl                             # LabelEncoder for segment names
├── marketing_campaign.xlsx                       # Raw dataset
├── requirements.txt                              # Python dependencies
└── README.md
```

---

## ⚙️ Key Design Decisions

1. **K=4 over K=2** — K=2 had a higher silhouette score (0.22 vs 0.18), but only produced "high vs low spenders." K=4 gives four actionable, business-distinct personas.

2. **StandardScaler re-applied in classification** — The scaler is fit only on the training split in Phase 3, preventing data leakage from the clustering phase.

3. **Clustering → Classification conversion** — Avoids cluster drift, enables confidence scores, and simplifies deployment to a single `.pkl` file.

4. **Median imputation for Income** — Only 24 missing values (~1%), but Income is right-skewed, so median is more representative than mean.

5. **Minimal outlier removal** — Only 4 rows removed (impossible values like income = 666,666 and age > 100), not statistical outliers.

---

## 🛠️ Tech Stack

- **Python** — pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn
- **Clustering** — K-Means, Agglomerative Clustering, DBSCAN
- **Classification** — Logistic Regression, Random Forest, XGBoost, SVM
- **Deployment** — Streamlit, Streamlit Community Cloud
- **Version Control** — Git & GitHub

---

## 🏃 Run Locally

```bash
# Clone the repo
git clone https://github.com/monsrock2024/Customer-Segmentation-Data-Science-Project.git
cd Customer-Segmentation-Data-Science-Project

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

---

## 👨‍💻 Team

**Mentor:** Saigeetha K Panikker

Built as part of the ExcelR Institute Data Science certification program.

---

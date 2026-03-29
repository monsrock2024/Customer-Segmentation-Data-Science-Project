"""
Customer Segmentation — Prediction App
Uses the classifier trained in Customer_Segmentation_Classification.ipynb
Artifacts: classification_scaler.pkl, classification_model.pkl, label_encoder.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main-header {
        font-size: 2rem; font-weight: 700; color: #0d47a1;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem; color: #546e7a; margin-bottom: 1.5rem;
    }
    
    .segment-card {
        padding: 1.2rem; border-radius: 12px; margin: 0.5rem 0;
        border-left: 5px solid; color: #212121;
    }
    .premium-loyal { background: #ffebee; border-color: #e74c3c; }
    .high-value { background: #e8f5e9; border-color: #2ecc71; }
    .deal-seeking { background: #fff3e0; border-color: #f39c12; }
    .budget-conscious { background: #e3f2fd; border-color: #3498db; }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e3f2fd 0%, #ffffff 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS & DATA
# ============================================================
@st.cache_resource
def load_models():
    scaler = joblib.load('classification_scaler.pkl')
    classifier = joblib.load('classification_model.pkl')
    le = joblib.load('label_encoder.pkl')
    with open('feature_columns.json') as f:
        feature_cols = json.load(f)
    with open('segment_profiles.json') as f:
        profiles = json.load(f)
    with open('model_results.json') as f:
        model_results = json.load(f)
    return scaler, classifier, le, feature_cols, profiles, model_results

try:
    scaler, classifier, le, feature_cols, profiles, model_results = load_models()
    models_loaded = True
except Exception as ex:
    models_loaded = False
    load_error = str(ex)

# ============================================================
# SEGMENT CONFIG
# ============================================================
SEGMENT_CONFIG = {
    'Premium Loyal': {
        'color': '#e74c3c', 'icon': '⭐', 'css': 'premium-loyal',
        'strategy': 'Exclusive loyalty programs, early access to new products, premium experiences. Do NOT offer discounts — they don\'t need them.'
    },
    'High-Value': {
        'color': '#2ecc71', 'icon': '💎', 'css': 'high-value',
        'strategy': 'Try different channels (catalog, in-store), personalized recommendations. Current campaigns aren\'t reaching them — only 0.3% responded last time.'
    },
    'Deal-Seeking Parents': {
        'color': '#f39c12', 'icon': '🛒', 'css': 'deal-seeking',
        'strategy': 'Discount codes, bundle deals, family-oriented promotions. Engage through the website — they\'re already browsing 6.7 times/month.'
    },
    'Budget-Conscious': {
        'color': '#3498db', 'icon': '💰', 'css': 'budget-conscious',
        'strategy': 'Minimal marketing investment. Low-cost email newsletters, entry-level products. Focus on retention, not upselling.'
    }
}

# ============================================================
# SIDEBAR — CUSTOMER INPUT
# ============================================================
st.sidebar.markdown("## 🎯 Customer Details")
st.sidebar.markdown("Adjust the sliders to predict a customer's segment.")
st.sidebar.markdown("---")

if models_loaded:
    income = st.sidebar.slider("💵 Annual Income ($)",
        min_value=1000, max_value=200000, value=50000, step=1000, format="$%d")

    recency = st.sidebar.slider("📅 Days Since Last Purchase",
        min_value=0, max_value=100, value=45)

    age = st.sidebar.slider("🎂 Age",
        min_value=18, max_value=90, value=45)

    total_spend = st.sidebar.slider("🛍️ Total Spend ($)",
        min_value=0, max_value=3000, value=500, step=10, format="$%d")

    total_purchases = st.sidebar.slider("📦 Total Purchases",
        min_value=0, max_value=40, value=12)

    total_dependents = st.sidebar.selectbox("👨‍👩‍👧‍👦 Total Dependents",
        options=[0, 1, 2, 3], index=1)

    campaigns_accepted = st.sidebar.selectbox("📢 Campaigns Accepted",
        options=[0, 1, 2, 3, 4, 5], index=0)

    deal_purchases = st.sidebar.slider("🏷️ Deal Purchases",
        min_value=0, max_value=15, value=2)

    web_visits = st.sidebar.slider("🌐 Web Visits / Month",
        min_value=0, max_value=20, value=5)

    response = st.sidebar.selectbox("📩 Last Campaign Response",
        options=["No", "Yes"], index=0)
    response_val = 1 if response == "Yes" else 0

    education = st.sidebar.selectbox("🎓 Education",
        options=["Undergraduate", "Graduate", "Postgraduate"], index=1)
    education_val = {"Undergraduate": 0, "Graduate": 1, "Postgraduate": 2}[education]

    marital = st.sidebar.selectbox("💍 Marital Status",
        options=["Partnered", "Single"], index=0)
    marital_val = {"Partnered": 0, "Single": 1}[marital]

# ============================================================
# MAIN CONTENT
# ============================================================
st.markdown('<p class="main-header">🎯 Customer Segmentation Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict which segment a new customer belongs to — with confidence scores and feature importance</p>', unsafe_allow_html=True)

if not models_loaded:
    st.error(f"⚠️ Could not load model files: {load_error}")
    st.info("Ensure these files are in the same directory as app.py: "
            "classification_scaler.pkl, classification_model.pkl, label_encoder.pkl, "
            "feature_columns.json, segment_profiles.json, model_results.json")
    st.stop()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["🔮 **Predict Segment**", "📊 **Segment Overview**", "ℹ️ **About**"])

# ============================================================
# TAB 1: PREDICTION
# ============================================================
with tab1:
    # Build input
    input_values = [income, recency, age, total_spend, total_purchases,
                    total_dependents, campaigns_accepted, deal_purchases,
                    web_visits, response_val, education_val, marital_val]
    input_data = pd.DataFrame([input_values], columns=feature_cols)

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    predicted_encoded = classifier.predict(input_scaled)[0]
    predicted_label = le.inverse_transform([predicted_encoded])[0]

    # Get probabilities
    if hasattr(classifier, 'predict_proba'):
        probabilities = classifier.predict_proba(input_scaled)[0]
        prob_labels = le.inverse_transform(classifier.classes_)
    else:
        probabilities = None
        prob_labels = None

    seg_config = SEGMENT_CONFIG.get(predicted_label, list(SEGMENT_CONFIG.values())[0])

    # --- Layout ---
    col1, col2 = st.columns([1, 1])

    with col1:
        # Result card
        st.markdown(f"""
        <div class="segment-card {seg_config['css']}">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{seg_config['icon']}</div>
            <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">{predicted_label}</div>
            <div style="font-size: 0.9rem; color: #455a64; line-height: 1.6;">{seg_config['strategy']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Key metrics
        st.markdown("#### Customer Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Income", f"${income:,}")
        m2.metric("Spend", f"${total_spend:,}")
        m3.metric("Dependents", total_dependents)
        m4.metric("Campaigns", campaigns_accepted)

    with col2:
        # Probability chart
        if probabilities is not None:
            prob_df = pd.DataFrame({
                'Segment': prob_labels,
                'Probability': probabilities * 100
            }).sort_values('Probability', ascending=True)

            colors = [SEGMENT_CONFIG.get(s, {}).get('color', '#999') for s in prob_df['Segment']]

            fig = go.Figure(go.Bar(
                x=prob_df['Probability'],
                y=prob_df['Segment'],
                orientation='h',
                marker_color=colors,
                text=[f"{p:.1f}%" for p in prob_df['Probability']],
                textposition='outside',
                textfont=dict(size=14, color='#212121')
            ))
            fig.update_layout(
                title="Confidence Breakdown",
                xaxis_title="Probability (%)",
                xaxis=dict(range=[0, 110]),
                height=300,
                margin=dict(l=0, r=50, t=40, b=30),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Probability scores not available for this model type.")

    # --- Feature Importance ---
    st.markdown("---")
    st.markdown("#### What Drives Segmentation?")

    if hasattr(classifier, 'feature_importances_'):
        importances = pd.Series(classifier.feature_importances_, index=feature_cols)
    elif hasattr(classifier, 'coef_'):
        importances = pd.Series(np.abs(classifier.coef_).mean(axis=0), index=feature_cols)
    else:
        importances = None

    if importances is not None:
        importances = importances.sort_values(ascending=True)
        fig_imp = go.Figure(go.Bar(
            x=importances.values,
            y=importances.index,
            orientation='h',
            marker_color='#1565c0',
            text=[f"{v:.3f}" for v in importances.values],
            textposition='outside'
        ))
        fig_imp.update_layout(
            height=380,
            margin=dict(l=0, r=70, t=10, b=30),
            xaxis_title="Importance Score",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ============================================================
# TAB 2: SEGMENT OVERVIEW
# ============================================================
with tab2:
    st.markdown("#### The 4 Customer Segments")

    cols = st.columns(4)
    for i, (label, config) in enumerate(SEGMENT_CONFIG.items()):
        if label in profiles:
            p = profiles[label]
            with cols[i]:
                st.markdown(f"""
                <div class="segment-card {config['css']}" style="min-height: 280px;">
                    <div style="font-size: 1.8rem;">{config['icon']}</div>
                    <div style="font-size: 1.1rem; font-weight: 700; margin: 0.3rem 0;">{label}</div>
                    <div style="font-size: 0.85rem; color: #455a64; line-height: 1.8;">
                        <b>{p['count']}</b> customers ({p['pct']}%)<br>
                        Avg Income: <b>${p['Income']:,.0f}</b><br>
                        Avg Spend: <b>${p['Total_Spend']:,.0f}</b><br>
                        Dependents: <b>{p['Total_Dependents']:.1f}</b><br>
                        Campaigns: <b>{p['Total_Campaigns_Accepted']:.2f}</b><br>
                        Deals: <b>{p['NumDealsPurchases']:.1f}</b><br>
                        Web Visits: <b>{p['NumWebVisitsMonth']:.1f}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Segment Comparison — Key Metrics")

    compare_features = ['Income', 'Total_Spend', 'Total_Purchases',
                        'Total_Dependents', 'Total_Campaigns_Accepted', 'NumDealsPurchases']

    label_order = sorted(profiles.keys(), key=lambda x: -profiles[x]['Income'])

    for feat in compare_features:
        values = [profiles[l][feat] for l in label_order]
        colors = [SEGMENT_CONFIG[l]['color'] for l in label_order]

        fig = go.Figure(go.Bar(
            x=label_order, y=values,
            marker_color=colors,
            text=[f"${v:,.0f}" if feat in ['Income', 'Total_Spend'] else f"{v:.1f}" for v in values],
            textposition='outside', textfont=dict(size=12)
        ))
        fig.update_layout(
            title=feat, height=280,
            margin=dict(l=0, r=0, t=40, b=30),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3: ABOUT
# ============================================================
with tab3:
    st.markdown("#### About This App")
    st.markdown("""
    **Customer Personality Analysis — Segmentation Predictor**

    This app predicts which customer segment a new customer belongs to using a trained classifier.
    """)

    st.markdown("#### Pipeline")
    st.markdown("""
    1. **EDA Notebook** — Cleaned 2,240 customers → 2,236 rows, engineered 12 features
    2. **Model Building Notebook** — Compared K-Means, Hierarchical, DBSCAN. Selected K-Means (K=4)
    3. **Classification Notebook** — Trained 4 classifiers on K-Means labels for production deployment
    4. **This App** — Uses the best classifier to predict segments with confidence scores
    """)

    st.markdown("#### Model Comparison")

    results_md = "| Model | Accuracy | F1-Score |\n|---|---|---|\n"
    best = model_results.get('best', '')
    for name in ['Logistic Regression', 'Random Forest', 'XGBoost', 'SVM']:
        if name in model_results:
            r = model_results[name]
            marker = " ✅" if name == best else ""
            results_md += f"| {name}{marker} | {r['accuracy']}% | {r['f1']}% |\n"
    st.markdown(results_md)
    st.markdown(f"**Selected:** {best}")

    st.markdown("#### Segment Distribution")
    seg_md = "| Segment | Count | % |\n|---|---|---|\n"
    for label in sorted(profiles.keys(), key=lambda x: -profiles[x]['count']):
        p = profiles[label]
        seg_md += f"| {label} | {p['count']} | {p['pct']}% |\n"
    st.markdown(seg_md)

    st.markdown("---")
    st.markdown("**Project:** P-654 | ExcelR Institute")

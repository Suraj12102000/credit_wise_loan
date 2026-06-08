import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditWise — SecureTrust Bank",
    page_icon="🏦",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .main { background-color: #F5F4F0; }
    .block-container { padding: 2rem 3rem; max-width: 1200px; }

    /* Header */
    .cw-header {
        background: #0F1923;
        color: #fff;
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .cw-header h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        margin: 0;
        color: #fff;
    }
    .cw-header p { color: #8A9BB0; margin: 0.4rem 0 0; font-size: 0.95rem; }
    .cw-badge {
        background: #1E3A52;
        color: #4FC3F7;
        font-size: 0.75rem;
        padding: 6px 14px;
        border-radius: 99px;
        font-weight: 500;
        letter-spacing: 0.04em;
    }

    /* Section labels */
    .cw-section {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #8A9BB0;
        margin: 1.8rem 0 0.8rem;
        border-bottom: 1px solid #E2E0D8;
        padding-bottom: 0.5rem;
    }

    /* Result cards */
    .result-approved {
        background: linear-gradient(135deg, #0D4F3C, #0A6B50);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        color: #fff;
        margin-top: 1rem;
    }
    .result-rejected {
        background: linear-gradient(135deg, #5C1A1A, #8B2020);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        color: #fff;
        margin-top: 1rem;
    }
    .result-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.8rem;
        margin: 0 0 0.5rem;
    }
    .result-sub { opacity: 0.8; font-size: 0.9rem; margin: 0; }

    /* Factor pills */
    .factor-good {
        background: #D4EDDA; color: #155724;
        padding: 6px 14px; border-radius: 99px;
        font-size: 0.82rem; font-weight: 500;
        display: inline-block; margin: 4px 4px 4px 0;
    }
    .factor-bad {
        background: #F8D7DA; color: #721C24;
        padding: 6px 14px; border-radius: 99px;
        font-size: 0.82rem; font-weight: 500;
        display: inline-block; margin: 4px 4px 4px 0;
    }
    .factor-neutral {
        background: #FFF3CD; color: #856404;
        padding: 6px 14px; border-radius: 99px;
        font-size: 0.82rem; font-weight: 500;
        display: inline-block; margin: 4px 4px 4px 0;
    }

    /* Metric boxes */
    .metric-box {
        background: #fff;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        border: 1px solid #E2E0D8;
        text-align: center;
    }
    .metric-label { font-size: 0.75rem; color: #8A9BB0; margin: 0 0 4px; }
    .metric-value { font-size: 1.4rem; font-weight: 600; color: #0F1923; margin: 0; }

    /* Input overrides */
    .stSelectbox > div > div { border-radius: 8px !important; }
    .stNumberInput > div > div > input { border-radius: 8px !important; }
    .stSlider > div { padding: 0 !important; }

    /* Submit button */
    .stButton > button {
        background: #0F1923 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        font-family: 'DM Sans', sans-serif !important;
        width: 100%;
        margin-top: 1.5rem;
        letter-spacing: 0.02em;
    }
    .stButton > button:hover { background: #1E3A52 !important; }

    div[data-testid="stSidebar"] { background: #0F1923; }
    div[data-testid="stSidebar"] * { color: #CBD5E0 !important; }
    div[data-testid="stSidebar"] h2 { color: #fff !important; font-family: 'DM Serif Display', serif !important; }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    required = [
        "loan_model.pkl", "scaler.pkl", "ohe_encoder.pkl",
        "num_imputer.pkl", "cat_imputer.pkl",
        "edu_mapping.pkl", "feature_columns.pkl"
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        return None, missing

    model           = joblib.load("loan_model.pkl")
    scaler          = joblib.load("scaler.pkl")
    ohe             = joblib.load("ohe_encoder.pkl")
    num_imp         = joblib.load("num_imputer.pkl")
    cat_imp         = joblib.load("cat_imputer.pkl")
    edu_mapping     = joblib.load("edu_mapping.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return {
        "model": model, "scaler": scaler, "ohe": ohe,
        "num_imp": num_imp, "cat_imp": cat_imp,
        "edu_mapping": edu_mapping, "feature_columns": feature_columns
    }, []


artifacts, missing_files = load_artifacts()


# ── Preprocessing pipeline ─────────────────────────────────────────────────────
OHE_COLS = ["Employment_Status", "Marital_Status", "Loan_Purpose",
            "Property_Area", "Gender", "Employer_Category"]

def preprocess(raw: dict, arts: dict) -> np.ndarray:
    df = pd.DataFrame([raw])

    # Encode Education_Level with saved mapping
    edu_map = arts["edu_mapping"]
    df["Education_Level"] = df["Education_Level"].map(edu_map).fillna(0).astype(int)

    # ── Numerical imputation ──────────────────────────────────────────────────
    # The saved num_imputer may have been fitted with extra columns
    # (Applicant_ID, raw Education_Level). We align to whatever it expects.
    num_imp      = arts["num_imp"]
    imp_num_cols = list(num_imp.feature_names_in_) \
                   if hasattr(num_imp, "feature_names_in_") else \
                   ["Applicant_Income", "Coapplicant_Income", "Age", "Dependents",
                    "Credit_Score", "Existing_Loans", "DTI_Ratio", "Savings",
                    "Collateral_Value", "Loan_Amount", "Loan_Term", "Education_Level"]

    # Build a temp frame with exactly the columns the imputer expects
    tmp_num = pd.DataFrame(columns=imp_num_cols)
    for col in imp_num_cols:
        tmp_num[col] = df[col] if col in df.columns else 0.0
    tmp_num = pd.DataFrame(num_imp.transform(tmp_num), columns=imp_num_cols)

    # Copy imputed values back for the columns we actually need
    our_num_cols = ["Applicant_Income", "Coapplicant_Income", "Age", "Dependents",
                    "Credit_Score", "Existing_Loans", "DTI_Ratio", "Savings",
                    "Collateral_Value", "Loan_Amount", "Loan_Term", "Education_Level"]
    for col in our_num_cols:
        if col in tmp_num.columns:
            df[col] = tmp_num[col].values

    # ── Categorical imputation ────────────────────────────────────────────────
    cat_imp      = arts["cat_imp"]
    imp_cat_cols = list(cat_imp.feature_names_in_) \
                   if hasattr(cat_imp, "feature_names_in_") else OHE_COLS

    tmp_cat = pd.DataFrame(columns=imp_cat_cols)
    for col in imp_cat_cols:
        tmp_cat[col] = df[col] if col in df.columns else "Unknown"
    tmp_cat = pd.DataFrame(cat_imp.transform(tmp_cat), columns=imp_cat_cols)

    for col in OHE_COLS:
        if col in tmp_cat.columns:
            df[col] = tmp_cat[col].values

    # OHE
    encoded = arts["ohe"].transform(df[OHE_COLS])
    enc_df  = pd.DataFrame(encoded,
                           columns=arts["ohe"].get_feature_names_out(OHE_COLS),
                           index=df.index)
    df = pd.concat([df.drop(columns=OHE_COLS), enc_df], axis=1)

    # Feature engineering (same as notebook)
    df["DTI_Ratio_sq"]    = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2
    df = df.drop(columns=["Credit_Score", "DTI_Ratio"], errors="ignore")

    # Align to training columns
    feat_cols = arts["feature_columns"]
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0
    df = df[feat_cols]

    return arts["scaler"].transform(df)


def analyse_factors(raw: dict) -> list:
    """Return human-readable risk factor pills."""
    factors = []
    cs  = raw["Credit_Score"]
    dti = raw["DTI_Ratio"]
    inc = raw["Applicant_Income"] + raw["Coapplicant_Income"] * 0.5
    lti = raw["Loan_Amount"] / inc if inc > 0 else 999
    ex  = raw["Existing_Loans"]
    sav = raw["Savings"]
    col = raw["Collateral_Value"]
    loa = raw["Loan_Amount"]

    factors.append(("✓ Credit score strong" if cs >= 700 else
                    "~ Credit score average" if cs >= 600 else
                    "✗ Credit score weak",
                    "good" if cs >= 700 else "neutral" if cs >= 600 else "bad"))

    factors.append(("✓ DTI ratio healthy" if dti < 0.35 else
                    "~ DTI ratio moderate" if dti < 0.5 else
                    "✗ DTI ratio high",
                    "good" if dti < 0.35 else "neutral" if dti < 0.5 else "bad"))

    factors.append(("✓ Low loan-to-income" if lti < 30 else
                    "~ Moderate loan-to-income" if lti < 60 else
                    "✗ High loan-to-income",
                    "good" if lti < 30 else "neutral" if lti < 60 else "bad"))

    factors.append(("✓ Collateral covers loan" if col >= loa else
                    "~ Partial collateral" if col >= loa * 0.5 else
                    "✗ Insufficient collateral",
                    "good" if col >= loa else "neutral" if col >= loa * 0.5 else "bad"))

    factors.append(("✓ No existing loans" if ex == 0 else
                    "~ 1–2 existing loans" if ex <= 2 else
                    "✗ Many existing loans",
                    "good" if ex == 0 else "neutral" if ex <= 2 else "bad"))

    factors.append(("✓ Good savings buffer" if sav > loa * 0.2 else
                    "~ Low savings",
                    "good" if sav > loa * 0.2 else "bad"))

    return factors


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cw-header">
  <div>
    <h1>🏦 CreditWise</h1>
    <p>SecureTrust Bank · Intelligent Loan Approval System</p>
  </div>
  <div class="cw-badge">Naive Bayes · ML-powered</div>
</div>
""", unsafe_allow_html=True)


# ── Model not found warning ────────────────────────────────────────────────────
if artifacts is None:
    st.error(f"⚠️ Missing model files: `{'`, `'.join(missing_files)}`")
    st.info("""
**Run your notebook first to generate these files.**

Add this cell at the bottom of `creditwise_loan.ipynb` and run it:

```python
import joblib

joblib.dump(nb_model,        "loan_model.pkl")
joblib.dump(scaler,          "scaler.pkl")
joblib.dump(ohe,             "ohe_encoder.pkl")
joblib.dump(num_imp,         "num_imputer.pkl")
joblib.dump(cat_imp,         "cat_imputer.pkl")

edu_mapping = dict(zip(le_edu.classes_, le_edu.transform(le_edu.classes_)))
joblib.dump(edu_mapping,     "edu_mapping.pkl")
joblib.dump(list(X_train.columns), "feature_columns.pkl")

print("All artifacts saved!")
```

Then place all `.pkl` files in the same folder as `app.py` and run:
```
streamlit run app.py
```
""")
    st.stop()


# ── Sidebar — about ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
This system uses a **Naive Bayes** classifier trained on historical
loan application data to predict whether an applicant should be
approved or rejected — before manual verification.

**Pipeline:**
- Missing value imputation
- Label + One-Hot Encoding
- Feature engineering (DTI², CreditScore²)
- Standard scaling
- Gaussian Naive Bayes

**Models compared:**
- Logistic Regression
- K-Nearest Neighbours
- Naive Bayes ✓ (best precision)
    """)
    st.divider()
    st.caption("CreditWise v1.0 · SecureTrust Bank")


# ── Input form ─────────────────────────────────────────────────────────────────
with st.form("loan_form"):
    st.markdown('<div class="cw-section">Applicant Profile</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        gender      = st.selectbox("Gender", ["Male", "Female"])
        marital     = st.selectbox("Marital Status", ["Married", "Single"])
    with c2:
        education   = st.selectbox("Education Level",
                                   ["Graduate", "Postgraduate", "Undergraduate"])
        dependents  = st.number_input("Dependents", min_value=0, max_value=10, value=0)
    with c3:
        age         = st.number_input("Age", min_value=18, max_value=80, value=32)
        property_area = st.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])

    st.markdown('<div class="cw-section">Employment</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        employment  = st.selectbox("Employment Status",
                                   ["Salaried", "Self-Employed", "Business"])
    with c2:
        employer_cat = st.selectbox("Employer Category", ["Govt", "Private", "Self"])
    with c3:
        existing_loans = st.number_input("Existing Loans", min_value=0, max_value=20, value=0)

    st.markdown('<div class="cw-section">Financial Details</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        app_income  = st.number_input("Applicant Income (₹/mo)",
                                       min_value=0, value=50000, step=1000)
        savings     = st.number_input("Savings Balance (₹)",
                                       min_value=0, value=100000, step=5000)
    with c2:
        co_income   = st.number_input("Co-applicant Income (₹/mo)",
                                       min_value=0, value=0, step=1000)
        collateral  = st.number_input("Collateral Value (₹)",
                                       min_value=0, value=500000, step=10000)
    with c3:
        credit_score = st.slider("Credit Score", 300, 900, 700, step=1)
        dti          = st.slider("DTI Ratio", 0.0, 1.0, 0.30, step=0.01,
                                 format="%.2f")

    st.markdown('<div class="cw-section">Loan Details</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        loan_amount  = st.number_input("Loan Amount (₹)",
                                        min_value=0, value=300000, step=10000)
    with c2:
        loan_term    = st.number_input("Loan Term (months)",
                                        min_value=12, max_value=360, value=120)
    with c3:
        loan_purpose = st.selectbox("Loan Purpose",
                                    ["Home", "Education", "Personal", "Business"])
    with c4:
        st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button("🔍  Predict Loan Approval")


# ── Prediction ─────────────────────────────────────────────────────────────────
if submitted:
    raw = {
        "Applicant_Income":    float(app_income),
        "Coapplicant_Income":  float(co_income),
        "Employment_Status":   employment,
        "Age":                 float(age),
        "Marital_Status":      marital,
        "Dependents":          float(dependents),
        "Credit_Score":        float(credit_score),
        "Existing_Loans":      float(existing_loans),
        "DTI_Ratio":           float(dti),
        "Savings":             float(savings),
        "Collateral_Value":    float(collateral),
        "Loan_Amount":         float(loan_amount),
        "Loan_Term":           float(loan_term),
        "Loan_Purpose":        loan_purpose,
        "Property_Area":       property_area,
        "Education_Level":     education,
        "Gender":              gender,
        "Employer_Category":   employer_cat,
    }

    try:
        X_input  = preprocess(raw, artifacts)
        pred     = artifacts["model"].predict(X_input)[0]
        prob     = artifacts["model"].predict_proba(X_input)[0]
        approved = int(pred) == 1
        conf     = prob[1] if approved else prob[0]

        # ── Result banner ──
        if approved:
            st.markdown(f"""
<div class="result-approved">
  <p class="result-title">✅ Loan Approved</p>
  <p class="result-sub">Model confidence: {conf*100:.1f}% · Recommend proceeding to manual verification</p>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="result-rejected">
  <p class="result-title">❌ Loan Rejected</p>
  <p class="result-sub">Model confidence: {conf*100:.1f}% · Application does not meet lending criteria</p>
</div>""", unsafe_allow_html=True)

        # ── Metrics row ──
        st.markdown('<div class="cw-section">Key Metrics</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        total_income = app_income + co_income
        lti = loan_amount / total_income if total_income > 0 else 0
        emi = (loan_amount * (dti / 12)) / (1 - (1 + dti / 12) ** -loan_term) \
              if dti > 0 else loan_amount / loan_term

        with m1:
            st.metric("Credit Score",    f"{credit_score}",
                      delta="Good" if credit_score >= 700 else "Low")
        with m2:
            st.metric("DTI Ratio",       f"{dti:.2f}",
                      delta="Healthy" if dti < 0.35 else "High",
                      delta_color="normal" if dti < 0.35 else "inverse")
        with m3:
            st.metric("Loan / Income",   f"{lti:.1f}x",
                      delta="Low risk" if lti < 30 else "High risk",
                      delta_color="normal" if lti < 30 else "inverse")
        with m4:
            st.metric("Approval chance", f"{prob[1]*100:.0f}%")

        # ── Risk factor pills ──
        st.markdown('<div class="cw-section">Risk Factors</div>', unsafe_allow_html=True)
        factors    = analyse_factors(raw)
        pill_html  = ""
        for label, kind in factors:
            cls = f"factor-{kind}"
            pill_html += f'<span class="{cls}">{label}</span>'
        st.markdown(pill_html, unsafe_allow_html=True)

        # ── Probability bar ──
        st.markdown('<div class="cw-section">Approval Probability</div>',
                    unsafe_allow_html=True)
        st.progress(float(prob[1]), text=f"{prob[1]*100:.1f}% probability of approval")

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.exception(e)
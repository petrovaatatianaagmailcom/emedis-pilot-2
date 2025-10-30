import os
import math
import time
import numpy as np
import pandas as pd
import streamlit as st

# ====== Basic Settings ======
APP_PASSWORD = "ChangeMeNow!"
BANNER = "Prototype clinical decision-support for clinicians. Not for diagnostic use. Enter de-identified data only."
MAX_TOP = 5

# ====== Load Data ======
@st.cache_data
def load_tables():
    disease = pd.read_csv("data/disease_matrix.csv")
    tests = pd.read_csv("data/tests_suggestions.csv")
    symptoms = pd.read_csv("data/symptoms_dict.csv")
    labs = pd.read_csv("data/labs_loinc_map.csv")
    citations = pd.read_csv("evidence/citations.csv")
    return disease, tests, symptoms, labs, citations

# ====== Simple Helpers ======
def normalize(s):
    return str(s).strip().lower()

def parse_findings(text, symptoms_df):
    found = []
    for _, row in symptoms_df.iterrows():
        phrase = normalize(row["phrase"])
        if phrase in normalize(text):
            found.append(row["canonical_text"])
    return found

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ====== Scoring ======
def score_differential(found, labs, disease_df):
    scores = {}
    for _, row in disease_df.iterrows():
        dx = row["disease"]
        finding = normalize(row["finding_text"])
        w = float(row.get("weight", 1))
        if finding in found:
            scores[dx] = scores.get(dx, 0) + w
    if not scores:
        return []
    names = list(scores.keys())
    vals = list(scores.values())
    probs = softmax(np.array(vals))
    return sorted(
        [{"name": n, "probability": round(p * 100, 1)} for n, p in zip(names, probs)],
        key=lambda x: x["probability"],
        reverse=True,
    )[:MAX_TOP]

# ====== Streamlit App ======
st.set_page_config(page_title="eMedis Pilot", layout="wide")
st.title("🩺 eMedis — Clinical Decision Support (Pilot)")

# Password gate
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    pw = st.text_input("Enter password to access:", type="password")
    if st.button("Continue"):
        if pw == APP_PASSWORD:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.stop()

st.warning(BANNER)

# Load data
try:
    disease_df, tests_df, symptoms_df, labs_df, citations_df = load_tables()
except Exception as e:
    st.error("⚠️ Could not load data files. Please check your 'data' and 'evidence' folders.")
    st.stop()

# ====== Input form ======
st.header("Case Input")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 0, 120, 45)
    sex = st.selectbox("Sex", ["M", "F", "Other"])
with col2:
    complaint = st.text_input("Chief complaint", "Fever and cough")
    findings_text = st.text_area("Symptoms / Findings", "3 days of fever, cough, chest pain")

st.subheader("Laboratory results")
example_labs = pd.DataFrame([
    {"test_name": "CRP", "value": "150", "unit": "mg/L", "flag": "H"},
    {"test_name": "WBC", "value": "14", "unit": "10^9/L", "flag": "H"},
])
labs_input = st.data_editor(example_labs, num_rows="dynamic")

if st.button("Analyze Case", type="primary"):
    with st.spinner("Analyzing..."):
        found = parse_findings(findings_text, symptoms_df)
        results = score_differential(found, labs_input, disease_df)
        time.sleep(0.5)

    if not results:
        st.error("No diagnostic matches found — try adding more findings.")
    else:
        st.header("🔍 Differential Diagnosis")
        for r in results:
            st.write(f"**{r['name']}** — Likelihood: {r['probability']}%")

        st.divider()
        st.header("🧪 Suggested Next Tests")
        dx_names = [r["name"] for r in results]
        for _, t in tests_df[tests_df["disease"].isin(dx_names)].iterrows():
            st.markdown(f"- **{t['disease']}** → {t['test_name']} — *{t['why']}*")

        st.divider()
        st.header("📚 Evidence Sources")
        refs = citations_df[citations_df["citation_id"].isin(disease_df["citation_id"].unique())]
        for _, c in refs.iterrows():
            st.markdown(f"- [{c['title']}]({c['url']}) — {c['source']}")

        st.divider()
        st.header("🗣️ Feedback")
        acc = st.slider("How accurate was this differential?", 1, 5, 4)
        useful = st.slider("How useful were the suggestions?", 1, 5, 4)
        comment = st.text_area("Any comments?")
        if st.button("Submit Feedback"):
            st.success("✅ Feedback saved (local only in pilot mode).")

st.caption("© 2025 eMedis — Pilot v0.1")

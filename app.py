#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# PAGE CONFIGURATION  (must be the first Streamlit command)
# ============================================================
st.set_page_config(
    page_title="Breast Cancer Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLING
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Manrope:wght@400;500;600;700&display=swap');

    :root {
        --bg: #fff6f9;
        --panel: #ffffff;
        --panel-2: #fff0f5;
        --line: #f1d3de;
        --text: #2d2230;
        --muted: #7d6b7a;
        --accent: #d6336c;
        --accent-soft: rgba(214,51,108,.14);
        --teal: #17a589;
        --coral: #e03131;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Manrope', sans-serif;
        color: var(--text);
    }
    .stApp { background: var(--bg); }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 2.2rem; max-width: 1100px; }

    h1, h2, h3 { font-family: 'Fraunces', serif !important; letter-spacing: -0.01em; }

    /* Hero */
    .hero {
        padding: 1.8rem 2rem;
        border: 1px solid var(--line);
        border-radius: 18px;
        background:
            radial-gradient(600px 220px at 100% 0%, rgba(214,51,108,.12), transparent 70%),
            var(--panel);
        margin-bottom: 1.4rem;
    }
    .hero h1 { margin: 0 0 .35rem 0; font-size: 2.1rem; line-height: 1.15; }
    .hero p  { margin: 0; color: var(--muted); font-size: 1rem; max-width: 60ch; }

    /* Section headings */
    .group-title { font-weight: 700; font-size: 1.02rem; margin: .2rem 0 .1rem 0; }
    .group-sub   { color: var(--muted); font-size: .85rem; margin-bottom: .8rem; }

    /* Inputs */
    div[data-testid="stNumberInput"] label p {
        font-size: .85rem; font-weight: 600; color: var(--text);
    }
    div[data-testid="stNumberInput"] input {
        background: var(--panel-2) !important;
        color: var(--text) !important;
        border-radius: 10px;
    }
    div[data-testid="stNumberInput"] > div > div {
        border-radius: 10px;
        border: 1px solid var(--line);
        background: var(--panel-2);
    }
    div[data-testid="stNumberInput"] > div > div:focus-within {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px var(--accent-soft);
    }

    /* Tabs */
    button[data-baseweb="tab"] { font-weight: 600; }
    button[data-baseweb="tab"][aria-selected="true"] { color: var(--accent); }
    div[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid var(--line);
        background: var(--panel-2);
        color: var(--text);
        font-weight: 600;
        transition: border-color .15s, color .15s;
    }
    .stButton > button:hover { border-color: var(--accent); color: var(--accent); }
    .stButton > button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
        color: #ffffff;
        font-weight: 700;
        padding: .7rem 1rem;
    }
    .stButton > button[kind="primary"]:hover { background: #bd2a60; border-color: #bd2a60; color: #ffffff; }

    /* Result */
    .result {
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        margin-top: 1.2rem;
        border: 1px solid var(--line);
        background: var(--panel);
        box-shadow: 0 6px 20px rgba(214,51,108,.08);
    }
    .result.benign    { border-left: 6px solid var(--teal); }
    .result.malignant { border-left: 6px solid var(--coral); }
    .result .label    { color: var(--muted); font-size: .85rem; margin-bottom: .2rem; }
    .result .verdict  { font-family: 'Fraunces', serif; font-size: 2rem; margin: 0; }
    .result.benign .verdict    { color: var(--teal); }
    .result.malignant .verdict { color: var(--coral); }
    .result .note { color: var(--muted); font-size: .92rem; margin-top: .6rem; }

    .bar { height: 10px; border-radius: 99px; background: var(--panel-2); margin-top: 1rem; overflow: hidden; }
    .bar > span { display: block; height: 100%; border-radius: 99px; }
    .benign .bar > span    { background: var(--teal); }
    .malignant .bar > span { background: var(--coral); }
    .bar-caption { display: flex; justify-content: space-between; color: var(--muted); font-size: .8rem; margin-top: .4rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
    .disclaimer {
        font-size: .82rem; color: var(--muted);
        border: 1px solid var(--line); border-radius: 12px;
        padding: .8rem .9rem; background: var(--panel-2);
    }

    /* ---- Force readable text everywhere (works even if the browser/Streamlit theme is dark) ---- */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp li, .stApp div,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
    [data-testid="stSidebar"], [data-testid="stSidebar"] *,
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary *,
    [data-testid="stSpinner"] *, [data-testid="stAlert"] * {
        color: var(--text);
    }
    .hero h1 { color: var(--text) !important; }
    .hero p, .group-sub, .result .label, .result .note, .bar-caption,
    .bar-caption span, .disclaimer, .disclaimer * { color: var(--muted) !important; }
    .result.benign .verdict { color: var(--teal) !important; }
    .result.malignant .verdict { color: var(--coral) !important; }

    /* Sidebar and its buttons */
    section[data-testid="stSidebar"] { background: var(--panel) !important; }
    section[data-testid="stSidebar"] .stButton > button,
    section[data-testid="stSidebar"] .stButton > button * { color: var(--text) !important; }
    section[data-testid="stSidebar"] .stButton > button:hover,
    section[data-testid="stSidebar"] .stButton > button:hover * { color: var(--accent) !important; }

    /* Buttons */
    .stButton > button, .stButton > button * { color: var(--text) !important; }
    .stButton > button[kind="primary"], .stButton > button[kind="primary"] * { color: #ffffff !important; }

    /* Tabs */
    button[data-baseweb="tab"], button[data-baseweb="tab"] * { color: var(--muted) !important; }
    button[data-baseweb="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"] * { color: var(--accent) !important; }

    /* Number input: text, +/- buttons, help icon */
    div[data-testid="stNumberInput"] input { color: var(--text) !important; -webkit-text-fill-color: var(--text) !important; }
    div[data-testid="stNumberInput"] button { background: var(--panel-2) !important; border-color: var(--line) !important; }
    div[data-testid="stNumberInput"] button svg,
    div[data-testid="stNumberInput"] svg { color: var(--text) !important; fill: var(--text) !important; }

    /* Expander + table of values */
    [data-testid="stExpander"] { background: var(--panel) !important; border: 1px solid var(--line) !important; border-radius: 12px; }
    table.vals { width: 100%; border-collapse: collapse; font-size: .9rem; }
    table.vals th, table.vals td { text-align: left; padding: .5rem .6rem; border-bottom: 1px solid var(--line); color: var(--text) !important; }
    table.vals th { font-weight: 700; background: var(--panel-2); }

    /* Warning box */
    [data-testid="stAlert"] { background: #fff4d6 !important; border: 1px solid #f0d58a; border-radius: 12px; }
    [data-testid="stAlert"] * { color: #5b4500 !important; }

    /* Tooltips */
    div[data-baseweb="tooltip"] div, div[data-baseweb="popover"] div { color: #ffffff !important; }

    /* ---- Extra-strong rules for buttons and tabs (covers newer Streamlit versions) ---- */
    button[data-testid="stBaseButton-secondary"],
    button[data-testid="stBaseButton-secondary"] p,
    button[data-testid="stBaseButton-secondary"] span,
    .stButton button[kind="secondary"], .stButton button[kind="secondary"] p {
        color: #2d2230 !important;
        -webkit-text-fill-color: #2d2230 !important;
    }
    button[data-testid="stBaseButton-secondary"] {
        background: #fff0f5 !important;
        border: 1px solid var(--line) !important;
    }
    button[data-testid="stBaseButton-secondary"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover p {
        color: var(--accent) !important;
        -webkit-text-fill-color: var(--accent) !important;
        border-color: var(--accent) !important;
    }
    button[data-testid="stBaseButton-primary"],
    button[data-testid="stBaseButton-primary"] p,
    button[data-testid="stBaseButton-primary"] span,
    .stButton button[kind="primary"] p {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    button[data-testid="stBaseButton-primary"] {
        background: var(--accent) !important;
        border: 1px solid var(--accent) !important;
        font-weight: 700 !important;
    }
    button[data-testid="stBaseButton-primary"]:hover { background: #bd2a60 !important; }
    button[data-baseweb="tab"] p {
        color: #7d6b7a !important;
        -webkit-text-fill-color: #7d6b7a !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #d6336c !important;
        -webkit-text-fill-color: #d6336c !important;
    }

    /* ---- Extra: uploader, caption, divider, metric cards ---- */
    [data-testid="stFileUploader"] section {
        background: var(--panel-2) !important;
        border: 1px dashed #e8b9ca !important;
        border-radius: 12px;
    }
    [data-testid="stFileUploader"] section *,
    [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] small {
        color: var(--text) !important;
    }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--muted) !important; }
    hr { border-color: var(--line) !important; }
    h2, h3 { color: var(--text) !important; }
    button[data-testid="stBaseButton-secondary"] svg { fill: currentColor; }
    [data-testid="stDownloadButton"] button, [data-testid="stDownloadButton"] button p {
        color: #2d2230 !important; -webkit-text-fill-color: #2d2230 !important;
    }
    .probs { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px; }
    .prob {
        background: var(--panel); border: 1px solid var(--line);
        border-radius: 14px; padding: 14px 18px;
    }
    .prob .k { color: var(--muted) !important; font-size: .85rem; }
    .prob .n { font-family: 'Fraunces', serif; font-size: 1.7rem; margin-top: 2px; }
    .prob.b .n { color: var(--teal) !important; }
    .prob.m .n { color: var(--coral) !important; }

    /* ---- Force light look without .streamlit/config.toml ---- */
    html, :root, .stApp { color-scheme: light !important; }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--bg) !important; }
    header[data-testid="stHeader"] { background: transparent !important; }
    header[data-testid="stHeader"] *, [data-testid="stToolbar"] *, [data-testid="stSidebarCollapseButton"] * {
        color: var(--text) !important; fill: var(--text) !important;
    }
    [data-testid="stSidebar"] > div { background: var(--panel) !important; }
    div[data-testid="stNumberInput"] [data-baseweb="input"],
    div[data-testid="stNumberInput"] [data-baseweb="base-input"] { background: var(--panel-2) !important; }
    div[data-testid="stTooltipIcon"] svg { fill: var(--muted) !important; color: var(--muted) !important; }
    div.tbl { overflow: auto; max-height: 420px; border: 1px solid var(--line); border-radius: 12px; background: var(--panel); }
    div.tbl table { width: 100%; border-collapse: collapse; font-size: .88rem; }
    div.tbl th, div.tbl td { text-align: left; padding: .5rem .7rem; border-bottom: 1px solid var(--line); color: var(--text) !important; white-space: nowrap; }
    div.tbl th { position: sticky; top: 0; background: var(--panel-2); font-weight: 700; }
    div.tbl tr:last-child td { border-bottom: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD TRAINED ML PIPELINE
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


model = load_model()

# ============================================================
# FEATURES  (this order must match the order used for training)
# ============================================================
FEATURES = [
    ("perimeter_worst", "Perimeter Worst"),
    ("area_worst", "Area Worst"),
    ("radius_worst", "Radius Worst"),
    ("concave_points_worst", "Concave Points Worst"),
    ("concave_points_mean", "Concave Points Mean"),
    ("perimeter_mean", "Perimeter Mean"),
    ("area_mean", "Area Mean"),
    ("radius_mean", "Radius Mean"),
    ("area_se", "Area SE"),
    ("concavity_mean", "Concavity Mean"),
]
REQUIRED_COLUMNS = [k for k, _ in FEATURES]

# Sample rows from the Wisconsin breast cancer dataset, for quick testing
SAMPLES = {
    "benign": dict(
        perimeter_worst=99.70, area_worst=711.2, radius_worst=15.11,
        concave_points_worst=0.1288, concave_points_mean=0.04781,
        perimeter_mean=87.46, area_mean=566.3, radius_mean=13.54,
        area_se=23.56, concavity_mean=0.06664,
    ),
    "malignant": dict(
        perimeter_worst=184.6, area_worst=2019.0, radius_worst=25.38,
        concave_points_worst=0.2654, concave_points_mean=0.1471,
        perimeter_mean=122.8, area_mean=1001.0, radius_mean=17.99,
        area_se=153.4, concavity_mean=0.3001,
    ),
}

for key in REQUIRED_COLUMNS:
    st.session_state.setdefault(key, 0.0)


def load_sample(name: str):
    for k, v in SAMPLES[name].items():
        st.session_state[k] = float(v)


def clear_values():
    for key in REQUIRED_COLUMNS:
        st.session_state[key] = 0.0


def field(key: str, label: str, step: float, help_text: str):
    return st.number_input(
        label, min_value=0.0, step=step, format="%.4f", key=key, help=help_text
    )



def show_table(df: pd.DataFrame):
    """Render a DataFrame as a light, always-readable HTML table."""
    html = df.to_html(index=False, border=0, escape=True, na_rep="")
    st.markdown(f'<div class="tbl">{html}</div>', unsafe_allow_html=True)


def is_malignant(label) -> bool:
    return label == "M" or label == 1


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🩺 About")
    st.write(
        "This tool predicts whether a breast mass is **benign** or "
        "**malignant** from measurements of cell nuclei taken from a "
        "fine needle aspirate (FNA)."
    )
    st.markdown("**📝 How to use**")
    st.write(
        "1. Enter the ten measurements, or load a sample.\n"
        "2. Press **Predict diagnosis**.\n"
        "3. Read the result and confidence.\n"
        "4. Or upload a CSV for batch predictions."
    )
    st.markdown("**⚡ Quick test**")
    st.button("🟢 Load benign sample", on_click=load_sample, args=("benign",), use_container_width=True)
    st.button("🔴 Load malignant sample", on_click=load_sample, args=("malignant",), use_container_width=True)
    st.button("🧹 Clear all values", on_click=clear_values, use_container_width=True)
    st.markdown(
        '<div class="disclaimer">⚕️ For education and research only. '
        "This is not a medical device and must not replace a clinician's diagnosis.</div>",
        unsafe_allow_html=True,
    )

# ============================================================
# TITLE
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🩺 Breast Cancer Diagnosis Prediction 🎗️</h1>
        <p>🔬 Enter the cell nucleus measurements below to get a prediction
        with a confidence score.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# INDIVIDUAL PREDICTION
# ============================================================
st.header("🔬 Individual Tumor Prediction")

tab_mean, tab_worst, tab_se = st.tabs(
    ["📊 Average values", "📈 Worst values", "🧪 Standard error"]
)

with tab_mean:
    st.markdown('<div class="group-title">📊 Mean of all cells in the sample</div>'
                '<div class="group-sub">Average size and shape of the nuclei.</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        field("radius_mean", "📏 Radius Mean", 0.1, "Mean distance from the center to the edge of the nucleus.")
        field("perimeter_mean", "⭕ Perimeter Mean", 0.1, "Mean size of the nucleus outline.")
        field("area_mean", "🔲 Area Mean", 1.0, "Mean area of the nucleus.")
    with c2:
        field("concavity_mean", "🌀 Concavity Mean", 0.001, "Mean severity of concave portions of the contour.")
        field("concave_points_mean", "🔹 Concave Points Mean", 0.001, "Mean number of concave portions of the contour.")

with tab_worst:
    st.markdown('<div class="group-title">📈 Largest values in the sample</div>'
                '<div class="group-sub">Mean of the three largest values found.</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        field("radius_worst", "📏 Radius Worst", 0.1, "Worst (largest) radius.")
        field("perimeter_worst", "⭕ Perimeter Worst", 0.1, "Worst (largest) perimeter.")
        field("area_worst", "🔲 Area Worst", 1.0, "Worst (largest) area.")
    with c2:
        field("concave_points_worst", "🔹 Concave Points Worst", 0.001, "Worst number of concave portions.")

with tab_se:
    st.markdown('<div class="group-title">🧪 Variation across cells</div>'
                '<div class="group-sub">How much the measurement differs between nuclei.</div>',
                unsafe_allow_html=True)
    c1, _ = st.columns(2)
    with c1:
        field("area_se", "🔬 Area SE", 0.1, "Standard error of the nucleus area.")

st.write("")
predict = st.button("🔍 Predict Diagnosis", type="primary", use_container_width=True)

# ============================================================
# PREDICTION
# ============================================================
if predict:
    values = [st.session_state[k] for k in REQUIRED_COLUMNS]

    if all(v == 0 for v in values):
        st.warning("⚠️ All values are 0. Enter the measurements or load a sample from the sidebar.")
    else:
        input_data = pd.DataFrame([values], columns=REQUIRED_COLUMNS)

        with st.spinner("🔬 Analysing measurements..."):
            prediction = model.predict(input_data)

            benign_probability = None
            malignant_probability = None
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(input_data)[0]
                class_probabilities = dict(zip(model.classes_, probability))
                benign_probability = (
                    class_probabilities.get("B", 0) * 100
                    + class_probabilities.get(0, 0) * 100
                )
                malignant_probability = (
                    class_probabilities.get("M", 0) * 100
                    + class_probabilities.get(1, 0) * 100
                )

        malignant = is_malignant(prediction[0])
        css = "malignant" if malignant else "benign"
        verdict = "⚠️ Potentially Malignant" if malignant else "✅ Benign"
        note = (
            "The measurements are consistent with a malignant mass. "
            "Please consult an oncologist for confirmation."
            if malignant
            else "The measurements are consistent with a benign mass. "
            "Routine follow-up with a clinician is still advised."
        )

        bar = ""
        if benign_probability is not None:
            conf = malignant_probability if malignant else benign_probability
            bar = f"""
            <div class="bar"><span style="width:{conf:.1f}%"></span></div>
            <div class="bar-caption"><span>🎯 Model confidence</span><span>{conf:.1f}%</span></div>
            """

        st.markdown(
            f"""
            <div class="result {css}">
                <div class="label">🩺 Predicted diagnosis</div>
                <p class="verdict">{verdict}</p>
                {bar}
                <div class="note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if benign_probability is not None:
            st.markdown(
                f"""
                <div class="probs">
                    <div class="prob b"><div class="k">🟢 Benign Probability</div>
                        <div class="n">{benign_probability:.2f}%</div></div>
                    <div class="prob m"><div class="k">🔴 Malignant Probability</div>
                        <div class="n">{malignant_probability:.2f}%</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("📋 Show the values used"):
            show_table(input_data)

# ============================================================
# SEPARATOR
# ============================================================
st.divider()

# ============================================================
# BATCH PREDICTION
# ============================================================
st.header("📂 Batch Prediction")
st.write(
    "Upload a CSV file containing the required tumor features "
    "to generate predictions for multiple records."
)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("📋 Uploaded Data")
        show_table(data)

        missing_columns = [c for c in REQUIRED_COLUMNS if c not in data.columns]

        if missing_columns:
            st.error("The following required columns are missing:")
            st.write(missing_columns)
        else:
            batch_input = data[REQUIRED_COLUMNS]
            predictions = model.predict(batch_input)

            result_data = data.copy()
            result_data["Prediction"] = predictions
            result_data["Diagnosis"] = result_data["Prediction"].apply(
                lambda x: "Malignant" if is_malignant(x) else "Benign"
            )

            st.subheader("📊 Prediction Results")
            show_table(result_data)

            csv = result_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Prediction Results",
                data=csv,
                file_name="breast_cancer_predictions.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "⚠️ Educational machine learning project. "
    "Not intended to replace professional medical diagnosis."
)


import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="MessMind AI",
    page_icon="🍽️",
    layout="wide"
)


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f4fff8 0%, #ecf9ff 50%, #fff9eb 100%);
        color: #0f172a;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        color: #0f172a;
    }

    .stMarkdownContainer,
    .stMarkdownContainer p,
    .stMarkdownContainer div,
    .stMarkdownContainer span,
    .stMarkdownContainer h1,
    .stMarkdownContainer h2,
    .stMarkdownContainer h3,
    .stMarkdownContainer h4,
    .stMarkdownContainer li,
    .stMarkdownContainer label,
    .stDataFrame,
    .stDataFrame div,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stVerticalBlock"] {
        color: #0f172a !important;
    }

    .hero {
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 35%, #f59e0b 100%);
        border-radius: 22px;
        padding: 2rem 2.2rem;
        color: white;
        box-shadow: 0 10px 30px rgba(15, 118, 110, 0.18);
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.6rem;
        font-weight: 800;
    }

    .hero p {
        margin-top: 0.5rem;
        font-size: 1.05rem;
        opacity: 0.96;
    }

    .subtle-text {
        font-size: 0.95rem;
        color: #334155;
        margin-bottom: 1.2rem;
    }

    .result-box {
        border-radius: 20px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 1.3rem;
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        color: #0f172a;
    }

    .result-box h3 {
        margin-top: 0;
        margin-bottom: 0.4rem;
        font-size: 1.2rem;
        color: #0f172a;
    }

    .risk-low {
        background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    }

    .risk-medium {
        background: linear-gradient(135deg, #fff7ed, #ffedd5);
    }

    .risk-high {
        background: linear-gradient(135deg, #fef2f2, #fee2e2);
    }

    .info-card {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 18px;
        padding: 1rem 1.15rem;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
        color: #0f172a;
    }

    .section-title {
        color: #0f172a;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }

    .stAlert,
    .stSuccess,
    .stInfo {
        background: rgba(255, 255, 255, 0.65);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 16px;
    }

    .stAlert > div,
    .stSuccess > div,
    .stInfo > div {
        color: #0f172a !important;
    }

    .sidebar .block-container {
        background: #f8fafc;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #eefdf6 100%);
    }

    [data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 700;
    }

    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 12px;
        background: linear-gradient(135deg, #0f766e, #14b8a6);
        color: white;
        font-weight: 700;
        padding: 0.7rem 1rem;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #115e59, #0d9488);
        box-shadow: 0 8px 20px rgba(20, 184, 166, 0.2);
    }

    .stDataFrame {
        border-radius: 14px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------
# LOAD MODEL
# -----------------------------------

@st.cache_resource
def load_model():
    return joblib.load("waste_model.pkl")


model = load_model()


# -----------------------------------
# LOAD DATA
# -----------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("food_waste_data.csv")


df = load_data()


# -----------------------------------
# HERO TITLE
# -----------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🍽️ MessMind AI</h1>
        <p>Smart Food Waste Prediction & Reduction System</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='subtle-text'>AI-powered decision support for college mess operations and sustainable food planning.</div>",
    unsafe_allow_html=True,
)


# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.header("Enter Meal Information")

day = st.sidebar.selectbox(
    "Day",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)


meal = st.sidebar.selectbox(
    "Meal",
    [
        "Breakfast",
        "Lunch",
        "Dinner"
    ]
)


menu = st.sidebar.selectbox(
    "Menu",
    [
        "Rice_Dal",
        "Roti_Veg",
        "Biryani",
        "Paneer_Rice",
        "Poha",
        "Chole_Rice"
    ]
)


students_present = st.sidebar.number_input(
    "Expected Students",
    min_value=10,
    max_value=2000,
    value=300,
    step=10
)


predict_button = st.sidebar.button("Predict Food Waste")


# -----------------------------------
# MAIN PREDICTION
# -----------------------------------

if predict_button:
    input_data = pd.DataFrame(
        {
            "day": [day],
            "meal": [meal],
            "menu": [menu],
            "students_present": [students_present]
        }
    )

    prediction = model.predict(input_data)[0]
    prediction = max(0, prediction)

    if prediction < 10:
        risk = "Low"
        recommendation = (
            "Current preparation appears relatively efficient. "
            "Maintain the planned quantity and continue monitoring."
        )
    elif prediction < 20:
        risk = "Medium"
        recommendation = (
            "Consider slightly reducing preparation and "
            "monitor actual attendance before cooking the full quantity."
        )
    else:
        risk = "High"
        recommendation = (
            "High waste risk detected. Consider reducing preparation, "
            "checking attendance, and preparing food in smaller batches."
        )

    risk_class = risk.lower()

    st.markdown(
        f"""
        <div class="result-box risk-{risk_class}">
            <h3>✅ Prediction Generated Successfully</h3>
            <div>Estimated leftover food for this meal is <strong>{prediction:.2f} kg</strong> with a <strong>{risk}</strong> waste risk.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Predicted Leftover", f"{prediction:.2f} kg")

    with col2:
        st.metric("Expected Students", students_present)

    with col3:
        st.metric("Waste Risk", risk)

    st.markdown("<div class='section-title'>AI Recommendation</div>", unsafe_allow_html=True)
    st.info(recommendation)

    st.markdown("<div class='section-title'>Sustainability Impact</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='info-card'>
            Based on the prediction, approximately <strong>{prediction:.2f} kg</strong> of food may remain after this meal.
            <br><br>
            Reducing avoidable food waste can help institutions use food, money, water, and energy more efficiently.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    st.markdown(
        """
        <div class='info-card'>
            Select meal details from the sidebar and click <strong>Predict Food Waste</strong> to generate a forecast.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------
# DASHBOARD
# -----------------------------------

st.markdown("<div class='section-title'>Food Waste Analytics</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(df))

with col2:
    st.metric("Average Waste", f"{df['leftover_kg'].mean():.2f} kg")

with col3:
    st.metric("Maximum Recorded Waste", f"{df['leftover_kg'].max():.2f} kg")


# -----------------------------------
# CHART
# -----------------------------------

st.markdown("<div class='section-title'>Average Food Waste by Meal</div>", unsafe_allow_html=True)

meal_waste = df.groupby("meal")["leftover_kg"].mean().sort_values()

fig, ax = plt.subplots(figsize=(8, 4))
meal_waste.plot(kind="bar", ax=ax, color=["#14b8a6", "#f59e0b", "#ef4444"])
ax.set_xlabel("Meal")
ax.set_ylabel("Average Leftover Food (kg)")
ax.set_title("Average Food Waste by Meal")
ax.grid(axis="y", linestyle="--", alpha=0.3)
for label in ax.get_xticklabels():
    label.set_rotation(0)

st.pyplot(fig)


# -----------------------------------
# DATASET PREVIEW
# -----------------------------------

st.markdown("<div class='section-title'>Historical Dataset</div>", unsafe_allow_html=True)
st.dataframe(df, width="stretch")


# -----------------------------------
# RESPONSIBLE AI
# -----------------------------------

st.divider()
st.markdown("<div class='section-title'>Responsible AI Considerations</div>", unsafe_allow_html=True)

st.write(
    """
    **Fairness:** The model should be trained on representative
    data from different days and meal types.

    **Transparency:** Predictions are based on historical food-waste
    patterns and the input values provided by the user.

    **Privacy:** The system does not require personally identifiable
    student information.

    **Human Oversight:** Predictions are recommendations and should
    not replace decisions made by mess administrators.

    **Data Quality:** Real-world performance depends on the quality
    and quantity of historical food-waste data.
    """
)


# -----------------------------------
# FOOTER
# -----------------------------------

st.divider()
st.caption("MessMind AI | AI for Sustainability | SDG 12")
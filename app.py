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
# TITLE
# -----------------------------------

st.title("MessMind AI")

st.subheader(
    "Smart Food Waste Prediction & Reduction System"
)

st.write(
    "An AI-powered decision-support system designed to "
    "help college messes predict food waste and improve "
    "food preparation."
)


st.divider()


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


predict_button = st.sidebar.button(
    "Predict Food Waste"
)


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


    # -----------------------------------
    # WASTE RISK
    # -----------------------------------

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


    # -----------------------------------
    # RESULTS
    # -----------------------------------

    st.success("Prediction generated successfully!")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Predicted Leftover",
            f"{prediction:.2f} kg"
        )


    with col2:

        st.metric(
            "Expected Students",
            students_present
        )


    with col3:

        st.metric(
            "Waste Risk",
            risk
        )


    st.divider()


    # -----------------------------------
    # RECOMMENDATION
    # -----------------------------------

    st.subheader("AI Recommendation")

    st.info(recommendation)


    # -----------------------------------
    # SUSTAINABILITY MESSAGE
    # -----------------------------------

    st.subheader("Sustainability Impact")

    st.write(
        f"Based on the prediction, approximately "
        f"**{prediction:.2f} kg** of food may remain after this meal."
    )

    st.write(
        "Reducing avoidable food waste can help institutions "
        "use food, money, water and energy more efficiently."
    )


# -----------------------------------
# DASHBOARD
# -----------------------------------

st.divider()

st.header("Food Waste Analytics")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Total Records",
        len(df)
    )


with col2:

    st.metric(
        "Average Waste",
        f"{df['leftover_kg'].mean():.2f} kg"
    )


with col3:

    st.metric(
        "Maximum Recorded Waste",
        f"{df['leftover_kg'].max():.2f} kg"
    )


# -----------------------------------
# CHART
# -----------------------------------

st.subheader("Average Food Waste by Meal")


meal_waste = (
    df.groupby("meal")["leftover_kg"]
    .mean()
    .sort_values()
)


fig, ax = plt.subplots()

meal_waste.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Meal")
ax.set_ylabel("Average Leftover Food (kg)")
ax.set_title("Average Food Waste by Meal")

st.pyplot(fig)


# -----------------------------------
# DATASET PREVIEW
# -----------------------------------

st.subheader("Historical Dataset")

st.dataframe(
    df,
    use_container_width=True
)


# -----------------------------------
# RESPONSIBLE AI
# -----------------------------------

st.divider()

st.header("Responsible AI Considerations")

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

st.caption(
    "MessMind AI | AI for Sustainability | SDG 12"
)
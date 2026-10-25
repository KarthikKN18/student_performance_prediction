
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("student_model.joblib")

with open("metrics.json", "r") as file:
    metrics = json.load(file)

model = load_model()

st.title("Student Performance Prediction System")
st.write(
    "Enter the student's information to estimate "
    "their final Mathematics grade."
)

st.info(
    "This is an educational estimate based on historical "
    "data, not a guaranteed exam result."
)

st.subheader("Enter student information")

with st.form("student_form"):
    age = st.slider("Student age", 15, 22, 17)

    studytime = st.selectbox(
        "Weekly study time",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Less than 2 hours",
            2: "2–5 hours",
            3: "5–10 hours",
            4: "More than 10 hours"
        }[x]
    )

    failures = st.selectbox(
        "Previous class failures",
        [0, 1, 2, 3, 4],
        format_func=lambda x: "3 or more" if x == 4 else str(x)
    )

    absences = st.number_input(
        "Number of absences",
        min_value=0,
        max_value=100,
        value=2
    )

    schoolsup = st.selectbox(
        "Extra school support",
        ["yes", "no"]
    )

    famsup = st.selectbox(
        "Family educational support",
        ["yes", "no"]
    )

    paid = st.selectbox(
        "Paid extra classes",
        ["yes", "no"]
    )

    activities = st.selectbox(
        "Extracurricular activities",
        ["yes", "no"]
    )

    submitted = st.form_submit_button("Predict final grade")

if submitted:
    student = pd.DataFrame([{
        "age": age,
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities
    }])

    prediction = float(model.predict(student)[0])

    # Grades in the original dataset range from 0 to 20.
    prediction = max(0.0, min(20.0, prediction))

    st.subheader("Prediction result")
    st.metric("Estimated final grade", f"{prediction:.1f} / 20")
    st.progress(prediction / 20)

    st.caption(
        "The prediction is an estimate. Actual results depend "
        "on many factors not included in this model."
    )

st.divider()
st.subheader("Model evaluation")

test_metrics = metrics["test_metrics"]

c1, c2, c3 = st.columns(3)
c1.metric("Test MAE", f'{test_metrics["MAE"]:.2f}')
c2.metric("Test RMSE", f'{test_metrics["RMSE"]:.2f}')
c3.metric("Test R²", f'{test_metrics["R2"]:.2f}')

st.write("Model selected:", metrics["selected_model"])
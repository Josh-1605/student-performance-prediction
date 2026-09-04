import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# Page settings
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Predictor")
st.write("Predict a student's final exam score using academic and lifestyle factors.")


# Load dataset
df = pd.read_csv("student_performance.csv")

features = [
    "Study_Hours_Per_Week",
    "Attendance_Percentage",
    "Previous_Term_Grade",
    "Sleep_Hours_Per_Night",
    "Participates_Extracurricular"
]

target = "Final_Exam_Score"


# Convert Yes/No to numbers if necessary
if df["Participates_Extracurricular"].dtype == "object":
    df["Participates_Extracurricular"] = (
        df["Participates_Extracurricular"]
        .map({"Yes": 1, "No": 0})
    )


X = df[features]
y = df[target]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# User inputs
st.subheader("Enter Student Details")

study_hours = st.number_input(
    "Study Hours Per Week",
    min_value=0.0,
    max_value=100.0,
    value=20.0
)

attendance = st.number_input(
    "Attendance Percentage",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

previous_grade = st.number_input(
    "Previous Term Grade",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

sleep_hours = st.number_input(
    "Sleep Hours Per Night",
    min_value=0.0,
    max_value=24.0,
    value=7.0
)

extracurricular = st.selectbox(
    "Participates in Extracurricular Activities?",
    ["No", "Yes"]
)

extra_value = 1 if extracurricular == "Yes" else 0


# Prediction
if st.button("🔮 Predict Final Exam Score"):

    input_data = np.array([[
        study_hours,
        attendance,
        previous_grade,
        sleep_hours,
        extra_value
    ]])

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Final Exam Score: **{prediction:.2f} / 100**"
    )
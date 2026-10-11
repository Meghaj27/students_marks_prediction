import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page setup
st.set_page_config(
    page_title="Student Outcome Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Card Container Styling */
    .stCard {
        background: #1e293b;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #334155;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    
    /* Heading Styling */
    h1 {
        color: #38bdf8;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Result Box Custom Badges */
    .pass-badge {
        background-color: #065f46;
        color: #34d399;
        border: 1px solid #059669;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .fail-badge {
        background-color: #881337;
        color: #fecdd3;
        border: 1px solid #e11d48;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained Gaussian Naive Bayes model
@st.cache_resource
def load_model():
    with open('naive_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Error: 'naive_model.pkl' was not found in the current directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Header
st.markdown("<h1>🎓 Student Pass/Fail Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Gaussian Naive Bayes Performance Prediction System</p>", unsafe_allow_html=True)

# Input Form Structure
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Demographic & Academic Info")
        age = st.number_input("Age", min_value=10, max_value=100, value=20, step=1)
        gender = st.number_input("Gender (Encoded Numerical)", min_value=0.0, max_value=10.0, value=0.0, step=1.0, help="e.g. 0 for Female, 1 for Male depending on your encoding standard")
        department = st.number_input("Department (Encoded Numerical)", min_value=0.0, max_value=50.0, value=1.0, step=1.0)
        study_hours = st.number_input("Study Hours Per Day", min_value=0.0, max_value=24.0, value=4.0, step=0.5)

    with col2:
        st.subheader("📊 Performance Metrics")
        attendance = st.number_input("Attendance Percentage (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
        assignments = st.number_input("Assignments Completed", min_value=0, max_value=100, value=10, step=1)
        midterm = st.number_input("Midterm Score", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
        final_score = st.number_input("Final Score", min_value=0.0, max_value=100.0, value=78.0, step=1.0)

    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button("Predict Outcome", type="primary", use_container_width=True)

# Process Prediction
if submit_button:
    # Prepare inputs matching exact model feature ordering
    input_df = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Department': department,
        'Study_Hours_Per_Day': study_hours,
        'Attendance_Percentage': attendance,
        'Assignments_Completed': assignments,
        'Midterm_Score': midterm,
        'Final_Score': final_score
    }])

    prediction = model.predict(input_df)[0]
    
    st.divider()
    st.subheader("Prediction Result")
    
    if str(prediction).lower() == 'pass':
        st.markdown(f"<div class='pass-badge'>🎉 Result: PASS</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='fail-badge'>⚠️ Result: FAIL</div>", unsafe_allow_html=True)

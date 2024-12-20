import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# 1️⃣ Load the trained model, scaler, and feature columns
def load_model():
    """Load the trained model from a file."""
    with open('./model/model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def load_scaler():
    """Load the trained scaler from a file."""
    with open('./model/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return scaler

def load_feature_columns():
    """Load the feature columns from a file."""
    with open('./model/features.pkl', 'rb') as file:
        feature_columns = pickle.load(file)
    return feature_columns

# Load trained model, scaler, and feature columns
rf_model = load_model()
scaler = load_scaler()
feature_columns = load_feature_columns()

# Title and description
st.title("Student Classification App")
st.write("This app classifies students based on various input features using a pre-trained Random Forest model.")

# Form for user inputs
st.header("Input Features")
with st.form("classification_form"):
    # Dictionary for categorical inputs with their possible values
    categorical_options = {
        'Marital_status': [1, 2, 3, 4, 5, 6],
        'Application_mode': [1, 2, 5, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57],
        'Daytime_evening_attendance': [0, 1],
        'Previous_qualification_grade': range(1, 45),
        'Mothers_qualification': range(1, 45),
        'Displaced': [0, 1],
        'Debtor': [0, 1],
        'Tuition_fees_up_to_date': [0, 1],
        'Gender': [0, 1],
        'Scholarship_holder': [0, 1]
    }

    # User inputs for categorical features
    user_input = {}
    for feature, options in categorical_options.items():
        user_input[feature] = st.selectbox(f"{feature}", options)

    # User inputs for numerical features
    numerical_features = ['Application_order', 'Admission_grade', 'Age_at_enrollment', 
                          'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations', 
                          'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade', 
                          'Curricular_units_1st_sem_without_evaluations', 'Curricular_units_2nd_sem_enrolled', 
                          'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved', 
                          'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_without_evaluations']

    for feature in numerical_features:
        user_input[feature] = st.number_input(f"{feature}", min_value=0.0, step=1.0)

    # Submit button
    submit_button = st.form_submit_button("Predict")

if submit_button:
    # Convert user input into DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Ensure the columns are in the same order as the trained model's features
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    # Feature scaling using pre-trained scaler
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = rf_model.predict(input_scaled)
    prediction_proba = rf_model.predict_proba(input_scaled)

    # Display results
    st.subheader("Prediction Result")
    status_label = "Student is classified as: " + ("Graduate" if prediction[0] == 1 else "Dropout")
    st.write(status_label)

    st.subheader("Prediction Probabilities")
    st.write(f"Graduate probability: {prediction_proba[0][1]:.2f}")
    st.write(f"Dropout probability: {prediction_proba[0][0]:.2f}")

    # Display the input data used for prediction
    st.subheader("Input Data Used for Prediction")
    st.write(input_df)

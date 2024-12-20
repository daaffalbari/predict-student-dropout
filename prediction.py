import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """Preprocess the input DataFrame by standardizing numerical columns and keeping the 'Status' column."""
    # Define the list of numerical columns to be standardized (excluding 'Status' column)
    numerical_columns = [
        'Application_order', 'Previous_qualification_grade', 'Admission_grade', 'Age_at_enrollment', 
        'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations', 
        'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade', 
        'Curricular_units_1st_sem_without_evaluations', 'Curricular_units_2nd_sem_enrolled', 
        'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved', 
        'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_without_evaluations'
    ]

    if 'Status' in df.columns:
        X = df.drop(columns=['Status'])
        y = df['Status']
    else:
        X = df
        y = None
    
    # Apply standardization to numerical columns only
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[numerical_columns])
    
    # Combine scaled columns back into the DataFrame, and keep non-numerical columns unchanged
    df_scaled = pd.DataFrame(X_scaled, columns=numerical_columns, index=df.index)
    for col in df.columns:
        if col not in numerical_columns and col != 'Status':
            df_scaled[col] = df[col]
    
    # Add 'Status' back if it was in the original DataFrame
    if y is not None:
        df_scaled['Status'] = y
    
    return df_scaled


def load_model(model_path='./model/rf_model.pkl'):
    """Load the trained Random Forest model from the specified path."""
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def classify_new_data(df, model_path='./model/rf_model.pkl'):
    """Classify new data using a pre-trained Random Forest model.
    
    Args:
        df (pd.DataFrame): The new input data to classify.
        model_path (str): The path to the saved model file.
    
    Returns:
        pd.Series: The classification predictions.
    """
    df_processed = preprocess_data(df)
    

    model = load_model(model_path)
    if model is None:
        raise ValueError("Failed to load the model.")
    
    feature_columns = [
        'Marital_status', 'Application_mode', 'Application_order', 'Daytime_evening_attendance', 
        'Previous_qualification_grade', 'Mothers_qualification', 'Admission_grade', 'Displaced', 
        'Debtor', 'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder', 'Age_at_enrollment', 
        'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations', 
        'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade', 
        'Curricular_units_1st_sem_without_evaluations', 'Curricular_units_2nd_sem_enrolled', 
        'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved', 
        'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_without_evaluations'
    ]
    
    X_new = df_processed[feature_columns]
    
    # Predict using the trained model
    predictions = model.predict(X_new)
    return predictions


# Example usage
if __name__ == "__main__":
    # Dummy data for testing
    new_data = pd.DataFrame({
        'Marital_status': [1, 2],
        'Application_mode': [1, 2],
        'Application_order': [0, 1],
        'Daytime_evening_attendance': [1, 0],
        'Previous_qualification_grade': [150, 180],
        'Mothers_qualification': [2, 3],
        'Admission_grade': [170, 190],
        'Displaced': [0, 1],
        'Debtor': [0, 1],
        'Tuition_fees_up_to_date': [1, 0],
        'Gender': [1, 0],
        'Scholarship_holder': [1, 0],
        'Age_at_enrollment': [20, 25],
        'Curricular_units_1st_sem_enrolled': [5, 6],
        'Curricular_units_1st_sem_evaluations': [5, 6],
        'Curricular_units_1st_sem_approved': [4, 5],
        'Curricular_units_1st_sem_grade': [12, 14],
        'Curricular_units_1st_sem_without_evaluations': [1, 1],
        'Curricular_units_2nd_sem_enrolled': [5, 6],
        'Curricular_units_2nd_sem_evaluations': [5, 6],
        'Curricular_units_2nd_sem_approved': [4, 5],
        'Curricular_units_2nd_sem_grade': [13, 15],
        'Curricular_units_2nd_sem_without_evaluations': [1, 1]
    })
    
    predictions = classify_new_data(new_data)
    print("Predictions:", predictions)
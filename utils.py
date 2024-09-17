import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR, SVC
import os

MODEL_DIR = 'saved_models'

def load_data():
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data.head())
        return data
    return None

def select_features(data):
    if len(data.columns) > 1:
        target_column = st.selectbox("Select the target variable", data.columns)
        feature_columns = st.multiselect("Select the feature columns", data.columns.drop(target_column))
        return target_column, feature_columns
    else:
        st.error("The dataset needs to have more than one column to build a model.")
    return None, None

def select_model():
    st.header("Select a machine learning model")
    model_name = st.selectbox("Choose a model:", ["Linear Regression", "Random Forest Classifier", "Random Forest Regressor", "Logistic Regression", "SVC (Classifier)", "SVR"])
    
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest Classifier":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "Random Forest Regressor":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == "Logistic Regression":
        model = LogisticRegression()
    elif model_name == "SVC (Classifier)":
        model = SVC()
    elif model_name == "SVR":
        model = SVR()
    
    return model_name, model

def display_predictions(y_test, y_pred):
    st.write("### Actual vs Predictions")
    results = pd.DataFrame({"Actual values": y_test, "Predictions": y_pred})
    st.write(results.head())

# Ensure the directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def save_model(model, model_name):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    if not model_name.endswith('.pkl'):
        model_name += '.pkl'
    
    filename = os.path.join(MODEL_DIR, model_name)
    
    try:
        st.joblib.dump(model, filename)
        print(f"Model saved successfully at {filename}")
        return filename
    except Exception as e:
        print(f"Error saving model: {e}")
        return None

def load_model(model_name):
    filename = os.path.join(MODEL_DIR, model_name)
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        st.error(f"Model {model_name} does not exist.")
        return None

def list_saved_models():
    return [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]



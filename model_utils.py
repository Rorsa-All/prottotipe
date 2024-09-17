from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, mean_absolute_error
import streamlit as st
from sklearn.model_selection import train_test_split

def train_model(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred

def evaluate_regression(y_test, y_pred, model_name):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"### {model_name} Results")
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"R-squared (R²): {r2}")

def evaluate_classification(y_test, y_pred, model_name):
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    st.write(f"### {model_name} Results")
    st.write(f"Accuracy: {accuracy}")
    st.write("Classification Report:")
    st.text(report)

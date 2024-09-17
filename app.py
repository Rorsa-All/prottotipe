import streamlit as st
import pandas as pd
from utils import load_data, select_features, select_model, display_predictions, save_model, load_model, list_saved_models
from model_utils import train_model, evaluate_regression, evaluate_classification

def main():
    st.title("Model Training, Saving, and Predictions App")

    # Load data
    data = load_data(key="upload_data")
    model_choice = st.radio("Choose an action:", ["Train New Model", "Use Saved Model"])
    
    if data is not None and model_choice == "Train New Model":
        # Select model and features
        model_name, model = select_model()
        target_column, feature_columns = select_features(data)
        
        if target_column and feature_columns:
            X = data[feature_columns]
            y = data[target_column]
            
            # Button to train model
            if st.button("Train Model"):
                st.write(f"### Training {model_name}")
                model, X_test, y_test, y_pred = train_model(model, X, y)
                
                # Display results and metrics
                display_predictions(y_test, y_pred)
                
                # Evaluate model
                if model_name in ["Linear Regression", "Random Forest Regressor", "SVR"]:
                    evaluate_regression(y_test, y_pred, model_name)
                else:
                    evaluate_classification(y_test, y_pred, model_name)
                
                # Save model
                model_filename = st.text_input("Enter a name to save your model (e.g., 'my_model.pkl'):")
                if st.button("Save Model"):
                    if model_filename:
                        if not model_filename.endswith('.pkl'):
                            model_filename += '.pkl'
                        save_model(model, model_filename)
                        st.success(f"Model saved as {model_filename}")
                    else:
                        st.error("Please provide a valid model name.")
    
    elif model_choice == "Use Saved Model":
        st.header("Load and Use Saved Model")
        saved_models = list_saved_models()
        model_file = st.selectbox("Select a saved model:", saved_models)
        
        if model_file:
            model = load_model(model_file)
            
            if model:
                # Select new data for prediction
                new_data = load_data(key="predict_data")
                if new_data is not None:
                    new_feature_columns = st.multiselect("Select the feature columns for prediction:", new_data.columns)
                    
                    if new_feature_columns:
                        X_new = new_data[new_feature_columns]
                        try:
                            predictions = model.predict(X_new)
                            st.write("### Predictions")
                            st.write(predictions)
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()

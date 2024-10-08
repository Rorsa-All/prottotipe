import streamlit as st
import pandas as pd
from utils import load_data, select_features, select_model, display_predictions, save_model, load_model, list_saved_models
from model_utils import train_model, evaluate_regression, evaluate_classification
import os

MODEL_DIR = 'saved_models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def main():
    st.title("Model Training, Saving, and Predictions App")

    if 'model_trained' not in st.session_state:
        st.session_state['model_trained'] = False

    if 'trained_model' not in st.session_state:
        st.session_state['trained_model'] = None
    
    if 'X_test' not in st.session_state:
        st.session_state['X_test'] = None

    if 'y_test' not in st.session_state:
        st.session_state['y_test'] = None

    if 'y_pred' not in st.session_state:
        st.session_state['y_pred'] = None

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
                
                st.session_state['trained_model'] = model
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred
                st.session_state['model_trained'] = True
                
                # Display results and metrics
                display_predictions(y_test, y_pred)
                
                # Evaluate model
                if model_name in ["Linear Regression", "Random Forest Regressor", "SVR"]:
                    evaluate_regression(y_test, y_pred, model_name)
                else:
                    evaluate_classification(y_test, y_pred, model_name)

    if st.session_state['model_trained']:
        model_filename = st.text_input("Enter a name to save your model (e.g., 'my_model.pkl'):")
        
        if st.button('Save Model'):
            if model_filename:
                if not model_filename.endswith('.pkl'):
                    model_filename += '.pkl'
                save_model(st.session_state['trained_model'], model_filename)
                st.success(f"Model saved as {model_filename}")
                st.session_state['model_saved'] = True
            else:
                st.error("Please provide a valid model name.")
    
    if model_choice == "Use Saved Model":
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
                        predictions = model.predict(X_new)    
                        if st.button('Predict data'):
                            try:

                                new_data['Predictions'] = predictions

                                uploaded_file = st.session_state.get("predict_data_file_name", "data")
                                output_filename = f"{os.path.splitext(uploaded_file)[0]}_pred.csv"
                            

                                new_data.to_csv(output_filename, index=False)
                                st.success(f"Data with predictions saved as {output_filename}")
                                st.write(new_data.head()) 
                            #    st.write("### Predictions")
                            #    st.write(predictions)
                        
                            except Exception as e:
                                st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()

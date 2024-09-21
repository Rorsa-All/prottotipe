import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR, SVC

MODEL_DIR = 'saved_models'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Функция для сохранения модели
def save_model(model, model_name):
    filename = os.path.join(MODEL_DIR, model_name)
    try:
        joblib.dump(model, filename)
        st.success(f"Model saved successfully as {model_name}")
        return filename
    except Exception as e:
        st.error(f"Error saving model: {e}")
        return None

# Загружаем сохраненные модели
def list_saved_models():
    return [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]

# Основная функция
def main():
    st.title("Model Training, Saving, and Predictions App")

    # Сохраняем состояние через st.session_state
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    # Действие пользователя: обучение новой модели или использование сохраненной
    model_choice = st.radio("Choose an action:", ["Train New Model", "Use Saved Model"])

    # Обучение новой модели
    if model_choice == "Train New Model":
        # Ввод имени модели
        model_filename = st.text_input("Enter a name to save your model (e.g., 'my_model.pkl'):")

        if st.button("Train and Save Model"):
            # Выбираем модель (например, Linear Regression)
            model = LinearRegression()
            X = pd.DataFrame([[1, 2], [2, 3], [3, 4]], columns=['feature1', 'feature2'])
            y = [5, 7, 9]
            
            # Обучаем модель
            model.fit(X, y)
            
            # Сохраняем модель в session_state
            st.session_state.model = model

            # Сохраняем модель на диск
            if model_filename:
                if not model_filename.endswith('.pkl'):
                    model_filename += '.pkl'
                save_model(st.session_state.model, model_filename)
            else:
                st.error("Please provide a valid model name.")

    # Использование сохраненной модели
    elif model_choice == "Use Saved Model":
        saved_models = list_saved_models()
        model_file = st.selectbox("Select a saved model:", saved_models)
        
        if st.button("Load Model"):
            # Загружаем модель
            if model_file:
                model_path = os.path.join(MODEL_DIR, model_file)
                model = joblib.load(model_path)
                st.session_state.model = model
                st.success(f"Model {model_file} loaded successfully.")

if __name__ == "__main__":
    main()

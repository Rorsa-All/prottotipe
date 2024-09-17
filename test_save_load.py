import streamlit as st
import joblib
import os
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Путь к директории для сохранения моделей
MODEL_DIR = 'saved_models'

# Убедитесь, что директория существует
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def save_model(model, model_name):
    if not model_name.endswith('.pkl'):
        model_name += '.pkl'
    filename = os.path.join(MODEL_DIR, model_name)
    st.write(f"Saving model to: {filename}")  # Debug statement
    try:
        joblib.dump(model, filename)
        st.success(f"Model successfully saved as {filename}")
        return filename
    except Exception as e:
        st.error(f"Error saving model: {e}")
        st.write(f"Exception details: {e}")  # Debug statement
    return None



def load_model(model_name):
    filename = os.path.join(MODEL_DIR, model_name)
    if os.path.exists(filename):
        st.write(f"Loading model from: {filename}")  # Для отладки
        try:
            return joblib.load(filename)
        except Exception as e:
            st.error(f"Error loading model: {e}")
    else:
        st.error(f"Model {model_name} does not exist.")
    return None

def main():
    st.title("Model Save and Load Test")

    # Создание и обучение модели
    st.header("Create and Train Model")
    if st.button("Create and Train Model"):
        # Загружаем данные
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Создаем и обучаем модель
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        # Проверяем точность модели
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model accuracy: {accuracy:.2f}")

        # Сохранение модели
        model_name = st.text_input("Enter a name to save your model (e.g., 'my_model.pkl'):")
        if st.button("Save Model"):
            if model_name:
                save_model(model, model_name)
            else:
                st.error("Please provide a valid model name.")

    # Загрузка и использование модели
    st.header("Load and Use Saved Model")
    saved_models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
    model_file = st.selectbox("Select a saved model:", saved_models)

    if model_file:
        model = load_model(model_file)
        if model:
            # Загрузка новых данных и предсказание
            st.header("Make Predictions with Loaded Model")
            iris = load_iris()
            new_data = st.text_input("Enter new data (comma-separated, e.g., '5.1,3.5,1.4,0.2'):")
            if new_data:
                try:
                    new_data = [float(x) for x in new_data.split(',')]
                    new_data = [new_data]  # Преобразуем в формат [ [features] ]
                    predictions = model.predict(new_data)
                    st.write(f"Predictions: {predictions}")
                except ValueError:
                    st.error("Invalid data format. Please enter numeric values separated by commas.")

if __name__ == "__main__":
    main()

import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Загружаем данные
iris = load_iris()
X, y = iris.data, iris.target

# Создаем модель
model = RandomForestClassifier()
model.fit(X, y)

# Функция для сохранения модели
def save_model(model, filename):
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    joblib.dump(model, f'saved_models/{filename}.pkl')

# Интерфейс Streamlit
st.title('Сохранение ML модели')
filename = st.text_input('Введите имя для сохранения модели')

if st.button('Сохранить модель'):
    save_model(model, filename)
    st.success(f'Модель сохранена как {filename}.pkl')

# Показать список сохраненных моделей
st.subheader('Сохраненные модели:')
saved_models = os.listdir('saved_models') if os.path.exists('saved_models') else []
for model_file in saved_models:
    st.write(model_file)

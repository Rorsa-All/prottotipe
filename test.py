import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Исходные данные
data = pd.read_csv('Housing.csv')

# Преобразование данных в DataFrame
df = pd.DataFrame(data)

# Выделение целевой переменной
target = df['price']

# Примеры наборов признаков для построения разных моделей
feature_sets = [
    ['area', 'bedrooms', 'bathrooms'],  # Модель 1: Площадь, спальни, ванные
    ['area', 'stories', 'parking'],  # Модель 2: Площадь, этажи, парковка
    ['area', 'bedrooms', 'bathrooms', 'stories'],  # Модель 3: Площадь, спальни, ванные, этажи
    ['bedrooms', 'stories', 'parking'],  # Модель 4: Спальни, этажи, парковка
    ['area', 'bedrooms', 'stories', 'bathrooms', 'parking']  # Модель 5: Площадь, спальни, этажи, ванные, парковка
]

# Разделение данных на тренировочные и тестовые выборки
X_train, X_test, y_train, y_test = train_test_split(df.drop('price', axis=1), target, test_size=0.2, random_state=42)

# Сохранение каждой модели
for i, feature_set in enumerate(feature_sets, 1):
    # Выбор только нужных признаков
    X_train_subset = X_train[feature_set]
    X_test_subset = X_test[feature_set]

    # Создание и обучение модели
    model = LinearRegression()
    model.fit(X_train_subset, y_train)

    # Сохранение модели в файл .pkl
    with open(f'model_{i}.pkl', 'wb') as f:
        pickle.dump(model, f)

    print(f'Модель {i} сохранена как model_{i}.pkl')

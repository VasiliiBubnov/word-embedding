

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Чтение файла Excel и преобразование его в датафрейм
df = pd.read_excel(r'C:\Users\ext17\Downloads\jj.xlsx', engine='openpyxl')

# Разделение датафрейма на обучающий и тестовый наборы
train_df = df.iloc[:40, :]
test_df = df.iloc[40:, :]

# Разделение данных на входные переменные (X) и целевую переменную (y)
X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

# Определение диапазонов гиперпараметров для перебора
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Создание базовой модели
rf = RandomForestClassifier(random_state=42)

# Создание объекта GridSearchCV с указанными параметрами
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Обучение модели с перебором гиперпараметров
grid_search.fit(X_train, y_train)

# Вывод наилучших параметров
print("Best parameters found:")
print(grid_search.best_params_)

# Создание и обучение модели с оптимальными гиперпараметрами
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# Предсказание значений на тестовом наборе
y_pred = best_rf.predict(X_test)

# Вывод метрик качества модели
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

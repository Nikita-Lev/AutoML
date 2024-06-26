# AutoML
Автоматический подбор ML | DL модели для ML задач: регрессия, *классификация, кластеризация

Построение модели состоит из четырёх шагов:

```python
from AutoML import autoML
model = autoML(task) # task = 'Regression' | 'Classification' | 'Clusterization'
model.fit(data, target)
model.predict(X_test)
```


Используемые модели:
| Регрессия         | Классификация       | Кластеризация             |
| -------------     | -------------       | -------------             |
| Linear Regression | Logistic Regression | DBSCAN (авто)             |
| AutoKeras         | SVM                 | K-Means (число кластеров) |
| SVM               | XGBoost             |                           |
| Random Forest     | CatBoost            |                           |
| CatBoost          | Random Forest       |                           |
| XGBoost           | AutoKeras           |                           |
| Lasso Regression  |                     |                           |
| Ridge Regression  |                     |                           |

Предобработка данных включает следующие этапы:
1) Удаление пропусков в target
2) Удаление выбросов в target методом IQR (только для регрессии)
3) Удаление признаков с пропусками больше 20%
4) Заполнение пропусков медианой по группам целевой переменной
5) Удаление признаков с одинаковыми значениями больше 90%
6) Стандартизация вещественных признаков
7) One-Hot кодирование категориальных признаков
8) Применение PCA при соотношении n_samples : n_features < 10 : 1

*Бинарная

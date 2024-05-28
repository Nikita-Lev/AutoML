# AutoML
Автоматический подбор ML модели для различных ML задач

На данный момент проект в разработке, поддерживаются следующие задачи: регрессия, *классификация

Построение модели состоит из трёх шагов:

0) from AutoML import autoML
1) model = autoML('Regression')
2) model.fit(data, target)
3) model.predict(X_test)


Используемые модели:
| Регрессия         | Классификация       | _Скоро..._          |
| -------------     | -------------       | -------------       |
| Linear Regression | Logistic Regression |                     |
| Lasso Regression  | SVM                 |                     |
| Ridge Regression  | XGBoost             |                     |
| SVM               | Random Forest       |                     |
| XGBoost           | _скоро..._          |                     |
| Random Forest     |                     |                     |
| _скоро..._        |                     |                     |


* Бинарная

# AutoML
Автоматический подбор ML модели для различных ML задач

На данный момент проект в разработке, поддерживаются следующие задачи: *регрессия

Построение модели состоит из трёх шагов:

0) from AutoML import autoML
1) model = autoML('Regression')
2) model.fit(data, target)
3) model.predict(X_test)

* Для числовых признаков

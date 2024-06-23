# coding: utf-8

import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

class Model:
    def fit(self, data, target, metrics = [], preprocessing = True):#, model = None, **kwargs):
        '''
        Подготовка данных, обучение моделей
        
        Parameters
        ----------
        data : dict or array_like or pd.DataFrame
            Данные в формате словаря {Признак: список значений} или массива размерности (n_samples, n_features)
        target : str or array_like
            Название целевой переменнной, если содержится в data или явно массив длины n_samples
        metrics : array_like, default None
            Список метрик качества модели для вывода, доступные варианты: 'explained_variance', 'r2', 'max_error', 'neg_median_absolute_error', 'neg_mean_absolute_error', neg_mean_absolute_percentage_error', 'neg_mean_squared_log_error', 'neg_root_mean_squared_error', 'neg_mean_poisson_deviance', 'neg_mean_gamma_deviance', 'accuracy', 'top_k_accuracy', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted', 'balanced_accuracy', 'average_precision', 'neg_log_loss', 'neg_brier_score', 'adjusted_rand_score', 'rand_score', 'homogeneity_score', 'completeness_score', 'v_measure_score', 'mutual_info_score', 'adjusted_mutual_info_score', 'normalized_mutual_info_score', 'fowlkes_mallows_score', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples', 'jaccard_weighted'
        preprocessing : bool, default True
            Использование встроенной предобработки данных, по умолчанию да
        model : {'Linear Regression', ...}, default None
            Модель для выбора, по умолчанию автоподбор
        kwargs
            Параметры выбранной модели
        '''
        
        
        if type(data) != dict and type(data) != pd.DataFrame and type(target) == str:
            raise TypeError(f'Нельзя найти {target} в массиве data')
        
        self.data = pd.DataFrame(data)
        
        if type(target) != str:
            if len(target) != self.data.shape[0]:
                raise TypeError(f'n_samples = {self.data.shape[0]}, но длина target = {len(target)}')
            self.data['y'] = target
            target = 'y'
        
        elif target not in data.keys():
            raise ValueError(f'Нельзя найти {target} в массиве data')
        
        self.target = target
        self.metrics += np.array(metrics).tolist()
        
        self.steps = [] # Этапы работы с данными, шаги pipeline
        
        self.Prepare_data()
        
        # Подбор лучшей модели
        self.model = None
        best_score = -np.inf
        results = [] 
        
        for model_name in tqdm(self.models):
            self.pipeline = Pipeline(self.steps + [('', self.models[model_name][0])])
            
            # Подбор гиперпараметров
            if model_name != 'Auto Keras':
                self.searcher = GridSearchCV(self.pipeline, param_grid = self.models[model_name][1], cv = 5, n_jobs = -1, scoring = self.metrics, refit = self.metrics[0])
            else:
                self.searcher = self.pipeline
                
            self.searcher.fit(self.X, self.y)
            
            # Выделение метрик качества лучшей конфигурации модели
            if model_name != 'Auto Keras':
                scorings = pd.DataFrame(self.searcher.cv_results_)[list(map(lambda x : 'mean_test_' + x, self.metrics))]
                metrics_val = np.round(scorings.iloc[np.where(scorings == self.searcher.best_score_)[0][0]].values, 2).tolist()
            else:
                cv = cross_validate(self.searcher, self.X, self.y, cv = 5, scoring = self.metrics)
                metrics_val = np.round(pd.DataFrame(cv)[list(map(lambda x : 'test_' + x, self.metrics))].mean().values, 2).tolist()
                self.searcher.best_params_ = ''
                self.searcher.best_score_ = metrics_val[0]
                self.searcher.best_estimator_ = self.searcher

            results.append([model_name, self.searcher.best_params_, abs(self.searcher.best_score_)] + metrics_val[1:])
            
            if self.searcher.best_score_ > best_score:
                best_score = self.searcher.best_score_
                self.model = self.searcher.best_estimator_
                top_model_name = model_name
                
        display(pd.DataFrame(results, columns = ['Название модели', 'Конфигурация', 'Score'] + np.array(metrics).tolist()))
        print(f'\nЛучшая модель: {top_model_name}\nScore: {abs(best_score)}')
                    
    
    def Prepare_data(self):
        '''
        Подготовка данных для обучения моделей
        '''
        # Удаление пропусков в target
        df = self.data.dropna(subset = [self.target])
    
        # Удаление выбросов целевой переменной
        df = self.Remove_outliers(df)
        
        # Удаление признаков с пропусками > 20%
        df.dropna(thresh = df.shape[0] * 0.8, axis = 1, inplace = True)

        # Заполнение пропусков медианой по группам целевой переменной
        df = df.groupby(self.target).apply(lambda group: group.fillna(group.median(numeric_only = True)))
           
        # Выделение целевой переменной
        self.y = df[self.target]
        df = df.drop([self.target], axis = 1)
        
        # Удаление признаков с одинаковыми значениями > 90%
        dropped = df[df == df.mode().iloc[0]].count() > df.shape[0] * 0.9
        df = df[dropped[dropped == False].index]
        
        
        # Onehot кодирование категориальных признаков
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy = 'most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown = 'ignore', sparse = False))
        ])
        
        # Стандартизация вещественных признаков
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy = 'median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        
        
        self.X = df
        self.columns = self.X.columns
        
        self.steps.append(('preprocessing', preprocessor))
        
        # Применить метод главных компонент при избыточном числе признаков, получив соотношение n_samples : n_features = 10 : 1
        if self.X.shape[0] / self.X.shape[1] < 10:
            warnings.warn(f"Недостаточно данных для обучения модели. Соотношение данных к признакам меньше 10. Используется РСА",
            UserWarning)
            self.steps.append(('pca', PCA(n_components = int(self.X.shape[0] / 10))))

    def predict(self, X_test):# plot = False):
        '''
        Предсказания моделей по тестовой выборке
        
        Parameters
        ----------
        X_test : dict or array_like or pd.DataFrame
            Данные в формате словаря {Признак: список значений} или массива размерности (n_samples, n_features)
            n_features должно соответствовать n_features of data
        plot : bool, default False
            Построение графика прогнозов
        
        Returns
        -------
        array
            Список прогнозов
        '''
        
        if self.model:
            X_test = pd.DataFrame(X_test)[self.columns]
            return self.model.predict(X_test)
        
        raise ValueError('Модель отсутствует.')
            
    def Remove_outliers(self, df):
        '''
        Удаление выбросов целевой переменной методом межквартильного размаха
        '''
        q1, q3 = df[self.target].quantile([0.25, 0.75])
        IQR = q3 - q1
        return df[(df[self.target] > q1 - 1.5 * IQR) & (df[self.target] < q3 + 1.5 * IQR)]
    
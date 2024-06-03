# coding: utf-8

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import warnings

from itertools import product

class pDBSCAN(DBSCAN):
    '''
    DBSCAN с методом predict для новых данных
    '''
    def fit_predict(self, X, y=None, sample_weight=None):
        self.X = X
        return super().fit_predict(self.X)
    
    def predict(self, X_test):
        '''
        Предсказание меток кластеров для новых данных
        '''
        knn = KNeighborsClassifier(n_neighbors = 1)
        knn.fit(self.X, self.labels_)
        return knn.predict(X_test)
        

class Clusterizator():
    ''' Выделение кластеров в данных
    '''
        
    def fit(self, data, n_clusters = None, return_labels = False, preprocessing = True):
        '''
        Подготовка данных, обучение моделей
        
        Parameters
        ----------
        data : dict or array_like or pd.DataFrame
            Данные признаков и целевой переменной или только целевая переменная
        n_clusters : int, default None
            Количество кластеров, по умолчанию автоматически
        return_labels : bool, default False
            Возвращать метки кластеров для переданных данных, по умолчанию False
        preprocessing : bool, default True
            Использование встроенной предобработки данных, по умолчанию True
        '''
        
        self.data = pd.DataFrame(data)
        
        self.steps = [] # Этапы работы с данными, шаги pipeline
        
        if preprocessing:
            self.Prepare_data()
        else:
            self.X = self.data
        
        # При автоподборе кластеров используется DBSCAN, иначе K-means
        if n_clusters == None:
        
            param_grid = {
            'eps': np.arange(0.1, 1.5, 0.1),
            'min_samples': range(3, 10)
            }

            best_score = -1
            self.model = None
            for params in product(param_grid['eps'], param_grid['min_samples']):
                self.pipeline = Pipeline(self.steps + [('model', pDBSCAN(eps = params[0], min_samples = params[1]))])
                labels = self.pipeline.fit_predict(self.X)

                # Кластеров должно быть больше одного
                if len(set(labels)) > 1:
                    score = silhouette_score(self.X, labels)
                    if score > best_score:
                        best_score = score
                        self.model = self.pipeline['model']
                        
            results = ['DBSCAN']
        
        else:
            self.model = KMeans(n_clusters)
            labels = self.model.fit_predict(self.X)
            best_score = silhouette_score(self.X, labels)
            
            results = ['K-Means']
        results+= [self.model.get_params(), best_score]
        
        display(pd.DataFrame([results], columns = ['Название модели', 'Конфигурация', 'Score']))
        
        if return_labels: return self.model.labels_   
        
        
    def Prepare_data(self):
        '''
        Подготовка данных для обучения моделей
        '''
        df = self.data.copy()
        
        # Удаление признаков с пропусками > 20%
        df.dropna(thresh = df.shape[0] * 0.8, axis = 1, inplace = True)

        # Удаление признаков с одинаковыми значениями > 90%
        dropped = df[df == df.mode().iloc[0]].count() > df.shape[0] * 0.9
        df = df[dropped[dropped == False].index]
        
        # Onehot кодирование категориальных признаков
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy = 'most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
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
        
    
    def predict(self, X_test):
        '''
        Определение меток кластеров для новых данных
        '''
        if self.model:
            X_test = pd.DataFrame(X_test)[self.columns]

            return self.model.predict(X_test)
        
        raise ValueError('Модель отсутствует.')
          


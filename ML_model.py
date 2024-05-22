#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from tqdm import tqdm

class Model:
    def fit(self, data, target):#, metrics = None, model = None, **kwargs):
        '''
        Подготовка данных, обучение моделей
        
        Parameters
        ----------
        data : dict or array_like
            Данные в формате словаря {Признак: список значений} или массива размерности (n_samples, n_features)
        target : str or array_like
            Название целевой переменнной, если содержится в data или явно массив длины n_samples
        metrics : array_like, default = None
            Список метрик качества модели для вывода
        model : {'Linear Regression', ...}, default = None
            Модель для выбора, по умолчанию автоподбор
        kwargs
            Параметры выбранной модели
        '''
        
        if type(data) != dict and type(target) == str:
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
        
        self.steps = [] # Этапы работы с данными, шаги pipeline
        
        self.Prepare_data()
        
        # Подбор лучшей модели
        self.model = None
        best_score = -np.inf
        results = [] 
        for model_name in tqdm(self.models):
            self.pipeline = Pipeline(self.steps + [('', self.models[model_name][0])])
            # Подбор гиперпараметров
            self.searcher = GridSearchCV(self.pipeline, param_grid = self.models[model_name][1], cv = 5, n_jobs = -1,
                                         scoring = self.metric)
            self.searcher.fit(self.X, self.y)
            
            results.append([model_name, self.searcher.best_params_, abs(self.searcher.best_score_)])
            
            if self.searcher.best_score_ > best_score:
                best_score = self.searcher.best_score_
                self.model = self.searcher.best_estimator_
                
        
        display(pd.DataFrame(results, columns = ['Название модели', 'Конфигурация', 'MSE']))
        print(best_score, self.model)
        
        
    def predict(self, X_test):#, metrics = None, plot = False):
        '''
        Предсказания моделей по тестовой выборке
        
        Parameters
        ----------
        X_test : dict or array_like
            Данные в формате словаря {Признак: список значений} или массива размерности (n_samples, n_features)
            n_features должно соответствовать n_features of data
        metrics : array_like, default = None
            Список метрик качества моделей для вывода
        plot : bool, default = False
            Построение графика прогнозов
        '''
        
        if self.model:
            return self.model.predict(X_test)
        
        raise ValueError('Модель отсутствует.')
            
    
    
    def Prepare_data(self):
        '''
        Подготовка данных для обучения моделей
        '''
        self.X = self.data.drop([self.target], axis = 1)
        self.y = self.data[self.target]

        
        self.steps.append(('scaler', StandardScaler()))

